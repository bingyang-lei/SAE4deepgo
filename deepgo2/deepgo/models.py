from .base import BaseDeepGOModel, Residual, MLPBlock
from torch import nn
import torch as th

from dgl.nn import GATConv


class DeepGOModel(BaseDeepGOModel):
    """
    DeepGO model with ElEmbeddings loss functions.

    Args:
        input_length (int): The number of input features
        nb_gos (int): The number of Gene Ontology (GO) classes to predict
        nb_zero_gos (int): The number of GO classes without training annotations
        nb_rels (int): The number of relations in GO axioms
        device (string): The compute device (cpu:0 or cuda:0)
        hidden_dim (int): The hidden dimension for an MLP
        embed_dim (int): Embedding dimension for GO classes and relations
        margin (float): The margin parameter of ELEmbedding method
    """

    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        # MLP Layers to project the input protein
        net = []
        net.append(MLPBlock(input_length, embed_dim))
        net.append(Residual(MLPBlock(embed_dim, embed_dim)))
        self.net = nn.Sequential(*net)

    def forward(self, features):
        """
        Forward pass of the DeepGO Model.
        
        Args:
            features (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predictions after passing through DeepGOModel layers.
        """
        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits

    def forward_nf4(self, features):
        """
        Forward pass with Normal Form four.
        
        Args:
            features (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predictions after passing through DeepGOModel layers.
        """
        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = x.unsqueeze(dim=1).expand(-1, self.nb_gos, -1)
        dst = th.linalg.norm(x - hasFuncGO, dim=2)
        logits = th.relu(dst - go_rad - self.margin)
        return logits



class DeepGOGATModel(BaseDeepGOModel):
    """
    DeepGOGAT model with ElEmbeddings loss functions.

    Args:
        input_length (int): The number of input features
        nb_gos (int): The number of Gene Ontology (GO) classes to predict
        nb_zero_gos (int): The number of GO classes without training annotations
        nb_rels (int): The number of relations in GO axioms
        device (string): The compute device (cpu:0 or gpu:0)
        hidden_dim (int): The hidden dimension for an MLP
        embed_dim (int): Embedding dimension for GO classes and relations
        margin (float): The margin parameter of ELEmbedding method
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=1024, embed_dim=1024, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)

        self.net1 = MLPBlock(input_length, hidden_dim)
        self.conv1 = GATConv(hidden_dim, embed_dim, num_heads=1)
        self.net2 = nn.Sequential(
            nn.Linear(embed_dim, nb_gos),
            nn.Sigmoid())

    
    def forward(self, input_nodes, output_nodes, blocks):
        """
        Forward pass of the DeepGOGAT Model.
        
        Args:
            input_nodes (torch.Tensor): Input tensor.
            output_nodes (torch.Tensor): Input tensor.
            blocks (graphs): DGL Graphs
        Returns:
            torch.Tensor: Predictions after passing through DeepGOModel layers.
        """
        g1 = blocks[0]
        features = g1.ndata['feat']['_N']
        x = self.net1(features)
        x = self.conv1(g1, x).squeeze(dim=1)

        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))

        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits


    
class MLPModel(nn.Module):
    """
    Baseline MLP model with two fully connected layers with residual connection
    """
    
    def __init__(self, input_length, nb_gos, device, nodes=[1024,]):
        super().__init__()
        self.nb_gos = nb_gos
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)


import torch.nn.functional as F
from dataclasses import dataclass
import torch
DTYPES = {"fp32": th.float32, "fp16": th.float16, "bf16": th.bfloat16}

@dataclass
class AutoEncoderConfig:
    '''Class for storing configuration parameters for the autoencoder'''
    seed: int = 42
    batch_size: int = 32
    buffer_mult: int = 384
    epochs: int = 10
    lr: float = 1e-3
    num_tokens: int = int(2e9)
    l1_coeff: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    dict_mult: int = 8
    seq_len: int = 128
    d_mlp: int = 2560 
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_batch_size: int = 64

    def __post_init__(self):
        '''Using kwargs, so that we can pass in a dict of parameters which might be
        a superset of the above, without error.'''
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.dtype = DTYPES[self.enc_dtype]
        self.d_hidden = 21356 # 暂时只使用bp的terms


class SAEModel(BaseDeepGOModel):
    """
    SAEModel with ElEmbeddings loss functions, using esm2/protgpt2 as input features.
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, cfg: AutoEncoderConfig,hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        self.cfg = cfg
        th.manual_seed(cfg.seed)

        # W_enc has shape (d_mlp, d_encoder), where d_encoder is a multiple of d_mlp (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, cfg.d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to("cuda")

    def forward(self, feature: torch.Tensor):
        feature_cent = feature - self.b_dec
        acts = F.relu(feature_cent @ self.W_enc + self.b_enc)
        prediction = F.sigmoid(feature_cent @ self.W_enc + self.b_enc)
        feature_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (feature_reconstruct.float() - feature.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.cfg.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, feature_reconstruct, acts, l2_loss, l1_loss, prediction
    
class GateSAEModel(BaseDeepGOModel):
    """
    GateSAEModel with ElEmbeddings loss functions, using esm2/protgpt2 as input features.
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, cfg: AutoEncoderConfig,hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        self.cfg = cfg
        th.manual_seed(cfg.seed)

        self.W_mag = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, cfg.d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_mag = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.W_gate = self.W_mag * self.r_mag
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))

    def forward(self, feature: torch.Tensor):
        a = 1
    # def loss(self, feature: torch.Tensor):