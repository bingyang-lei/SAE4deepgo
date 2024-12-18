from .base import BaseDeepGOModel, Residual, MLPBlock
from torch import nn
import torch as th

from dgl.nn import GATConv
import torch.nn.functional as F
import time
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
    
    # 不用的都注释掉
    seed: int = 42
    batch_size: int = 32
    # buffer_mult: int = 384
    epochs: int = 10
    lr: float = 5e-4  # 全部学习率
    # num_tokens: int = int(2e9)
    l1_coeff: float = 3e-4
    # beta1: float = 0.9
    # beta2: float = 0.99
    dict_mult: int = 8
    seq_len: int = 128
    d_mlp: int = 2560 
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_batch_size: int = 64

    def __post_init__(self):
        '''Using kwargs, so that we can pass in a dict of parameters which might be
        a superset of the above, without error.'''
        # self.buffer_size = self.batch_size * self.buffer_mult
        # self.buffer_batches = self.buffer_size // self.seq_len
        
        self.dtype = DTYPES[self.enc_dtype]
        self.d_hidden = 21356 # 暂时只使用bp的terms


class SAEModel(BaseDeepGOModel):
    """
    SAEModel with ElEmbeddings loss functions, using esm2/protgpt2 as input features.
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, cfg , hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        self.cfg = cfg
        th.manual_seed(cfg.seed)

        # W_enc has shape (d_mlp, d_encoder), where d_encoder is a multiple of d_mlp (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, cfg.d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True) # 这里是干啥的没太明白

        self.to("cuda:0")

    def forward(self, feature: torch.Tensor):
        feature_cent = feature - self.b_dec
        acts = F.relu(feature_cent @ self.W_enc + self.b_enc)
        prediction = F.sigmoid(feature_cent @ self.W_enc + self.b_enc)
        feature_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (feature_reconstruct.float() - feature.float()).pow(2).sum(-1).mean(0) # 回头检查一下这个和下面那个写法一样不
        l1_loss = self.cfg.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, feature_reconstruct, acts, l2_loss, l1_loss, prediction
    
class GateSAEModel(BaseDeepGOModel):
    """
    GateSAEModel with ElEmbeddings loss functions, using esm2/protgpt2 as input features.
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, cfg ,hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        self.cfg = cfg
        th.manual_seed(cfg.seed)

        self.W_mag = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, cfg.d_hidden, dtype=cfg.dtype)))
        # nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_mag = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype)) # 后面可以尝试用kaiming初始化
        # self.r_mag = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, dtype=cfg.dtype)))

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        
        self.W_dec = nn.Parameter(self.W_mag.T.clone())

    def forward(self, x: torch.Tensor, Y: torch.Tensor):

        W_gate = self.W_mag * torch.exp(self.r_mag) 
        
        # Apply pre-encoder bias
        x_center = x - self.b_dec
        
        # Gating encoder (estimates which features are active)
        # 这里要替换成标签Y来代表激活
        # active_features = (x_center @ self.W_gate + self.b_gate) > 0
        pi_gate = x_center @ W_gate + self.b_gate
        # Magnitudes encoder (estimates active features’ magnitudes)
        feature_magnitudes = F.relu(x_center @ self.W_mag + self.b_mag)
        
        # 重建后的激活
        # reconstruction = (Y * feature_magnitudes) @ self.W_dec + self.b_dec
        
        # 暂时不使用label，而是使用gate进行重构
        f_gate = (pi_gate > 0).float()
        reconstruction = (f_gate * feature_magnitudes) @ self.W_dec + self.b_dec
        # 只进行重构
        # reconstruction = ( feature_magnitudes) @ self.W_dec + self.b_dec
        reconstruction_loss = torch.sum((reconstruction - x)**2, dim=-1).mean()

        predition = F.sigmoid(pi_gate)
        
        
        # 计算辅助误差
        with torch.no_grad():
            W_dec_stop_grad = self.W_dec.detach()
            b_dec_stop_grad = self.b_dec.detach()

        via_reconstruction = (F.relu(pi_gate) @ W_dec_stop_grad + b_dec_stop_grad)
        via_reconstruction_loss = torch.sum((via_reconstruction - x)**2, dim=-1).mean()
        
        return reconstruction,reconstruction_loss, predition, via_reconstruction_loss


class GateSAEModel2(BaseDeepGOModel):
    """
    GateSAEModel with ElEmbeddings loss functions, using esm2/protgpt2 as input features.
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, cfg ,hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        self.cfg = cfg
        th.manual_seed(cfg.seed)
        extra_ratio = cfg.extra_ratio
        self.all_d_hidden = int(cfg.d_hidden*(1+extra_ratio))
        
        # self.W_mag = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, self.all_d_hidden, dtype=cfg.dtype)))
        self.W_mag = nn.Parameter(torch.zeros(cfg.d_mlp, self.all_d_hidden, dtype=cfg.dtype))
        print('self.W_mag.shape: ',self.W_mag.shape)
        
        # nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_mag = nn.Parameter(torch.zeros(self.all_d_hidden, dtype=cfg.dtype))
        # gate只保留一部分
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype)) # 后面可以尝试用kaiming初始化
        # self.r_mag = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, dtype=cfg.dtype)))

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        
        self.W_dec = nn.Parameter(self.W_mag.T.clone())
        # self.W_dec = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(self.all_d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        
    def forward(self, x: torch.Tensor, Y: torch.Tensor):
        
        # (d_mlp, d_hidden)
        W_gate = self.W_mag[:, :self.cfg.d_hidden] * torch.exp(self.r_mag) 
        
        # Apply pre-encoder bias
        x_center = x - self.b_dec
        
        # Gating encoder (estimates which features are active)
        # 这里要替换成标签Y来代表激活
        # active_features = (x_center @ self.W_gate + self.b_gate) > 0
        
        # pi_gate (d_hidden)
        pi_gate = x_center @ W_gate + self.b_gate
        
        # Magnitudes encoder (estimates active features’ magnitudes)
        feature_magnitudes = F.relu(x_center @ self.W_mag + self.b_mag)
        
        # 重建后的激活
        # reconstruction = (Y * feature_magnitudes) @ self.W_dec + self.b_dec
        
        # 暂时不使用label，而是使用gate进行重构
        mask_vector = torch.ones(self.all_d_hidden - self.cfg.d_hidden, device=pi_gate.device).expand(x_center.shape[0], -1)
        f_gate = (torch.cat([pi_gate, mask_vector], dim=-1)> 0).float()
        
        reconstruction = (f_gate * feature_magnitudes) @ self.W_dec + self.b_dec
        # reconstruction = (feature_magnitudes) @ self.W_dec + self.b_dec
        
        # test=torch.sum((reconstruction - x)**2, dim=-1)
        reconstruction_loss = torch.sum((reconstruction - x)**2, dim=-1).mean()
        
        predition = F.sigmoid(pi_gate)
        
        
        # 计算辅助误差
        with torch.no_grad():
            W_dec_stop_grad = self.W_dec.detach()
            b_dec_stop_grad = self.b_dec.detach()

        W_part = self.W_mag[:, self.cfg.d_hidden:]
        part_act = F.relu(x_center @ W_part + self.b_mag[self.cfg.d_hidden:])
        
        via_reconstruction = (torch.cat([F.relu(pi_gate), part_act ], dim=-1)@ W_dec_stop_grad + b_dec_stop_grad)
        via_reconstruction_loss = torch.sum((via_reconstruction - x)**2, dim=-1).mean()
        # via_reconstruction_loss = 0
        return reconstruction,reconstruction_loss, predition, via_reconstruction_loss

# %%


class GateSAEModel3(BaseDeepGOModel):
    """
    GateSAEModel with ElEmbeddings loss functions, using esm2/protgpt2 as input features.
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, cfg ,hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        self.cfg = cfg
        th.manual_seed(cfg.seed)
        extra_ratio = cfg.extra_ratio
        # 
        self.pred_hidden = cfg.d_hidden
        self.rec_hidden = int(cfg.d_hidden*extra_ratio)
        self.all_d_hidden = self.pred_hidden + self.rec_hidden
        
        # self.W_mag = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, self.all_d_hidden, dtype=cfg.dtype)))
        # todo 初始化重写
        # nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_mag = nn.Parameter(torch.zeros(self.all_d_hidden, dtype=cfg.dtype))
        # gate只保留一部分
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype)) # 后面可以尝试用kaiming初始化
        # self.r_mag = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, dtype=cfg.dtype)))

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        
        time1 = time.time()
        # self.W_mag = nn.Parameter(self.init_W_mag(cfg))
        self.W_mag = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(cfg.d_mlp, self.all_d_hidden, dtype=cfg.dtype)))
        
        time2 = time.time()
        print(time2-time1)
        self.W_dec = nn.Parameter(self.W_mag.T.clone())
        # self.W_dec = nn.Parameter(th.nn.init.kaiming_uniform_(th.empty(self.all_d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
    
    def init_W_mag(self, cfg):
        # Initialize the two parts
        W_part1 = torch.nn.init.kaiming_uniform_(
            torch.empty(cfg.d_mlp, self.pred_hidden, dtype=cfg.dtype)
        )
        W_part2 = torch.nn.init.kaiming_uniform_(
            torch.empty(cfg.d_mlp, self.rec_hidden, dtype=cfg.dtype)
        )

        # Orthogonalize W_part2 with respect to W_part1
        Q, _ = torch.linalg.qr(W_part1)  # QR decomposition to get orthonormal basis
        W_part2_projected = W_part2 - Q @ (Q.T @ W_part2)  # Remove components in W_part1's span

        # Normalize W_part2 to ensure numerical stability
        W_part2_orthogonal = F.normalize(W_part2_projected, dim=0)

        # Concatenate the two parts
        W_mag = torch.cat([W_part1, W_part2_orthogonal], dim=1)
        return W_mag
    
    def forward(self, x: torch.Tensor, Y: torch.Tensor):
        # time1 = time.time()
        # 提取pred部分
        W_gate = self.W_mag[:, :self.pred_hidden] * torch.exp(self.r_mag) 
        x_center = x - self.b_dec
        pi_gate = x_center @ W_gate + self.b_gate
        
        feature_magnitudes = F.relu(x_center @ self.W_mag + self.b_mag)
        
        mask_vector = torch.ones(self.rec_hidden, device=pi_gate.device).expand(x_center.shape[0], -1)
        f_gate = (torch.cat([pi_gate, mask_vector], dim=-1)> 0).float()
        
        reconstruction = (f_gate * feature_magnitudes) @ self.W_dec + self.b_dec
        # reconstruction = (feature_magnitudes) @ self.W_dec + self.b_dec
        
        # test=torch.sum((reconstruction - x)**2, dim=-1)
        reconstruction_loss = torch.sum((reconstruction - x)**2, dim=-1).mean()
        
        predition = F.sigmoid(pi_gate)



        W_pred_dir = self.W_mag[:, :self.pred_hidden]
        W_rec_dir = self.W_mag[:, self.pred_hidden:]
        W_pred_dir = W_pred_dir/(torch.norm(W_pred_dir, dim=1, keepdim=True) + 1e-8)
        W_rec_dir = W_rec_dir / (torch.norm(W_rec_dir, dim=1, keepdim=True) + 1e-8)
        dot = torch.matmul(W_pred_dir.T, W_rec_dir)

        penalty_loss = torch.mean(torch.sum(dot**2))
        # penalty_loss = torch.mean(torch.sum(dot)**2)

        

        #penalty_loss = reconstruction_loss
        # time2 = time.time()
        # print(time2-time1)
        return reconstruction,reconstruction_loss, predition, penalty_loss


if __name__ == '__main__':
    from torch import nn
    import torch
