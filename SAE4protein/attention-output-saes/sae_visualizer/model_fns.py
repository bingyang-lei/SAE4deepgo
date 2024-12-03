from transformer_lens import utils
import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
import tqdm.notebook as tqdm
from dataclasses import dataclass


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

@dataclass
class AutoEncoderConfig:
    '''Class for storing configuration parameters for the autoencoder'''
    # 随机种子，用于确保实验的可重复性
    seed: int = 42
    # 训练时的批量大小
    batch_size: int = 32
    # 缓冲区大小的倍数，用于数据加载
    buffer_mult: int = 384
    # 训练的轮数
    epochs: int = 10
    # 学习率
    lr: float = 1e-3
    # 训练过程中使用的令牌数量
    num_tokens: int = int(2e9)
    # L1 正则化系数
    l1_coeff: float = 3e-4
    # Adam 优化器的 beta1 参数
    beta1: float = 0.9
    # Adam 优化器的 beta2 参数
    beta2: float = 0.99
    # 字典大小的倍数，用于确定隐藏层的维度
    dict_mult: int = 8
    # 序列长度
    seq_len: int = 128
    # 多层感知机（MLP）的维度
    d_mlp: int = 2048
    # 编码器的数据类型（如 "fp32" 表示 32 位浮点数）
    enc_dtype: str = "fp32"
    # 是否移除罕见的方向
    remove_rare_dir: bool = False
    # 模型的批量大小
    model_batch_size: int = 64

    def __post_init__(self):
        '''Using kwargs, so that we can pass in a dict of parameters which might be
        a superset of the above, without error.'''       
        # 计算缓冲区大小，等于 batch_size 和 buffer_mult 的乘积
        self.buffer_size = self.batch_size * self.buffer_mult       
        # 计算缓冲区批次数量，等于 buffer_size 除以 seq_len
        self.buffer_batches = self.buffer_size // self.seq_len        
        # 根据 enc_dtype 属性从 DTYPES 字典中获取数据类型
        self.dtype = DTYPES[self.enc_dtype]        
        # 计算隐藏层的维度，等于 d_mlp 和 dict_mult 的乘积
        self.d_hidden = self.d_mlp * self.dict_mult



class AutoEncoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # W_enc has shape (d_mlp, d_encoder), where d_encoder is a multiple of d_mlp (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, cfg.d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to("cuda")

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.cfg.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    @classmethod
    def load_from_hf(cls, version, verbose=False):
        """
        Loads the saved autoencoder from HuggingFace. 

        Note, this is a classmethod, because we'll be using it as `auto_encoder = AutoEncoder.load_from_hf("run1")`

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """

        assert version in ["run1", "run2"]
        version = 25 if version=="run1" else 47

        cfg: dict = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        # There are some unnecessary params in cfg cause they're defined in post_init for config dataclass; we remove them
        cfg.pop("buffer_batches", None)
        cfg.pop("buffer_size", None)

        if verbose: pprint.pprint(cfg)
        cfg = AutoEncoderConfig(**cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self

    def __repr__(self):
        return f"AutoEncoder(d_mlp={self.cfg.d_mlp}, dict_mult={self.cfg.dict_mult})"