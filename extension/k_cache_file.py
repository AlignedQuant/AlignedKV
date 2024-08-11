import torch
import math
from extension.KV_cache_cpp_extention import cuda_module

bits_in_a_channel = 256
column_block = bits_in_a_channel // 4


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class K_Cache_Class(torch.nn.Module):
    def __init__(self, bsz: int, max_seq_len: int, n_local_kv_heads: int, head_dim: int, layer_idx: int,
                 device: str | torch.device = "cuda:0", register_buffer: bool = False):
        super().__init__()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.n_max_blocks = cdiv(max_seq_len, column_block)
        # 希望最内层是seqlen
        # 希望KCache的数据存储模式是bsz, n_local_kv_heads, max_seq_len // column_block, head_dim, column_block
        if register_buffer:
            self.register_buffer(f"k_cache_first_8_layer_{layer_idx}", torch.zeros(
                (bsz, n_local_kv_heads, cdiv(max_seq_len, column_block), head_dim, column_block), dtype=torch.uint8,
                device=device))
            self.register_buffer(f"k_cache_mid_4_layer_{layer_idx}", torch.zeros(
                (bsz, n_local_kv_heads, cdiv(max_seq_len, column_block), head_dim, column_block // 2), dtype=torch.uint8,
                device=device))
            self.register_buffer(f"k_cache_last_4_layer_{layer_idx}", torch.zeros(
                (bsz, n_local_kv_heads, cdiv(max_seq_len, column_block), head_dim, column_block // 2), dtype=torch.uint8,
                device=device))
            self.register_buffer(f"k_cache_exp_column_max_layer_{layer_idx}", torch.zeros(
                (bsz, 1, n_local_kv_heads, head_dim), dtype=torch.uint8, device=device))
            self.k_cache_first_8 = getattr(self, f"k_cache_first_8_layer_{layer_idx}")
            self.k_cache_mid_4 = getattr(self, f"k_cache_mid_4_layer_{layer_idx}")
            self.k_cache_last_4 = getattr(self, f"k_cache_last_4_layer_{layer_idx}")
            self.k_cache_exp_column_max = getattr(self, f"k_cache_exp_column_max_layer_{layer_idx}")
            torch._dynamo.mark_static_address(self.k_cache_first_8)
            torch._dynamo.mark_static_address(self.k_cache_mid_4)
            torch._dynamo.mark_static_address(self.k_cache_last_4)
            torch._dynamo.mark_static_address(self.k_cache_exp_column_max)
        else:
            self.k_cache_first_8 = torch.zeros(
                (bsz, n_local_kv_heads, cdiv(max_seq_len, column_block), head_dim, column_block), dtype=torch.uint8,
                device=device)
            self.k_cache_mid_4 = torch.zeros(
                (bsz, n_local_kv_heads, cdiv(max_seq_len, column_block), head_dim, column_block // 2), dtype=torch.uint8,
                device=device)
            self.k_cache_last_4 = torch.zeros(
                (bsz, n_local_kv_heads, cdiv(max_seq_len, column_block), head_dim, column_block // 2), dtype=torch.uint8,
                device=device)
            self.k_cache_exp_column_max = torch.zeros(
                (bsz, 1, n_local_kv_heads, head_dim), dtype=torch.uint8, device=device)
        self.device = device
        self.restk = None
        self.restklen = 0
        self.reststartpos = 0
        self.current_cache_len = 0

    def zerolize(self):
        self.k_cache_first_8.zero_()
        self.k_cache_mid_4.zero_()
        self.k_cache_last_4.zero_()
        self.k_cache_exp_column_max.zero_()
        self.current_cache_len = 0

    def save(self, new_k: torch.Tensor, start_pos: int, seqlen: int):
        assert new_k.shape == (self.bsz, seqlen, self.n_local_kv_heads,
                               self.head_dim), f"new_k.shape: {new_k.shape} != {(self.bsz, seqlen, self.n_local_kv_heads, self.head_dim)}"
        assert start_pos + seqlen <= self.max_seq_len, f"start_pos + seqlen: {start_pos + seqlen} > {self.max_seq_len}"
        assert start_pos == 0 or seqlen == 1, f"start_pos: {start_pos}, seqlen: {seqlen}"
        assert start_pos == self.current_cache_len, "start pos != current KV-Cache len"
        self.current_cache_len += seqlen
        if start_pos == 0:
            self.k_cache_exp_column_max = (torch.max(new_k, dim=1, keepdim=True).values.view(torch.int16) >> 10).to(
                torch.uint8) & 0x1F
            cuda_module.k_cache_save(self.k_cache_first_8,
                                     self.k_cache_mid_4,
                                     self.k_cache_last_4,
                                     new_k.view(torch.uint8),
                                     self.bsz,
                                     self.n_local_kv_heads,
                                     seqlen // column_block,
                                     self.n_max_blocks,
                                     self.head_dim,
                                     column_block,
                                     start_pos,
                                     seqlen)
            print(f"k_cache_save: {self.k_cache_first_8.shape}, {self.k_cache_mid_4.shape}, {self.k_cache_last_4.shape}, {new_k.view(torch.uint8).shape}, {self.bsz}, {self.n_local_kv_heads}, {seqlen // column_block}, {self.n_max_blocks}, {self.head_dim}, {column_block}, {start_pos}, {seqlen}")
            self.restk = torch.zeros((self.bsz, column_block, self.n_local_kv_heads, self.head_dim), dtype=torch.half,
                                     device=self.device)
            self.restk[:, :seqlen % column_block, :, :] = new_k[:, seqlen - (seqlen % column_block):, :]
            self.restklen = seqlen % column_block
            self.reststartpos = seqlen - self.restklen
        else:
            self.k_cache_exp_column_max = torch.maximum(self.k_cache_exp_column_max,
                                                        (new_k.view(torch.int16) >> 10).to(torch.uint8) & 0x1F)
            self.restk[:, self.restklen:self.restklen + seqlen, :, :] = new_k
            self.restklen += seqlen
            if self.restklen == column_block:
                cuda_module.k_cache_save(self.k_cache_first_8,
                                         self.k_cache_mid_4,
                                         self.k_cache_last_4,
                                         self.restk.view(torch.uint8),
                                         self.bsz,
                                         self.n_local_kv_heads,
                                         1,
                                         self.n_max_blocks,
                                         self.head_dim,
                                         column_block,
                                         self.reststartpos,
                                         column_block)
                self.restklen = 0
                self.reststartpos += column_block
                self.restk = torch.zeros((self.bsz, column_block, self.n_local_kv_heads, self.head_dim), dtype=torch.half,
                                            device=self.device)
        """void k_cache_save(torch::Tensor &k_cache_first_8,
                  torch::Tensor &k_cache_mid_4,
                  torch::Tensor &k_cache_last_4,
                  torch::Tensor &k_new,
                  const int bsz,
                  const int n_kv_heads,
                  const int n_blocks,
                  const int n_max_blocks,
                  const int d_head,
                  const int d_block,
                  const int start_pos,
                  const int seqlen);"""

    def decoding_compute(self, q: torch.Tensor, start_pos: int, seqlen: int, reference: bool = False):
        assert q.shape[2] == self.n_local_kv_heads, "we currently only support MHA, don't support GQA"
        assert q.shape == (self.bsz, seqlen, self.n_local_kv_heads,
                           self.head_dim), f"q.shape: {q.shape} != {(self.bsz, seqlen, self.n_local_kv_heads, self.head_dim)}"
        q_exp = (q.view(torch.int16) >> 10).to(torch.uint8) & 0x1F
        k_column_max_multiply_q = q_exp + self.k_cache_exp_column_max
        k_column_max_multiply_q_max = torch.max(k_column_max_multiply_q, dim=3, keepdim=True).values
        s = cuda_module.k_cache_compute(self.k_cache_first_8,
                                              self.k_cache_mid_4,
                                              self.k_cache_last_4,
                                              k_column_max_multiply_q,
                                              k_column_max_multiply_q_max,
                                              q.view(torch.uint8),
                                              self.bsz,
                                              self.n_local_kv_heads,
                                              self.reststartpos // column_block,
                                              self.n_max_blocks,
                                              self.head_dim,
                                              column_block,
                                              self.reststartpos,
                                              1.0 / math.sqrt(self.head_dim),
                                              reference)
        s = s.view(torch.half)
        assert self.restklen + self.reststartpos == start_pos + seqlen
        s_rest = torch.matmul(q.transpose(1, 2), self.restk.transpose(1, 2).transpose(2, 3)[:,:,:, :self.restklen]) / math.sqrt(self.head_dim)
        s[:, :, :, self.reststartpos:self.reststartpos + self.restklen] = s_rest
        s[:, :, :, start_pos + seqlen:] = float("-inf")
        return s
