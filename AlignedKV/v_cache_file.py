import torch
import math
from torch.utils.cpp_extension import load
import torch.nn.functional as F
from .KV_cache_cpp_extention import cuda_module

bits_in_a_channel = 256
column_block = bits_in_a_channel // 4


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b

class V_Cache_Class(torch.nn.Module):
    def __init__(self, bsz: int, max_seq_len: int, n_local_kv_heads: int, head_dim: int, layer_id: int,
                 device: str | torch.device = "cuda:0", register_buffer: bool = False, top_max_k: int = 32):
        super().__init__()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.n_max_blocks = cdiv(max_seq_len, column_block)
        # 希望VCache的储存模式为bsz, n_local_kv_heads, max_seq_len, head_dim
        if register_buffer:
            self.register_buffer(f"v_cache_first_8_layer_{layer_id}", torch.zeros(
                (bsz, n_local_kv_heads, max_seq_len, head_dim), dtype=torch.uint8,
                device=device))
            self.register_buffer(f"v_cache_mid_4_layer_{layer_id}", torch.zeros(
                (bsz, n_local_kv_heads, max_seq_len, head_dim // 2), dtype=torch.uint8,
                device=device))
            self.register_buffer(f"v_cache_last_4_layer_{layer_id}", torch.zeros(
                (bsz, n_local_kv_heads, max_seq_len, head_dim // 2), dtype=torch.uint8,
                device=device))
            self.register_buffer(f"v_cache_exp_column_max_layer_{layer_id}", torch.zeros(
                (bsz, 1, n_local_kv_heads, head_dim), dtype=torch.uint8,
                device=device))
            self.v_cache_first_8 = getattr(self, f"v_cache_first_8_layer_{layer_id}")
            self.v_cache_mid_4 = getattr(self, f"v_cache_mid_4_layer_{layer_id}")
            self.v_cache_last_4 = getattr(self, f"v_cache_last_4_layer_{layer_id}")
            self.v_cache_exp_column_max = getattr(self, f"v_cache_exp_column_max_layer_{layer_id}")
            torch._dynamo.mark_static_address(self.v_cache_first_8)
            torch._dynamo.mark_static_address(self.v_cache_mid_4)
            torch._dynamo.mark_static_address(self.v_cache_last_4)
            torch._dynamo.mark_static_address(self.v_cache_exp_column_max)
        else:
            self.v_cache_first_8 = torch.zeros(
                (bsz, n_local_kv_heads, max_seq_len, head_dim), dtype=torch.uint8,
                device=device)
            self.v_cache_mid_4 = torch.zeros(
                (bsz, n_local_kv_heads, max_seq_len, head_dim // 2), dtype=torch.uint8,
                device=device)
            self.v_cache_last_4 = torch.zeros(
                (bsz, n_local_kv_heads, max_seq_len, head_dim // 2), dtype=torch.uint8,
                device=device)
            self.v_cache_exp_column_max = torch.zeros(
                (bsz, 1, n_local_kv_heads, head_dim), dtype=torch.uint8,
                device=device)
        self.device = device
        self.top_max_k = top_max_k
        self.restv = None
        self.restvlen = 0
        self.reststartpos = 0
        self.current_cache_len = 0

    def zerolize(self):
        self.v_cache_first_8.zero_()
        self.v_cache_mid_4.zero_()
        self.v_cache_last_4.zero_()
        self.v_cache_exp_column_max.zero_()
        self.current_cache_len = 0

    def save(self, new_v: torch.Tensor, start_pos: int, seqlen: int):
        assert new_v.shape == (self.bsz, seqlen, self.n_local_kv_heads,
                               self.head_dim), f"new_v.shape: {new_v.shape} != {(self.bsz, seqlen, self.n_local_kv_heads, self.head_dim)}"
        assert start_pos + seqlen <= self.max_seq_len, f"start_pos + seqlen: {start_pos + seqlen} > {self.max_seq_len}"
        assert start_pos == 0 or seqlen == 1, f"start_pos: {start_pos}, seqlen: {seqlen}"
        assert start_pos == self.current_cache_len, "start pos != current KV-Cache len"
        self.current_cache_len += seqlen
        if start_pos == 0:
            self.v_cache_exp_column_max = (torch.max(new_v, dim=1, keepdim=True).values.view(torch.int16) >> 10).to(
                torch.uint8) & 0x1F
            cuda_module.v_cache_save(self.v_cache_first_8,
                                     self.v_cache_mid_4,
                                     self.v_cache_last_4,
                                     new_v.view(torch.uint8),
                                     self.bsz,
                                     self.n_local_kv_heads,
                                     self.max_seq_len,
                                     self.head_dim,
                                     start_pos,
                                     seqlen - seqlen % column_block)
            self.restv = torch.zeros((self.bsz, column_block, self.n_local_kv_heads, self.head_dim), dtype=torch.half,
                                     device=self.device)
            self.restv[:, :seqlen % column_block, :, :] = new_v[:, seqlen - (seqlen % column_block):, :]
            self.restvlen = seqlen % column_block
            self.reststartpos = seqlen - self.restvlen
        else:
            self.v_cache_exp_column_max = torch.maximum(self.v_cache_exp_column_max,
                                                        (new_v.view(torch.int16) >> 10).to(torch.uint8) & 0x1F)
            self.restv[:, self.restvlen:self.restvlen + seqlen, :, :] = new_v
            self.restvlen += seqlen
            if self.restvlen == column_block:
                cuda_module.v_cache_save(self.v_cache_first_8,
                                         self.v_cache_mid_4,
                                         self.v_cache_last_4,
                                         self.restv.view(torch.uint8),
                                         self.bsz,
                                         self.n_local_kv_heads,
                                         self.max_seq_len,
                                         self.head_dim,
                                         self.reststartpos,
                                         self.restvlen)
                self.restvlen = self.restvlen % column_block
                self.reststartpos += column_block
                self.restv = torch.zeros((self.bsz, column_block, self.n_local_kv_heads, self.head_dim), dtype=torch.half,
                                            device=self.device)


    def test_compute(self, s: torch.Tensor, start_pos: int, seqlen: int, top_max_k: int):
        # test: find top-k
        if start_pos + seqlen == s.shape[-1]:
            padding_group_size = cdiv(start_pos + seqlen, top_max_k)
            padding_to = padding_group_size * top_max_k
            padding_len = padding_to - start_pos - seqlen
            s = F.pad(s, (0, padding_len), "constant", 0)
        else:
            padding_group_size = s.shape[-1] // top_max_k
        s = s.view(self.bsz, self.n_local_kv_heads, padding_group_size, top_max_k)
        test_s_value, test_s_index = torch.max(s, dim=2, keepdim=True)
        test_s_index = test_s_index.to(torch.int32)
        test_o = cuda_module.v_cache_test_compute(self.v_cache_first_8,
                                                  self.v_cache_mid_4,
                                                  self.v_cache_last_4,
                                                  test_s_value.view(torch.uint8),
                                                  test_s_index.view(torch.uint8),
                                                  self.bsz,
                                                  self.n_local_kv_heads,
                                                  self.max_seq_len,
                                                  self.head_dim,
                                                  top_max_k)
        test_o = test_o.view(torch.half)
        test_o_exp = (test_o.view(torch.int16) >> 10).to(torch.uint8) & 0x1F - 1 # -1 is for abs compute
        return test_o_exp

    def decoding_compute(self, s: torch.Tensor, start_pos: int, seqlen: int, reference: bool = False):
        assert s.shape[1] == self.n_local_kv_heads, "we currently only support MHA, don't support GQA"
        assert s.shape == (self.bsz, self.n_local_kv_heads, 1, s.shape[-1]), f"s.shape: {s.shape} != {(self.bsz, self.n_local_kv_heads, 1, start_pos + seqlen)}"
        assert start_pos + seqlen <= self.max_seq_len, f"start_pos + seqlen: {start_pos + seqlen} > {self.max_seq_len}"
        assert start_pos == 0 or seqlen == 1, f"start_pos: {start_pos}, seqlen: {seqlen}"
        s_cut = s[:, :, :, :self.reststartpos]
        s_test = s_cut if self.reststartpos else s
        test_o_exp = self.test_compute(s_test, start_pos, seqlen, self.top_max_k)
        # print("test_o_exp", test_o_exp)

        # calculate s_exp alignment
        fp16_exp_bias = 2 ** 4 - 1
        s_exp_expect_alignment = test_o_exp + fp16_exp_bias - self.v_cache_exp_column_max
        # print("v_cache_exp_column_max", self.v_cache_exp_column_max)
        # print("s_exp_expect_alignment", s_exp_expect_alignment)
        s_exp_expect_alignment_min = torch.min(s_exp_expect_alignment, dim=-1, keepdim=True).values # min - s即为可以省略的位数
        # print("s_exp_expect_alignment_min", s_exp_expect_alignment_min)
        # o = torch.zeros((self.bsz, self.n_local_kv_heads, 1, self.head_dim), dtype=torch.half, device=self.device)
        """void v_cache_compute(torch::Tensor &v_cache_first_8,
                     torch::Tensor &v_cache_mid_4,
                     torch::Tensor &v_cache_last_4,
                     torch::Tensor &s,
                     torch::Tensor &o,
                     torch::Tensor &s_exp_expect_alignment_min,
                     const int bsz,
                     const int n_local_kv_heads,
                     const int max_seq_len,
                     const int head_dim,
                     const int s_last_dim,
                     const int start_pos_add_seqlen);"""
        o = cuda_module.v_cache_compute(self.v_cache_first_8,
                                                self.v_cache_mid_4,
                                                self.v_cache_last_4,
                                                s_cut.view(torch.uint8),
                                                s_exp_expect_alignment_min,
                                                self.bsz,
                                                self.n_local_kv_heads,
                                                self.max_seq_len,
                                                self.head_dim,
                                                s_cut.shape[-1],
                                                self.reststartpos,
                                                reference)
        o = o.view(torch.half)
        assert self.reststartpos + self.restvlen == start_pos + seqlen, f"self.reststartpos + self.restvlen: {self.reststartpos + self.restvlen} != {start_pos + seqlen}"
        o += torch.matmul(s[:, :, :, self.reststartpos:start_pos + seqlen], self.restv.transpose(1, 2)[:, :, :self.restvlen, :])
        return o






