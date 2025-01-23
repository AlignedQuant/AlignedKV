import torch
import math
from .myextension import AlignedKV_extension, COLUMN_BLOCK, COLUMN_BLOCK_SELF

def make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device = "cuda:0"):
    assert max_seq_len % COLUMN_BLOCK == 0, f"max_seq_len {max_seq_len} must be divisible by COLUMN_BLOCK {COLUMN_BLOCK}"
    n_max_blocks = max_seq_len // COLUMN_BLOCK
    k_cache_first_8 = torch.zeros((bsz, n_local_kv_heads, n_max_blocks, head_dim, COLUMN_BLOCK // 4), dtype=torch.uint32, device=device)
    k_cache_mid_4 = torch.zeros((bsz, n_local_kv_heads, n_max_blocks, head_dim, COLUMN_BLOCK // 8), dtype=torch.uint32, device=device)
    k_cache_last_4 = torch.zeros((bsz, n_local_kv_heads, n_max_blocks, head_dim, COLUMN_BLOCK // 8), dtype=torch.uint32, device=device)
    return k_cache_first_8, k_cache_mid_4, k_cache_last_4

class KCache(object):
    def __init__(self, bsz, max_seq_len, n_local_kv_heads, head_dim, gqa, device = "cuda:0", layer_idx = 0):
        # config
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.bsz = bsz
        self.device = device
        self.gqa = gqa
        self.layer_idx = layer_idx

        # cache
        self.k_cache_first_8, self.k_cache_mid_4, self.k_cache_last_4 = make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device)
        self.k_cache_exp_column_max = torch.zeros((bsz, n_local_kv_heads, head_dim), dtype=torch.int16, device=device)
        self.start_block = 0

        # rest
        self.restk = None # (bsz, seqlen, n_local_kv_heads, head_dim)

    def save_rest(self):
        if self.restk is None:
            return
        rest_len = self.restk.shape[1]
        my_len = self.start_block * COLUMN_BLOCK
        assert self.restk.shape == (self.bsz, rest_len, self.n_local_kv_heads, self.head_dim), f"restk shape {self.restk.shape} != {(self.bsz, rest_len, self.n_local_kv_heads, self.head_dim)}"
        assert my_len + rest_len <= self.max_seq_len, f"my_len {my_len} + rest_len {rest_len} > max_seq_len {self.max_seq_len}"
        save_len = rest_len - (rest_len % COLUMN_BLOCK_SELF)
        if save_len > 0:
            to_save = self.restk.permute(0, 2, 3, 1)[:, :, :, :save_len].contiguous() # (bsz, n_local_kv_heads, head_dim, save_len)
            self.restk = self.restk[:, save_len:, :, :] if rest_len % COLUMN_BLOCK_SELF > 0 else None
            AlignedKV_extension.k_cache_save(to_save, self.k_cache_first_8, self.k_cache_mid_4, self.k_cache_last_4, self.start_block)
            self.start_block += save_len // COLUMN_BLOCK

            to_save_column_max = torch.max(torch.abs(to_save), dim=-1).values.view(self.bsz, self.n_local_kv_heads, self.head_dim)
            to_save_column_max = to_save_column_max.view(torch.int16) >> 11
            self.k_cache_exp_column_max = torch.maximum(self.k_cache_exp_column_max, to_save_column_max)

    def add_rest(self, newk):
        newk = newk.to(self.device)
        if self.restk is None:
            self.restk = newk
        else:
            self.restk = torch.cat([self.restk, newk], dim=1)

    def gemm_qkt_main(self, q, reference = False, use_tensorcore = False):
        q = q.to(self.device)
        if self.start_block == 0:
            return None
        n_head = q.shape[2]
        assert q.shape == (self.bsz, 1, n_head, self.head_dim), f"q shape {q.shape} != {(self.bsz, 1, n_head, self.head_dim)}"
        assert n_head == self.gqa * self.n_local_kv_heads, f"n_head {n_head} != gqa {self.gqa} * n_local_kv_heads {self.n_local_kv_heads}"
        gqa = self.gqa
        assert n_head % self.n_local_kv_heads == 0, f"n_head {n_head} must be divisible by n_local_kv_heads {self.n_local_kv_heads}"
        q = q.view(self.bsz, self.n_local_kv_heads, gqa, self.head_dim)

        # get q exp
        q_exp = torch.abs(q).view(torch.int16) >> 11

        # diff exp
        diff_exp = q_exp + self.k_cache_exp_column_max.view(self.bsz, self.n_local_kv_heads, 1, self.head_dim)
        max_diff_exp = torch.max(diff_exp, dim=-1).values
        diff_exp = max_diff_exp.view(self.bsz, self.n_local_kv_heads, gqa, 1) - diff_exp
        diff_exp = torch.min(diff_exp, dim=2).values.view(self.bsz, self.n_local_kv_heads, self.head_dim)

        # calculate
        result = AlignedKV_extension.k_cache_compute(q,
                                                     self.k_cache_first_8,
                                                     self.k_cache_mid_4,
                                                     self.k_cache_last_4,
                                                     diff_exp,
                                                     self.start_block,
                                                     reference,
                                                     use_tensorcore)
        result = result.view(self.bsz, n_head, 1, self.start_block * COLUMN_BLOCK)
        return result # (bsz, n_head, 1, seqlen)
    
    def gemm_qkt_rest(self, q):
        if self.restk is None:
            return None
        n_head = q.shape[2]
        assert q.shape == (self.bsz, 1, n_head, self.head_dim), f"q shape {q.shape} != {(self.bsz, 1, n_head, self.head_dim)}"
        assert n_head == self.gqa * self.n_local_kv_heads, f"n_head {n_head} != gqa {self.gqa} * n_local_kv_heads {self.n_local_kv_heads}"
        assert n_head % self.n_local_kv_heads == 0, f"n_head {n_head} must be divisible by n_local_kv_heads {self.n_local_kv_heads}"
        q = q.view(self.bsz, self.n_local_kv_heads, self.gqa, self.head_dim)
        rest_kt = self.restk.permute(0, 2, 3, 1) # (bsz, n_local_kv_heads, head_dim, seqlen)
        result = torch.matmul(q, rest_kt) # (bsz, n_local_kv_heads, gqa, seqlen)
        result = result.view(self.bsz, n_head, 1, self.restk.shape[1])
        return result # (bsz, n_head, 1, seqlen)
    
    def gemm_qkt(self, q, reference = False, use_tensorcore = False):
        result_main = self.gemm_qkt_main(q, reference, use_tensorcore)
        result_rest = self.gemm_qkt_rest(q)
        if result_main is None:
            return result_rest
        if result_rest is None:
            return result_main
        return torch.cat([result_main, result_rest], dim=-1)
    
    def decoding(self, q, k, reference = False, use_tensorcore = False):
        self.add_rest(k)
        s = self.gemm_qkt(q, reference, use_tensorcore)
        self.save_rest()

        # check have inf or nan
        if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
            print(f"s: {s}")
            raise Exception("have inf or nan in s")
        return s
    
    def prefill_save_k(self, k):
        self.add_rest(k)
        self.save_rest()

    def zeorlize(self):
        self.k_cache_first_8.zero_()
        self.k_cache_mid_4.zero_()
        self.k_cache_last_4.zero_()
        self.k_cache_exp_column_max.zero_()
        self.start_block = 0
        self.restk = None