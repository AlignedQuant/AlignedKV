import torch
import math
from .myextension import AlignedKV_extension, COLUMN_BLOCK, COLUMN_BLOCK_SELF

o_list_len = 8
fp16_exp_bias = 2 ** 4 - 1

def make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device = "cuda:0"):
    assert max_seq_len % COLUMN_BLOCK == 0, f"max_seq_len {max_seq_len} must be divisible by COLUMN_BLOCK {COLUMN_BLOCK}"
    v_cache_first_8 = torch.zeros((bsz, n_local_kv_heads, max_seq_len, head_dim // 4), dtype=torch.uint32, device=device)
    v_cache_mid_4 = torch.zeros((bsz, n_local_kv_heads, max_seq_len, head_dim // 8), dtype=torch.uint32, device=device)
    v_cache_last_4 = torch.zeros((bsz, n_local_kv_heads, max_seq_len, head_dim // 8), dtype=torch.uint32, device=device)
    return v_cache_first_8, v_cache_mid_4, v_cache_last_4

class VCache(object):
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
        self.v_cache_first_8, self.v_cache_mid_4, self.v_cache_last_4 = make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device)
        self.v_cache_exp_row_max = torch.zeros((bsz, n_local_kv_heads, max_seq_len), dtype=torch.int16, device=device)
        self.start_pos = 0

        # rest
        self.restv = None # (bsz, seqlen, n_local_kv_heads, head_dim)

        # old o
        self.o_list = torch.zeros((o_list_len, bsz, n_local_kv_heads, gqa, head_dim), dtype=torch.half, device=device)
        self.o_list_evict = 0

    def save_rest(self):
        if self.restv is None:
            return
        rest_len = self.restv.shape[1]
        my_len = self.start_pos
        assert self.restv.shape == (self.bsz, rest_len, self.n_local_kv_heads, self.head_dim), f"restv shape {self.restv.shape} != {(self.bsz, rest_len, self.n_local_kv_heads, self.head_dim)}"
        assert my_len + rest_len <= self.max_seq_len, f"my_len {my_len} + rest_len {rest_len} > max_seq_len {self.max_seq_len}"
        save_len = rest_len - (rest_len % COLUMN_BLOCK_SELF)
        if save_len > 0:
            to_save = self.restv.transpose(1, 2)[:, :, :save_len, :].contiguous() # (bsz, n_local_kv_heads, save_len, head_dim)
            self.restv = self.restv[:, save_len:, :, :] if rest_len % COLUMN_BLOCK_SELF > 0 else None
            AlignedKV_extension.v_cache_save(to_save, self.v_cache_first_8, self.v_cache_mid_4, self.v_cache_last_4, self.start_pos)

            to_save_row_max = torch.max(torch.abs(to_save), dim=-1).values.view(self.bsz, self.n_local_kv_heads, save_len)
            to_save_row_max = to_save_row_max.view(torch.int16) >> 11
            self.v_cache_exp_row_max[:, :, self.start_pos:self.start_pos+save_len] = to_save_row_max

            self.start_pos += save_len

    def add_rest(self, newv):
        newv = newv.to(self.device)
        if self.restv is None:
            self.restv = newv
        else:
            self.restv = torch.cat([self.restv, newv], dim=1)

    def prefill_o_list(self, o):
        o = o.to(self.device)
        # o(bsz, seqlen, n_heads * head_dim)
        n_head = self.gqa * self.n_local_kv_heads
        seq_len = o.shape[1]
        assert o.shape == (self.bsz, seq_len, n_head * self.head_dim), f"o shape {o.shape} != {(self.bsz, seq_len, n_head * self.head_dim)}"
        o = o.view(self.bsz, seq_len, self.n_local_kv_heads, self.gqa, self.head_dim)
        o = o.permute(1, 0, 2, 3, 4).contiguous() # (seqlen, bsz, n_local_kv_heads, gqa, head_dim)
        start_len = max(0, seq_len - o_list_len)
        for i in range(start_len, seq_len):
            self.o_list[self.o_list_evict, :, :, :, :] = o[i, :, :, :, :]
            self.o_list_evict = (self.o_list_evict + 1) % o_list_len

    def gemm_sv_main(self, s, reference = False, use_tensorcore = False):
        s = s.to(self.device)
        if self.start_pos == 0:
            return None
        n_head = s.shape[1]
        s = s[:, :, :, :self.start_pos].contiguous()
        assert s.shape == (self.bsz, n_head, 1, self.start_pos), f"s shape {s.shape} != {(self.bsz, n_head, 1, self.start_pos)}"
        assert n_head == self.gqa * self.n_local_kv_heads, f"n_head {n_head} != gqa {self.gqa} * n_local_kv_heads {self.n_local_kv_heads}"
        gqa = self.gqa
        assert n_head % self.n_local_kv_heads == 0, f"n_head {n_head} must be divisible by n_local_kv_heads {self.n_local_kv_heads}"
        s = s.view(self.bsz, self.n_local_kv_heads, gqa, self.start_pos)

        # get s exp
        s_exp = torch.abs(s).view(torch.int16) >> 11 # (bsz, n_local_kv_heads, gqa, start_pos)

        # get old o exp
        o_exp = torch.abs(self.o_list)
        o_exp = torch.max(o_exp, dim=0).values.view(self.bsz, self.n_local_kv_heads, gqa, self.head_dim)
        o_exp = torch.min(o_exp, dim=-1).values.view(self.bsz, self.n_local_kv_heads, gqa, 1)
        o_exp = o_exp.view(torch.int16) >> 11 # (bsz, n_local_kv_heads, gqa, 1)

        # v_cache_exp_row_max
        v_cache_exp_row_max = self.v_cache_exp_row_max[:, :, :self.start_pos].contiguous() # (bsz, n_local_kv_heads, start_pos)
        v_cache_exp_row_max = v_cache_exp_row_max.view(self.bsz, self.n_local_kv_heads, 1, self.start_pos)

        # diff exp
        diff_exp = o_exp + fp16_exp_bias - s_exp - v_cache_exp_row_max
        assert diff_exp.shape == (self.bsz, self.n_local_kv_heads, gqa, self.start_pos), f"diff_exp shape {diff_exp.shape} != {(self.bsz, self.n_local_kv_heads, gqa, self.start_pos)}"
        diff_exp = torch.min(diff_exp, dim=2).values.view(self.bsz, self.n_local_kv_heads, self.start_pos)

        # calculate
        result = AlignedKV_extension.v_cache_compute(s,
                                                     self.v_cache_first_8,
                                                     self.v_cache_mid_4,
                                                     self.v_cache_last_4,
                                                     diff_exp,
                                                     self.start_pos,
                                                     reference,
                                                     use_tensorcore)
        
        # sum up
        result = torch.sum(result, dim=2)

        # view
        result = result.view(self.bsz, 1, n_head, self.head_dim)
        return result
    
    def gemm_sv_rest(self, s):
        if self.restv is None:
            return None
        n_head = s.shape[1]
        s = s[:, :, :, self.start_pos:]
        rest_len = self.restv.shape[1]
        assert s.shape == (self.bsz, n_head, 1, rest_len), f"s shape {s.shape} != {(self.bsz, n_head, 1, rest_len)}"
        assert n_head == self.gqa * self.n_local_kv_heads, f"n_head {n_head} != gqa {self.gqa} * n_local_kv_heads {self.n_local_kv_heads}"
        gqa = self.gqa
        assert n_head % self.n_local_kv_heads == 0, f"n_head {n_head} must be divisible by n_local_kv_heads {self.n_local_kv_heads}"

        s = s.reshape(self.bsz, self.n_local_kv_heads, gqa, rest_len)
        rest_v = self.restv.transpose(1, 2) # (bsz, n_local_kv_heads, rest_len, head_dim)
        assert rest_v.shape == (self.bsz, self.n_local_kv_heads, rest_len, self.head_dim), f"rest_v shape {rest_v.shape} != {(self.bsz, self.n_local_kv_heads, rest_len, self.head_dim)}"
        o = torch.matmul(s, rest_v)

        # view
        o = o.view(self.bsz, 1, n_head, self.head_dim)
        return o
    
    def gemm_sv(self, s, reference = False, use_tensorcore = False):
        result_main = self.gemm_sv_main(s, reference, use_tensorcore)
        result_rest = self.gemm_sv_rest(s)
        if result_main is None:
            result = result_rest
        elif result_rest is None:
            result = result_main
        else:
            result = result_main + result_rest
        
        # update o_list
        if result is not None:
            self.o_list[self.o_list_evict, :, :, :, :] = result.view(self.bsz, self.n_local_kv_heads, self.gqa, self.head_dim)
            self.o_list_evict = (self.o_list_evict + 1) % o_list_len

        return result
    
    def decoding(self, s, v, reference = False, use_tensorcore = False):
        self.add_rest(v)
        o = self.gemm_sv(s, reference, use_tensorcore)
        self.save_rest()
        return o
    
    def prefill_save_v(self, v):
        self.add_rest(v)
        self.save_rest()

    def zerolize(self):
        self.v_cache_first_8.zero_()
        self.v_cache_mid_4.zero_()
        self.v_cache_last_4.zero_()
        self.v_cache_exp_row_max.zero_()
        self.start_pos = 0
        self.restv = None
        self.o_list.zero_()
        self.o_list_evict = 0