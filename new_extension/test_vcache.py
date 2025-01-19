import torch
from torch import nn
import math
try:
    from .myextension import AlignedKV_extension, COLUMN_BLOCK
except:
    from myextension import AlignedKV_extension, COLUMN_BLOCK

def make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device = "cuda:0"):
    assert max_seq_len % COLUMN_BLOCK == 0, f"max_seq_len {max_seq_len} must be divisible by COLUMN_BLOCK {COLUMN_BLOCK}"
    v_cache_first_8 = torch.zeros((bsz, n_local_kv_heads, max_seq_len, head_dim // 4), dtype=torch.uint32, device=device)
    v_cache_mid_4 = torch.zeros((bsz, n_local_kv_heads, max_seq_len, head_dim // 8), dtype=torch.uint32, device=device)
    v_cache_last_4 = torch.zeros((bsz, n_local_kv_heads, max_seq_len, head_dim // 8), dtype=torch.uint32, device=device)
    return v_cache_first_8, v_cache_mid_4, v_cache_last_4

class VCache:
    def __init__(self, first_tensor, mid_tensor, last_tensor):
        # Ensure input tensors are of uint32 type
        assert first_tensor.dtype == torch.uint32, "first tensor must be of type torch.uint32"
        assert mid_tensor.dtype == torch.uint32, "mid tensor must be of type torch.uint32"
        assert last_tensor.dtype == torch.uint32, "last tensor must be of type torch.uint32"

        # Initialize the tensors
        self.v_cache_first_8_data = first_tensor
        self.v_cache_mid_4_data = mid_tensor
        self.v_cache_last_4_data = last_tensor

        # Initialize strides
        self.stride_bsz = last_tensor.stride(0)
        self.stride_n_local_kv_heads = last_tensor.stride(1)
        self.stride_seqlen = last_tensor.stride(2)
        self.stride_d_head = last_tensor.stride(3)

    def load_8_uint16(self, bsz, n_local_kv_heads, seqlen, d_head, diff):
        
        # Load data according to the diff value
        first_8_data_1 = self.v_cache_first_8_data[bsz, n_local_kv_heads, seqlen, d_head // 4] if diff < 11 else 0
        first_8_data_2 = self.v_cache_first_8_data[bsz, n_local_kv_heads, seqlen, d_head // 4 + 1] if diff < 11 else 0
        mid_data = self.v_cache_mid_4_data[bsz, n_local_kv_heads, seqlen, d_head // 8] if diff < 8 else 0
        last_data = self.v_cache_last_4_data[bsz, n_local_kv_heads, seqlen, d_head // 8] if diff < 8 else 0
        first_8_data_1 = first_8_data_1.item()
        first_8_data_2 = first_8_data_2.item()
        mid_data = mid_data.item()
        last_data = last_data.item()

        # Reconstruct the original uint32 values
        num = torch.zeros(4, dtype=torch.uint32)

        num[0] = ((first_8_data_1 & 0x00FF00FF) << 8) | ((mid_data & 0x000F000F) << 4) | (last_data & 0x000F000F)
        num[1] = ((first_8_data_1 & 0xFF00FF00)) | ((mid_data & 0x00F000F0)) | ((last_data & 0x00F000F0) >> 4)
        num[2] = ((first_8_data_2 & 0x00FF00FF) << 8) | ((mid_data & 0x0F000F00) >> 4) | ((last_data & 0x0F000F00) >> 8)
        num[3] = ((first_8_data_2 & 0xFF00FF00)) | ((mid_data & 0xF000F000) >> 8) | ((last_data & 0xF000F000) >> 12)

        return num
    
    def load_all(self):
        bsz, n_local_kv_heads, seqlen, d_head = self.v_cache_last_4_data.shape
        d_head *= 8
        result = torch.zeros((bsz, n_local_kv_heads, seqlen, d_head), dtype=torch.half, device=self.v_cache_last_4_data.device)
        for bsz_idx in range(bsz):
            for n_local_kv_heads_idx in range(n_local_kv_heads):
                for seqlen_idx in range(seqlen):
                    for d_head_idx in range(0, d_head, 8):
                        temp = self.load_8_uint16(bsz_idx, n_local_kv_heads_idx, seqlen_idx, d_head_idx, 0)
                        temp = temp.view(torch.half)
                        result[bsz_idx, n_local_kv_heads_idx, seqlen_idx, d_head_idx:d_head_idx + 8] = temp

        return result


def print_error_info(result, result_ref_torch, name1, name2 = "result_ref_torch"):
    print(f"============== {name1} fail ==============")
    print(f"============== {name1} ==============")
    print(result.shape)
    print(result)
    print(f"============== {name2} ==============")
    print(result_ref_torch.shape)
    print(result_ref_torch)
    print(f"============== diff between {name1} and {name2} ==============")
    print(result - result_ref_torch)
    print(f"============== relative diff between {name1} and {name2} ==============")
    print((result - result_ref_torch) / (result_ref_torch + 1e-3))


def test_vcache():
    bsz = 2
    max_seq_len = 512
    seqlen = 256
    n_local_kv_heads = 4
    head_dim = 128
    device = "cuda:1"
    n_gqa = 4
    v_cache_first_8, v_cache_mid_4, v_cache_last_4 = make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device)
    diff_tensor = torch.zeros((bsz, n_local_kv_heads, seqlen), dtype=torch.int16, device=device)
    
    # save
    new_v = torch.randn((bsz, seqlen, n_local_kv_heads, head_dim), dtype=torch.half, device=device)
    # new_v = torch.ones((bsz, seqlen, n_local_kv_heads, head_dim), dtype=torch.half, device=device)
    # new_v[0, 0, 0, :] = torch.arange(128, dtype=torch.half, device=device)
    # new_v[0, 0, 0, :] = 1
    # new_v[0, 0:32, 0, 16] = torch.arange(32, dtype=torch.half, device=device) + 1
    new_v_norm_order = new_v.permute(0, 2, 1, 3).contiguous()
    assert new_v_norm_order.shape == (bsz, n_local_kv_heads, seqlen, head_dim)
    AlignedKV_extension.v_cache_save(new_v_norm_order, v_cache_first_8, v_cache_mid_4, v_cache_last_4, 0)

    # check
    print("testing v_cache_save")
    v_cache = VCache(v_cache_first_8, v_cache_mid_4, v_cache_last_4)
    result = v_cache.load_all()[:, :, :seqlen, :]
    if torch.allclose(result, new_v_norm_order, atol=1e-3):
        print("v_cache_save passed!")
    else:
        print_error_info(result, new_v_norm_order, "v_cache_save")

    # save append
    assert seqlen * 2 <= max_seq_len, f"seqlen * 2 {seqlen * 2} must be less than max_seq_len {max_seq_len}"
    new_v_append = torch.randn((bsz, seqlen, n_local_kv_heads, head_dim), dtype=torch.half, device=device)
    print("testing v_cache_save_append")
    new_v_append_norm_order = new_v_append.permute(0, 2, 1, 3).contiguous()
    assert new_v_append_norm_order.shape == (bsz, n_local_kv_heads, seqlen, head_dim)
    AlignedKV_extension.v_cache_save(new_v_append_norm_order, v_cache_first_8, v_cache_mid_4, v_cache_last_4, seqlen)
    result = v_cache.load_all()[:, :, seqlen:seqlen * 2, :]
    if torch.allclose(result, new_v_append_norm_order, atol=1e-3):
        print("v_cache_save_append passed!")
    else:
        print_error_info(result, new_v_append_norm_order, "v_cache_save_append")

    # check calculate
    print("testing v_cache_compute reference ...")
    s_shape = (bsz, n_local_kv_heads, n_gqa, seqlen)
    s = torch.randn(s_shape, dtype=torch.half, device=device)
    # s = torch.zeros(s_shape, dtype=torch.half, device=device)
    # s[0, 0, 0, 50] = 1
    # s[0, 0, 1, 1] = 1
    # s[0, 0, 0, 1] = 2
    # s[0, 0, 0, 2] = 3
    # s[0, 0, 0, 3] = 4
    diff_array_shape = (bsz, n_local_kv_heads, seqlen)
    diff_array = torch.zeros(diff_array_shape, dtype=torch.int16, device=device)
    # torch::Tensor o = torch::ones({bsz, n_local_kv_heads, expand_seq_acc, n_gqa_group, d_head}
    result_ref_use_tensorcore = AlignedKV_extension.v_cache_compute(s, 
                                                                    v_cache_first_8, 
                                                                    v_cache_mid_4, 
                                                                    v_cache_last_4, 
                                                                    diff_array, 
                                                                    seqlen,
                                                                    True,
                                                                    True)
    result_ref_no_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                     v_cache_first_8,
                                                                     v_cache_mid_4,
                                                                     v_cache_last_4,
                                                                     diff_array,
                                                                     seqlen,
                                                                     True,
                                                                     False)
    result_ref_torch = torch.matmul(s, new_v_norm_order)
    result_ref_use_tensorcore_sum = torch.sum(result_ref_use_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    result_ref_no_tensorcore_sum = torch.sum(result_ref_no_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    if torch.allclose(result_ref_use_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-2):
        print("v_cache_compute with tensorcore passed!")
    else:
        print_error_info(result_ref_use_tensorcore_sum, result_ref_torch, "v_cache_compute with tensorcore")
    if torch.allclose(result_ref_no_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-2):
        print("v_cache_compute with cudacore passed!")
    else:
        print_error_info(result_ref_no_tensorcore_sum, result_ref_torch, "v_cache_compute with cudacore")

    # check diff
    print("testing v_cache_compute mymethod ...")
    # diff = 0
    diff_array = torch.zeros(diff_array_shape, dtype=torch.int16, device=device)
    result_use_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                v_cache_first_8,
                                                                v_cache_mid_4,
                                                                v_cache_last_4,
                                                                diff_array,
                                                                seqlen,
                                                                False,
                                                                True)
    result_no_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                v_cache_first_8,
                                                                v_cache_mid_4,
                                                                v_cache_last_4,
                                                                diff_array,
                                                                seqlen,
                                                                False,
                                                                False)
    result_use_tensorcore_sum = torch.sum(result_use_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    result_no_tensorcore_sum = torch.sum(result_no_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    if torch.allclose(result_use_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("v_cache_compute diff = 0 with tensorcore passed!")
    else:
        print_error_info(result_use_tensorcore_sum, result_ref_torch, "v_cache_compute with tensorcore diff = 0")
    if torch.allclose(result_no_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("v_cache_compute diff = 0 with cudacore passed!")
    else:
        print_error_info(result_no_tensorcore_sum, result_ref_torch, "v_cache_compute with cudacore diff = 0")
    # diff = 4
    diff_array = torch.ones(diff_array_shape, dtype=torch.int16, device=device) * 4
    result_use_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                v_cache_first_8,
                                                                v_cache_mid_4,
                                                                v_cache_last_4,
                                                                diff_array,
                                                                seqlen,
                                                                False,
                                                                True)
    result_no_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                v_cache_first_8,
                                                                v_cache_mid_4,
                                                                v_cache_last_4,
                                                                diff_array,
                                                                seqlen,
                                                                False,
                                                                False)
    result_use_tensorcore_sum = torch.sum(result_use_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    result_no_tensorcore_sum = torch.sum(result_no_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    if torch.allclose(result_use_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("v_cache_compute with tensorcore passed!")
    else:
        print_error_info(result_use_tensorcore_sum, result_ref_torch, "v_cache_compute with tensorcore")
    if torch.allclose(result_no_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("v_cache_compute with cudacore passed!")
    else:
        print_error_info(result_no_tensorcore_sum, result_ref_torch, "v_cache_compute with cudacore")
    
    # random mask test
    print("testing v_cache_compute random mask ...")
    diff_array = torch.randint(0, 11, diff_array_shape, dtype=torch.int16, device=device)
    # diff_array = torch.zeros(diff_array_shape, dtype=torch.int16, device=device)
    # diff_array[0, 0, 50] = 4
    # diff_array[0, 0, 1] = 8
    new_v_norm_order_copy = new_v_norm_order.clone()
    new_v_norm_order_copy = new_v_norm_order_copy.view(torch.int16)
    new_v_norm_order_copy[diff_array >= 4] &= 0xFFF0
    new_v_norm_order_copy[diff_array >= 4] |= 0x0008
    new_v_norm_order_copy[diff_array >= 8] &= 0xFF00
    new_v_norm_order_copy[diff_array >= 8] |= 0x0088
    new_v_norm_order_copy[diff_array >= 11] = 0
    new_v_norm_order_copy = new_v_norm_order_copy.view(torch.half)
    result_ref_torch = torch.matmul(s, new_v_norm_order_copy)
    result_use_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                v_cache_first_8,
                                                                v_cache_mid_4,
                                                                v_cache_last_4,
                                                                diff_array,
                                                                seqlen,
                                                                False,
                                                                True)
    result_no_tensorcore = AlignedKV_extension.v_cache_compute(s,
                                                                v_cache_first_8,
                                                                v_cache_mid_4,
                                                                v_cache_last_4,
                                                                diff_array,
                                                                seqlen,
                                                                False,
                                                                False)
    result_use_tensorcore_sum = torch.sum(result_use_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    result_no_tensorcore_sum = torch.sum(result_no_tensorcore, dim=2).view(bsz, n_local_kv_heads, n_gqa, head_dim)
    if torch.allclose(result_use_tensorcore_sum, result_ref_torch, atol=1e-2, rtol=5e-2):
        print("v_cache_compute with tensorcore random mask passed!")
    else:
        print_error_info(result_use_tensorcore_sum, result_ref_torch, "v_cache_compute with tensorcore random mask")
    if torch.allclose(result_no_tensorcore_sum, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("v_cache_compute with cudacore random mask passed!")
    else:
        print_error_info(result_no_tensorcore_sum, result_ref_torch, "v_cache_compute with cudacore random mask")




if __name__ == "__main__":
    test_vcache()