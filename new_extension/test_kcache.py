import torch
from torch import nn
import math
try:
    from .myextension import AlignedKV_extension, COLUMN_BLOCK
except:
    from myextension import AlignedKV_extension, COLUMN_BLOCK

def make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device = "cuda:0"):
    assert max_seq_len % COLUMN_BLOCK == 0, f"max_seq_len {max_seq_len} must be divisible by COLUMN_BLOCK {COLUMN_BLOCK}"
    n_max_blocks = max_seq_len // COLUMN_BLOCK
    k_cache_first_8 = torch.zeros((bsz, n_local_kv_heads, n_max_blocks, head_dim, COLUMN_BLOCK // 4), dtype=torch.uint32, device=device)
    k_cache_mid_4 = torch.zeros((bsz, n_local_kv_heads, n_max_blocks, head_dim, COLUMN_BLOCK // 8), dtype=torch.uint32, device=device)
    k_cache_last_4 = torch.zeros((bsz, n_local_kv_heads, n_max_blocks, head_dim, COLUMN_BLOCK // 8), dtype=torch.uint32, device=device)
    return k_cache_first_8, k_cache_mid_4, k_cache_last_4

class KCache:
    def __init__(self, first_tensor, mid_tensor, last_tensor):
        # Ensure input tensors are of uint32 type
        assert first_tensor.dtype == torch.uint32, "first tensor must be of type torch.uint32"
        assert mid_tensor.dtype == torch.uint32, "mid tensor must be of type torch.uint32"
        assert last_tensor.dtype == torch.uint32, "last tensor must be of type torch.uint32"

        # Initialize the tensors
        self.k_cache_first_8_data = first_tensor
        self.k_cache_mid_4_data = mid_tensor
        self.k_cache_last_4_data = last_tensor

        # Initialize strides
        self.stride_bsz = last_tensor.stride(0)
        self.stride_n_local_kv_heads = last_tensor.stride(1)
        self.stride_seq_per_columnblock = last_tensor.stride(2)
        self.stride_d_head = last_tensor.stride(3)

    def load_8_uint16(self, bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock, diff):

        # Load data according to the diff value
        first_8_data_1 = self.k_cache_first_8_data[bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock // 4] if diff < 11 else 0
        first_8_data_2 = self.k_cache_first_8_data[bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock // 4 + 1] if diff < 11 else 0
        mid_data = self.k_cache_mid_4_data[bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock // 8] if diff < 8 else 0
        last_data = self.k_cache_last_4_data[bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock // 8] if diff < 8 else 0
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
        bsz, n_local_kv_heads, blocknums, d_head, _ = self.k_cache_last_4_data.shape
        result = torch.zeros((bsz, n_local_kv_heads, d_head, blocknums * COLUMN_BLOCK), dtype=torch.half, device=self.k_cache_last_4_data.device)
        for bsz_idx in range(bsz):
            for n_local_kv_heads_idx in range(n_local_kv_heads):
                for blocknums_idx in range(blocknums):
                    for d_head_idx in range(d_head):
                        for seq_rest_columnblock in range(0, COLUMN_BLOCK, 8):
                            temp = self.load_8_uint16(bsz_idx, n_local_kv_heads_idx, blocknums_idx, d_head_idx, seq_rest_columnblock, 0)
                            temp = temp.view(torch.half)
                            result[bsz_idx, n_local_kv_heads_idx, d_head_idx, blocknums_idx * COLUMN_BLOCK + seq_rest_columnblock:blocknums_idx * COLUMN_BLOCK + seq_rest_columnblock + 8] = temp
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


def test_kcache():
    bsz = 1
    max_seq_len = 128
    n_local_kv_heads = 1
    head_dim = 128
    device = "cuda:1"
    start_block = 0
    seqlen = 128
    gqa = 1
    k_cache_first_8, k_cache_mid_4, k_cache_last_4 = make_empty_cache(bsz, max_seq_len, n_local_kv_heads, head_dim, device)
    diff_tensor = torch.zeros((bsz, n_local_kv_heads, head_dim), dtype=torch.int16, device=device)
    
    # save
    new_k = torch.randn((bsz, seqlen, n_local_kv_heads, head_dim), dtype=torch.half, device=device)
    # new_k = torch.zeros((bsz, seqlen, n_local_kv_heads, head_dim), dtype=torch.half, device=device)
    new_k_seqlen_last = new_k.permute(0, 2, 3, 1).contiguous()
    # new_k_seqlen_last = (-torch.zeros_like(new_k_seqlen_last)) + torch.eye(seqlen, dtype=torch.half, device=device)
    assert new_k_seqlen_last.shape == (bsz, n_local_kv_heads, head_dim, seqlen)
    AlignedKV_extension.k_cache_save(new_k_seqlen_last, k_cache_first_8, k_cache_mid_4, k_cache_last_4, start_block)

    # check
    print("testing k_cache_save ...")
    k_cache = KCache(k_cache_first_8, k_cache_mid_4, k_cache_last_4)
    result = k_cache.load_all()
    if torch.allclose(result[:,:,:,:seqlen], new_k_seqlen_last):
        print("k_cache_save passed!")
    else:
        print(result.shape)
        print(result)
        print(new_k_seqlen_last.shape)
        print(new_k_seqlen_last)
        print(result - new_k_seqlen_last)
        # print(k_cache_first_8.shape)
        # print(k_cache_first_8)
        # [bsz, n_local_kv_heads, head_dim, seqlen] # 后两个维度搞反了

    # check calculate
    print("testing k_cache_compute reference ...")
    q_shape = (bsz, n_local_kv_heads, gqa, head_dim)
    q = torch.randn(q_shape, dtype=torch.half, device=device)
    # q = torch.ones(q_shape, dtype=torch.half, device=device)
    diff_array_shape = (bsz, n_local_kv_heads, head_dim)
    diff_array = torch.zeros(diff_array_shape, dtype=torch.int16, device=device)
    result_ref_use_tensorcore = AlignedKV_extension.k_cache_compute(q, 
                                                                    k_cache_first_8, 
                                                                    k_cache_mid_4, 
                                                                    k_cache_last_4, 
                                                                    diff_array, 
                                                                    seqlen // COLUMN_BLOCK,
                                                                    True,
                                                                    True)
    result_ref_not_use_tensorcore = AlignedKV_extension.k_cache_compute(q, 
                                                                       k_cache_first_8, 
                                                                       k_cache_mid_4, 
                                                                       k_cache_last_4, 
                                                                       diff_array, 
                                                                       seqlen // COLUMN_BLOCK,
                                                                       True,
                                                                          False)
    # (bsz, n_local_kv_heads, head_dim, seqlen) -> (bsz, n_local_kv_heads, head_dim, seqlen)
    result_ref_torch = torch.matmul(q, new_k_seqlen_last)
    if torch.allclose(result_ref_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-2):
        print("k_cache_compute reference use tensorcore passed!")
    else:
        print_error_info(result_ref_use_tensorcore, result_ref_torch, "result_ref_use_tensorcore")
    if torch.allclose(result_ref_not_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-2):
        print("k_cache_compute reference use cudacore passed!")
    else:
        print(result_ref_not_use_tensorcore[0, 0, 0, :])
        print_error_info(result_ref_not_use_tensorcore, result_ref_torch, "result_ref_not_use_tensorcore")

    # check diff
    print("testing k_cache_compute mymethod ...")
    # diff = 0
    diff_array = torch.zeros(diff_array_shape, dtype=torch.int16, device=device)
    result_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                k_cache_first_8,
                                                                k_cache_mid_4,
                                                                k_cache_last_4,
                                                                diff_array,
                                                                seqlen // COLUMN_BLOCK,
                                                                False,
                                                                True)
    result_not_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                   k_cache_first_8,
                                                                   k_cache_mid_4,
                                                                   k_cache_last_4,
                                                                   diff_array,
                                                                   seqlen // COLUMN_BLOCK,
                                                                   False,
                                                                   False)
    if torch.allclose(result_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-2):
        print("k_cache_compute mymethod use tensorcore passed!(for diff = 0)")
    else:
        print_error_info(result_use_tensorcore, result_ref_torch, "result_use_tensorcore_diff_0")

    if torch.allclose(result_not_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-2):
        print("k_cache_compute mymethod use cudacore passed!(for diff = 0)")
    else:
        print_error_info(result_not_use_tensorcore, result_ref_torch, "result_not_use_tensorcore_diff_0")

    # diff = 4
    diff_array = torch.ones(diff_array_shape, dtype=torch.int16, device=device) * 4
    result_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                k_cache_first_8,
                                                                k_cache_mid_4,
                                                                k_cache_last_4,
                                                                diff_array,
                                                                seqlen // COLUMN_BLOCK,
                                                                False,
                                                                True)
    result_not_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                   k_cache_first_8,
                                                                   k_cache_mid_4,
                                                                   k_cache_last_4,
                                                                   diff_array,
                                                                   seqlen // COLUMN_BLOCK,
                                                                   False,
                                                                   False)
    if torch.allclose(result_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("k_cache_compute mymethod use tensorcore passed!(for diff = 4)")
    else:
        print_error_info(result_use_tensorcore, result_ref_torch, "result_use_tensorcore_diff_4")
    if torch.allclose(result_not_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-1):
        print("k_cache_compute mymethod use cudacore passed!(for diff = 4)")
    else:
        print_error_info(result_not_use_tensorcore, result_ref_torch, "result_not_use_tensorcore_diff_4")

    # diff = 8
    diff_array = torch.ones(diff_array_shape, dtype=torch.int16, device=device) * 8
    result_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                k_cache_first_8,
                                                                k_cache_mid_4,
                                                                k_cache_last_4,
                                                                diff_array,
                                                                seqlen // COLUMN_BLOCK,
                                                                False,
                                                                True)
    result_not_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                   k_cache_first_8,
                                                                   k_cache_mid_4,
                                                                   k_cache_last_4,
                                                                   diff_array,
                                                                   seqlen // COLUMN_BLOCK,
                                                                   False,
                                                                   False)
    if (torch.sum(torch.isclose(result_use_tensorcore, result_ref_torch, atol=1, rtol=2e-1)) > 0.8 * result_use_tensorcore.numel() and
        not torch.allclose(result_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-2)):
        print("k_cache_compute mymethod use tensorcore passed!(for diff = 8)")
    else:
        print_error_info(result_use_tensorcore, result_ref_torch, "result_use_tensorcore_diff_8")
    if (torch.sum(torch.isclose(result_not_use_tensorcore, result_ref_torch, atol=1, rtol=2e-1)) > 0.8 * result_not_use_tensorcore.numel() and
        not torch.allclose(result_not_use_tensorcore, result_ref_torch, atol=1e-1, rtol=1e-2)):
        print("k_cache_compute mymethod use cudacore passed!(for diff = 8)")
    else:
        print_error_info(result_not_use_tensorcore, result_ref_torch, "result_not_use_tensorcore_diff_8")

    # random mask test
    print("testing k_cache_compute random mask ...")
    diff_array = torch.randint(0, 11, diff_array_shape, dtype=torch.int16, device=device)
    # diff_array = torch.zeros(diff_array_shape, dtype=torch.int16, device=device)
    # diff_array[0, 0, 0] = 4
    # diff_array[0, 0, 1] = 8
    # (bsz, n_local_kv_heads, head_dim, seqlen) -> (bsz, n_local_kv_heads, head_dim, seqlen)
    new_k_seqlen_last_copy = new_k_seqlen_last.clone()
    new_k_seqlen_last_copy = new_k_seqlen_last_copy.view(torch.int16)
    new_k_seqlen_last_copy[diff_array >= 4] &= 0xFFF0
    new_k_seqlen_last_copy[diff_array >= 4] |= 0x0008
    new_k_seqlen_last_copy[diff_array >= 8] &= 0xFF00
    new_k_seqlen_last_copy[diff_array >= 8] |= 0x0088
    new_k_seqlen_last_copy[diff_array >= 11] = 0
    new_k_seqlen_last_copy = new_k_seqlen_last_copy.view(torch.half)
    result_ref_torch_mask = torch.matmul(q, new_k_seqlen_last_copy)
    result_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                k_cache_first_8,
                                                                k_cache_mid_4,
                                                                k_cache_last_4,
                                                                diff_array,
                                                                seqlen // COLUMN_BLOCK,
                                                                False,
                                                                True)
    result_not_use_tensorcore = AlignedKV_extension.k_cache_compute(q,
                                                                   k_cache_first_8,
                                                                   k_cache_mid_4,
                                                                   k_cache_last_4,
                                                                   diff_array,
                                                                   seqlen // COLUMN_BLOCK,
                                                                   False,
                                                                   False)
    if torch.allclose(result_use_tensorcore, result_ref_torch_mask, atol=1e-3, rtol=1e-3):
        print("k_cache_compute random mask use tensorcore passed!")
    else:
        print("diff_array shape", diff_array.shape)
        print("diff_array", diff_array)
        print_error_info(result_use_tensorcore, result_ref_torch_mask, "result_use_tensorcore_random_mask")
    if torch.allclose(result_not_use_tensorcore, result_ref_torch_mask, atol=1e-2, rtol=5e-2):
        print("k_cache_compute random mask use cudacore passed!")
    else:
        print("diff_array shape", diff_array.shape)
        print("diff_array", diff_array)
        print_error_info(result_not_use_tensorcore, result_ref_torch_mask, "result_not_use_tensorcore_random_mask")

if __name__ == "__main__":
    test_kcache()