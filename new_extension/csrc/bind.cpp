#include <torch/extension.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

void k_cache_save(
    torch::Tensor& k_new,                  // (bsz, n_local_kv_heads, d_head, seqlen)
    torch::Tensor& k_cache_first_8,       // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 4)
    torch::Tensor& k_cache_mid_4,         // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    torch::Tensor& k_cache_last_4,        // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    const unsigned int start_block             // 开始的block
);

torch::Tensor k_cache_compute(
    torch::Tensor& q,                  // (bsz, n_local_kv_heads, n_gqa_group, d_head)
    torch::Tensor& k_cache_first_8,    // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 4)
    torch::Tensor& k_cache_mid_4,      // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    torch::Tensor& k_cache_last_4,     // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    torch::Tensor& diff_array,         // (1, bsz, n_local_kv_heads, d_head)
    const unsigned int column_block_num, 
    bool reference,
    bool use_tensorcore
);

void v_cache_save(
    torch::Tensor v_new,                    // (bsz, n_local_kv_heads, seqlen, d_head)
    torch::Tensor v_cache_first_8_data,    // (bsz, n_local_kv_heads, seqlen, d_head // 4)
    torch::Tensor v_cache_mid_4_data,      // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    torch::Tensor v_cache_last_4_data,    // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    const unsigned int start_seq
);

torch::Tensor v_cache_compute(
    torch::Tensor& s,                  // (bsz, n_local_kv_heads, n_gqa_group, seqlen)
    torch::Tensor& v_cache_first_8,    // (bsz, n_local_kv_heads, seqlen, d_head // 4)
    torch::Tensor& v_cache_mid_4,      // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    torch::Tensor& v_cache_last_4,     // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    torch::Tensor& diff_array,         // (bsz, n_local_kv_heads, seqlen)
    const unsigned int seqlen,
    const bool reference,
    const bool use_tensorcore
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k_cache_save", &k_cache_save, "k_cache_save");
    m.def("k_cache_compute", &k_cache_compute, "k_cache_compute");
    m.def("v_cache_save", &v_cache_save, "v_cache_save");
    m.def("v_cache_compute", &v_cache_compute, "v_cache_compute");
}