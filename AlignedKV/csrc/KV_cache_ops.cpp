#include <torch/extension.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

void k_cache_save(torch::Tensor &k_cache_first_8,
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
                  const int seqlen);

torch::Tensor k_cache_compute(torch::Tensor &k_cache_first_8,
                              torch::Tensor &k_cache_mid_4,
                              torch::Tensor &k_cache_last_4,
                              torch::Tensor &k_column_max_multiply_q,
                              torch::Tensor &k_column_max_multiply_q_max,
                              torch::Tensor &q,
                              const int bsz,
                              const int n_kv_heads,
                              const int n_blocks,
                              const int n_max_blocks,
                              const int d_head,
                              const int d_block,
                              const int seqlen,
                              const float sqrt_dim,
                              const bool reference);

void v_cache_save(torch::Tensor &v_cache_first_8,
                  torch::Tensor &v_cache_mid_4,
                  torch::Tensor &v_cache_last_4,
                  torch::Tensor &v_new,
                  const int bsz,
                  const int n_local_kv_heads,
                  const int max_seq_len,
                  const int head_dim,
                  const int start_pos,
                  const int seq_len);

torch::Tensor v_cache_test_compute(torch::Tensor &v_cache_first_8,
                                   torch::Tensor &v_cache_mid_4,
                                   torch::Tensor &v_cache_last_4,
                                   torch::Tensor &test_s_value,
                                   torch::Tensor &test_s_index,
                                   const int bsz,
                                   const int n_local_kv_heads,
                                   const int max_seq_len,
                                   const int head_dim,
                                   const int top_max_k);

torch::Tensor v_cache_compute(torch::Tensor &v_cache_first_8,
                              torch::Tensor &v_cache_mid_4,
                              torch::Tensor &v_cache_last_4,
                              torch::Tensor &s,
                              torch::Tensor &s_exp_expect_alignment_min,
                              const int bsz,
                              const int n_local_kv_heads,
                              const int max_seq_len,
                              const int head_dim,
                              const int s_last_dim,
                              const int start_pos_add_seqlen,
                              bool reference);

torch::Tensor gemm_cuda(torch::Tensor A,
                        torch::Tensor B,
                        const int stride_batch_A,
                        const int stride_n_heads_A,
                        const int stride_l_A,
                        const int stride_m_A,
                        const int stride_batch_B,
                        const int stride_n_heads_B,
                        const int stride_m_B,
                        const int stride_n_B,
                        const int l,
                        const int m,
                        const int n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k_cache_save", &k_cache_save, "save new key to K_cache");
    m.def("k_cache_compute", &k_cache_compute, "load K_cache and calculate attention score");
    m.def("v_cache_save", &v_cache_save, "save new value to V_cache");
    m.def("v_cache_test_compute", &v_cache_test_compute, "load V_cache and calculate attention score for test");
    m.def("v_cache_compute", &v_cache_compute, "load V_cache and calculate attention score");
    m.def("gemm_cuda", &gemm_cuda, "gemm_cuda");
}