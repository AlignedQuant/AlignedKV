#include <torch/extension.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

/*
 * 本文件为V_cache的相关函数
 * 包括存储函数
 * 直接计算q->s函数
 * */

//self.v_cache_first_8   (bsz, n_local_kv_heads, max_seq_len, head_dim)
//self.v_cache_mid_4     (bsz, n_local_kv_heads, max_seq_len, head_dim // 2)
//self.v_cache_last_4    (bsz, n_local_kv_heads, max_seq_len, head_dim // 2)
//xv                     (bsz, seqlen, n_local_kv_heads, head_dim)
// grid  (bsz, n_local_kv_heads, 1)
// block (head_dim // 2, 1, 1)
__global__ void v_cache_save_kernel(unsigned char* v_cache_first_8_data,
                                    unsigned char* v_cache_mid_4_data,
                                    unsigned char* v_cache_last_4_data,
                                    unsigned char* v_new_data,
                                    const int stride_batch_v_cache,
                                    const int stride_head_v_cache,
                                    const int stride_seq_v_cache,
                                    const int stride_batch_v_new,
                                    const int stride_seq_v_new,
                                    const int stride_head_v_new,
                                    const int start_pos,
                                    const int seqlen){
    for(int pos = start_pos; pos < start_pos + seqlen; ++pos){
        v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + pos * stride_seq_v_cache + threadIdx.x) * 2]
                = v_new_data[(blockIdx.x * stride_batch_v_new + blockIdx.z * stride_seq_v_new + blockIdx.y * stride_head_v_new + threadIdx.x * 2) * 2 + 1];
        v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + pos * stride_seq_v_cache + threadIdx.x) * 2 + 1]
                = v_new_data[(blockIdx.x * stride_batch_v_new + blockIdx.z * stride_seq_v_new + blockIdx.y * stride_head_v_new + threadIdx.x * 2 + 1) * 2 + 1];
        v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + pos * stride_seq_v_cache + threadIdx.x)]
                = (v_new_data[(blockIdx.x * stride_batch_v_new + blockIdx.z * stride_seq_v_new + blockIdx.y * stride_head_v_new + threadIdx.x * 2) * 2] & 0xf0)
                  + ((v_new_data[(blockIdx.x * stride_batch_v_new + blockIdx.z * stride_seq_v_new + blockIdx.y * stride_head_v_new + threadIdx.x * 2 + 1) * 2] & 0xf0) >> 4);
        v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + pos * stride_seq_v_cache + threadIdx.x)]
                = ((v_new_data[(blockIdx.x * stride_batch_v_new + blockIdx.z * stride_seq_v_new + blockIdx.y * stride_head_v_new + threadIdx.x * 2) * 2] & 0x0f) << 4)
                  + (v_new_data[(blockIdx.x * stride_batch_v_new + blockIdx.z * stride_seq_v_new + blockIdx.y * stride_head_v_new + threadIdx.x * 2 + 1) * 2] & 0x0f);
    }
}

// 存储函数
void v_cache_save(torch::Tensor &v_cache_first_8,
                  torch::Tensor &v_cache_mid_4,
                  torch::Tensor &v_cache_last_4,
                  torch::Tensor &v_new,
                  const int bsz,
                  const int n_local_kv_heads,
                  const int max_seq_len,
                  const int head_dim,
                  const int start_pos,
                  const int seq_len){
    unsigned char* v_cache_first_8_data = (unsigned char*)v_cache_first_8.data_ptr<unsigned char>();
    unsigned char* v_cache_mid_4_data = (unsigned char*)v_cache_mid_4.data_ptr<unsigned char>();
    unsigned char* v_cache_last_4_data = (unsigned char*)v_cache_last_4.data_ptr<unsigned char>();
    unsigned char* v_new_data = (unsigned char*)v_new.data_ptr<unsigned char>();
    int stride_seq_v_cache = head_dim >> 1;
    int stride_head_v_cache = stride_seq_v_cache * max_seq_len;
    int stride_batch_v_cache = stride_head_v_cache * n_local_kv_heads;
    int stride_head_v_new = head_dim;
    int stride_seq_v_new = stride_head_v_new * n_local_kv_heads;
    int stride_batch_v_new = stride_seq_v_new * v_new.size(1);
    dim3 grid(bsz, n_local_kv_heads, 1);
    dim3 block(head_dim >> 1, 1, 1);
    v_cache_save_kernel<<<grid, block>>>(v_cache_first_8_data,
                                         v_cache_mid_4_data,
                                         v_cache_last_4_data,
                                         v_new_data,
                                         stride_batch_v_cache,
                                         stride_head_v_cache,
                                         stride_seq_v_cache,
                                         stride_batch_v_new,
                                         stride_seq_v_new,
                                         stride_head_v_new,
                                         start_pos,
                                         seq_len);
}

// 预读取，测试数量级
//self.v_cache_first_8   (bsz, n_local_kv_heads, max_seq_len, head_dim)
//self.v_cache_mid_4     (bsz, n_local_kv_heads, max_seq_len, head_dim // 2)
//self.v_cache_last_4    (bsz, n_local_kv_heads, max_seq_len, head_dim // 2)
// test: (self.bsz, self.n_local_kv_heads, 1, self.top_max_k)
// test_o: (self.bsz, self.n_local_kv_heads, 1, self.head_dim)
// grid  (bsz, n_local_kv_heads, 1)
// block (head_dim // 2, 1, 1)
__global__ void v_cache_test_compute_kernel(unsigned char* v_cache_first_8_data,
                                            unsigned char* v_cache_mid_4_data,
                                            unsigned char* v_cache_last_4_data,
                                            half* test_s_value,
                                            int* test_s_index,
                                            half* test_o,
                                            const int stride_batch_v_cache,
                                            const int stride_head_v_cache,
                                            const int stride_seq_v_cache,
                                            const int stride_batch_test_s,
                                            const int stride_head_test_s,
                                            const int stride_batch_test_o,
                                            const int stride_head_test_o,
                                            const int top_max_k){
    half2 result = __half2half2(__short_as_half(0x0000));
    for(int i = 0; i < top_max_k; ++i){
        unsigned short v_data_1 = 0x0080;
        unsigned short v_data_2 = 0x0080;
        // test exp only first 8
        int index = test_s_index[(blockIdx.x * stride_batch_test_s + blockIdx.y * stride_head_test_s + i)] * top_max_k + i;
//        int index = __ldg(test_s_index + (blockIdx.x * stride_batch_test_s + blockIdx.y * stride_head_test_s + i)) * top_max_k + i;
        v_data_1 += ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + index * stride_seq_v_cache + threadIdx.x) * 2]) << 8;
        v_data_2 += ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + index * stride_seq_v_cache + threadIdx.x) * 2 + 1]) << 8;
//        v_data_1 += ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + index * stride_seq_v_cache + threadIdx.x) * 2)) << 8;
//        v_data_2 += ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + index * stride_seq_v_cache + threadIdx.x) * 2 + 1)) << 8;
        v_data_1 &= 0x7FFF;
        v_data_2 &= 0x7FFF;
        half2 temp_v = __halves2half2(__ushort_as_half(v_data_1), __ushort_as_half(v_data_2));
        half2 temp_s = __half2half2(test_s_value[(blockIdx.x * stride_batch_test_s + blockIdx.y * stride_head_test_s + i)]);
        result = __hfma2(temp_v, temp_s, result);
    }
    test_o[(blockIdx.x * stride_batch_test_o + blockIdx.y * stride_head_test_o + threadIdx.x * 2)] = __low2half(result);
    test_o[(blockIdx.x * stride_batch_test_o + blockIdx.y * stride_head_test_o + threadIdx.x * 2 + 1)] = __high2half(result);
}

torch::Tensor v_cache_test_compute(torch::Tensor &v_cache_first_8,
                          torch::Tensor &v_cache_mid_4,
                          torch::Tensor &v_cache_last_4,
                          torch::Tensor &test_s_value,
                          torch::Tensor &test_s_index,
                          const int bsz,
                          const int n_local_kv_heads,
                          const int max_seq_len,
                          const int head_dim,
                          const int top_max_k){
    auto options = torch::TensorOptions().dtype(torch::kByte).device(test_s_value.device());
    torch::Tensor test_o = torch::empty({bsz, n_local_kv_heads, 1, head_dim * 2}, options);
    unsigned char* v_cache_first_8_data = (unsigned char*)v_cache_first_8.data_ptr<unsigned char>();
    unsigned char* v_cache_mid_4_data = (unsigned char*)v_cache_mid_4.data_ptr<unsigned char>();
    unsigned char* v_cache_last_4_data = (unsigned char*)v_cache_last_4.data_ptr<unsigned char>();
    half* test_s_value_data = (half*)test_s_value.data_ptr<unsigned char>();
    int* test_s_index_data = (int*)test_s_index.data_ptr<unsigned char>();
    half* test_o_data = (half*)test_o.data_ptr<unsigned char>();
    int stride_seq_v_cache = head_dim >> 1;
    int stride_head_v_cache = stride_seq_v_cache * max_seq_len;
    int stride_batch_v_cache = stride_head_v_cache * n_local_kv_heads;
    int stride_head_test_s = top_max_k;
    int stride_batch_test_s = stride_head_test_s * n_local_kv_heads;
    int stride_head_test_o = head_dim;
    int stride_batch_test_o = stride_head_test_o * n_local_kv_heads;
    dim3 grid(bsz, n_local_kv_heads, 1);
    dim3 block(head_dim >> 1, 1, 1);
    v_cache_test_compute_kernel<<<grid, block>>>(v_cache_first_8_data,
                                                 v_cache_mid_4_data,
                                                 v_cache_last_4_data,
                                                 test_s_value_data,
                                                 test_s_index_data,
                                                 test_o_data,
                                                 stride_batch_v_cache,
                                                 stride_head_v_cache,
                                                 stride_seq_v_cache,
                                                 stride_batch_test_s,
                                                 stride_head_test_s,
                                                 stride_batch_test_o,
                                                 stride_head_test_o,
                                                 top_max_k);
    // cudaDeviceSynchronize();
    return test_o;
}

//self.v_cache_first_8   (bsz, n_local_kv_heads, max_seq_len, head_dim)
//self.v_cache_mid_4     (bsz, n_local_kv_heads, max_seq_len, head_dim // 2)
//self.v_cache_last_4    (bsz, n_local_kv_heads, max_seq_len, head_dim // 2)
// s: (self.bsz, self.n_local_kv_heads, 1, seqlen or blocked(seqlen))
// o: (self.bsz, self.n_local_kv_heads, 1, self.head_dim)
// s_exp_expect_alignment_min: (self.bsz, self.n_local_kv_heads, 1, 1)
// grid  (bsz, n_local_kv_heads, 1)
// block (head_dim // 2, 1, 1)
template <bool reference>
__global__ void v_cache_compute_kernel(unsigned char* v_cache_first_8_data,
                                        unsigned char* v_cache_mid_4_data,
                                        unsigned char* v_cache_last_4_data,
                                        half* s,
                                        half* o,
                                        unsigned char* s_exp_expect_alignment_min,
                                        const int stride_batch_v_cache,
                                        const int stride_head_v_cache,
                                        const int stride_seq_v_cache,
                                        const int stride_batch_s,
                                        const int stride_head_s,
                                        const int stride_batch_o,
                                        const int stride_head_o,
                                        const int stride_batch_s_exp_expect_alignment_min,
                                        const int start_pos_add_seqlen){
    half2 result = __half2half2(__short_as_half(0x0000));
    short s_exp_expect_alignment_min_data = s_exp_expect_alignment_min[(blockIdx.x * stride_batch_s_exp_expect_alignment_min + blockIdx.y)];
//    short s_exp_expect_alignment_min_data = __ldg(s_exp_expect_alignment_min + (blockIdx.x * stride_batch_s_exp_expect_alignment_min + blockIdx.y));
    s_exp_expect_alignment_min_data <<= 10;
    __shared__ unsigned char trick[2];
    if (threadIdx.x == 0){
        trick[0] = 0x00;
        trick[1] = 0x88;
    }
    for(int i = 0; i < start_pos_add_seqlen; ++i){
        unsigned short v_data_1;
        unsigned short v_data_2;
        half s_data = s[(blockIdx.x * stride_batch_s + blockIdx.y * stride_head_s + i)];
//        half s_data = __ldg(s + (blockIdx.x * stride_batch_s + blockIdx.y * stride_head_s + i));

        if(not reference) {
            // 优化版本
            short diff = s_exp_expect_alignment_min_data - __half_as_short(s_data);
//            diff = (diff < 0) ? 0 : diff;
//            diff >>= 12;
//            switch (diff) {
//                case 0:
//                    // 全部
//                    v_data_1 -= 0x0008;
//                    v_data_2 -= 0x0008;
//                    v_data_1 += ((unsigned short)v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0) >> 4;
//                    v_data_2 += (unsigned short)v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f;
//                case 1:
//                    // 截后四位
//                    v_data_1 += (0x0008 - 0x0080);
//                    v_data_2 += (0x0008 - 0x0080);
//                    v_data_1 += (unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0;
//                    v_data_2 += ((unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f) << 4;
//                default:
//                    // 截后八位
//                    v_data_1 += ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2]) << 8;
//                    v_data_2 += ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1]) << 8;
//            }
//            if (diff >= 0x0800) {
//                v_data_1 = ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2]) << 8 + 0x0080;
//                v_data_2 = ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1]) << 8 + 0x0080;
//            } else if (diff >= 0x0400) {
//                v_data_1 = ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2]) << 8
//                            + ((unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0) + 0x0008;
//                v_data_2 = ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1]) << 8
//                            + ((unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f) << 4 + 0x0008;
//            } else {
//                v_data_1 = ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2]) << 8
//                            + ((unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0)
//                            + ((unsigned short)v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0) >> 4;
//                v_data_2 = ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1]) << 8
//                            + ((unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f) << 4
//                            + (unsigned short)v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f;
//            }
            unsigned char* v_data_1_first_ptr = &v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2];
            unsigned char* v_data_2_first_ptr = &v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1];
            unsigned char* v_data_mid_ptr = diff >= 0x0800 ? &trick[1] : &v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)];
            unsigned char* v_data_last_ptr = diff >= 0x0400 ? ((diff >= 0x0800 ? &trick[0] : &trick[1])) : &v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)];
            v_data_1 = (((unsigned short)*v_data_1_first_ptr) << 8) + (((unsigned short)*v_data_mid_ptr) & 0xf0) + ((((unsigned short)*v_data_last_ptr) & 0xf0) >> 4);
            v_data_2 = (((unsigned short)*v_data_2_first_ptr) << 8) + ((((unsigned short)*v_data_mid_ptr) & 0x0f) << 4) + (((unsigned short)*v_data_last_ptr) & 0x0f);
//            if (diff >= 0x0800) {
//                v_data_1 = ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2)) << 8 + 0x0080;
//                v_data_2 = ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1)) << 8 + 0x0080;
//            } else if (diff >= 0x0400) {
//                v_data_1 = ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2)) << 8
//                            + ((unsigned short)__ldg(v_cache_mid_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0xf0) + 0x0008;
//                v_data_2 = ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1)) << 8
//                            + ((unsigned short)__ldg(v_cache_mid_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0x0f) << 4 + 0x0008;
//            } else {
//                v_data_1 = ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2)) << 8
//                            + ((unsigned short)__ldg(v_cache_mid_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0xf0)
//                            + ((unsigned short)__ldg(v_cache_last_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0xf0) >> 4;
//                v_data_2 = ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1)) << 8
//                            + ((unsigned short)__ldg(v_cache_mid_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0x0f) << 4
//                            + (unsigned short)__ldg(v_cache_last_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0x0f;
//            }
        } else {
            // 参考版本
            unsigned char* v_data_1_first_ptr = &v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2];
            unsigned char* v_data_2_first_ptr = &v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1];
            unsigned char* v_data_mid_ptr = &v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)];
            unsigned char* v_data_last_ptr = &v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)];
//            v_data_1 += ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2]) << 8;
//            v_data_2 += ((unsigned short)v_cache_first_8_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1]) << 8;
//            v_data_1 += (unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0;
//            v_data_2 += ((unsigned short)v_cache_mid_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f) << 4;
//            v_data_1 += ((unsigned short)v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0xf0) >> 4;
//            v_data_2 += (unsigned short)v_cache_last_4_data[(blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)] & 0x0f;

//            v_data_1 += ((unsigned short)*v_data_1_first_ptr) << 8;
//            v_data_2 += ((unsigned short)*v_data_2_first_ptr) << 8;
//            v_data_1 += (unsigned short)*v_data_mid_ptr & 0xf0;
//            v_data_2 += ((unsigned short)*v_data_mid_ptr & 0x0f) << 4;
//            v_data_1 += ((unsigned short)*v_data_last_ptr & 0xf0) >> 4;
//            v_data_2 += (unsigned short)*v_data_last_ptr & 0x0f;

            v_data_1 = (((unsigned short)*v_data_1_first_ptr) << 8) + (((unsigned short)*v_data_mid_ptr) & 0xf0) + ((((unsigned short)*v_data_last_ptr) & 0xf0) >> 4);
            v_data_2 = (((unsigned short)*v_data_2_first_ptr) << 8) + ((((unsigned short)*v_data_mid_ptr) & 0x0f) << 4) + (((unsigned short)*v_data_last_ptr) & 0x0f);

//            v_data_1 += ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2)) << 8;
//            v_data_2 += ((unsigned short)__ldg(v_cache_first_8_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x) * 2 + 1)) << 8;
//            v_data_1 += (unsigned short)__ldg(v_cache_mid_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0xf0;
//            v_data_2 += ((unsigned short)__ldg(v_cache_mid_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0x0f) << 4;
//            v_data_1 += ((unsigned short)__ldg(v_cache_last_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0xf0) >> 4;
//            v_data_2 += (unsigned short)__ldg(v_cache_last_4_data + (blockIdx.x * stride_batch_v_cache + blockIdx.y * stride_head_v_cache + i * stride_seq_v_cache + threadIdx.x)) & 0x0f;
        }

        half2 temp_v = __halves2half2(__ushort_as_half(v_data_1), __ushort_as_half(v_data_2));
//        half2 temp_v = __halves2half2(__ushort_as_half(0x3c00), __ushort_as_half(0x3c00));
        half2 temp_s = __halves2half2(s_data, s_data);
        result = __hfma2(temp_v, temp_s, result);
    }
    o[(blockIdx.x * stride_batch_o + blockIdx.y * stride_head_o + threadIdx.x * 2)] = __low2half(result);
    o[(blockIdx.x * stride_batch_o + blockIdx.y * stride_head_o + threadIdx.x * 2 + 1)] = __high2half(result);
}

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
                     bool reference){
    auto options = torch::TensorOptions().dtype(torch::kByte).device(s.device());
    torch::Tensor o = torch::empty({bsz, n_local_kv_heads, 1, head_dim * 2}, options);
    unsigned char* v_cache_first_8_data = (unsigned char*)v_cache_first_8.data_ptr<unsigned char>();
    unsigned char* v_cache_mid_4_data = (unsigned char*)v_cache_mid_4.data_ptr<unsigned char>();
    unsigned char* v_cache_last_4_data = (unsigned char*)v_cache_last_4.data_ptr<unsigned char>();
    half* s_data = (half*)s.data_ptr<unsigned char>();
    half* o_data = (half*)o.data_ptr<unsigned char>();
    unsigned char* s_exp_expect_alignment_min_data = (unsigned char*)s_exp_expect_alignment_min.data_ptr<unsigned char>();
    int stride_seq_v_cache = head_dim >> 1;
    int stride_head_v_cache = stride_seq_v_cache * max_seq_len;
    int stride_batch_v_cache = stride_head_v_cache * n_local_kv_heads;
    int stride_head_s = s_last_dim;
    int stride_batch_s = stride_head_s * n_local_kv_heads;
    int stride_head_o = head_dim;
    int stride_batch_o = stride_head_o * n_local_kv_heads;
    int stride_batch_s_exp_expect_alignment_min = n_local_kv_heads;
    dim3 grid(bsz, n_local_kv_heads, 1);
    dim3 block(head_dim >> 1, 1, 1);
    if(reference)
        v_cache_compute_kernel<true><<<grid, block>>>(v_cache_first_8_data,
                                                      v_cache_mid_4_data,
                                                      v_cache_last_4_data,
                                                      s_data,
                                                      o_data,
                                                      s_exp_expect_alignment_min_data,
                                                      stride_batch_v_cache,
                                                      stride_head_v_cache,
                                                      stride_seq_v_cache,
                                                      stride_batch_s,
                                                      stride_head_s,
                                                      stride_batch_o,
                                                      stride_head_o,
                                                      stride_batch_s_exp_expect_alignment_min,
                                                      start_pos_add_seqlen);
    else
        v_cache_compute_kernel<false><<<grid, block>>>(v_cache_first_8_data,
                                                       v_cache_mid_4_data,
                                                       v_cache_last_4_data,
                                                       s_data,
                                                       o_data,
                                                       s_exp_expect_alignment_min_data,
                                                       stride_batch_v_cache,
                                                       stride_head_v_cache,
                                                       stride_seq_v_cache,
                                                       stride_batch_s,
                                                       stride_head_s,
                                                       stride_batch_o,
                                                       stride_head_o,
                                                       stride_batch_s_exp_expect_alignment_min,
                                                       start_pos_add_seqlen);
    // cudaDeviceSynchronize();
    return o;
}