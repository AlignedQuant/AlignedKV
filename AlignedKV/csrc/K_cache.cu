#include <torch/extension.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

/*
 * 本文件为K_cache的相关函数
 * 包括存储函数
 * 直接计算q->s函数
 * */

// self.k_cache_first_8 = torch.zeros((bsz, n_local_kv_heads, tl.cdiv(max_seq_len, column_block), head_dim, column_block), dtype=torch.uint8)
// self.k_cache_mid_4 = torch.zeros((bsz, n_local_kv_heads, tl.cdiv(max_seq_len, column_block), head_dim, column_block // 2), dtype=torch.uint8)
// self.k_cache_last_4 = torch.zeros((bsz, n_local_kv_heads, tl.cdiv(max_seq_len, column_block), head_dim, column_block // 2), dtype=torch.uint8)
// xk,"bsz, seqlen, n_local_kv_heads, head_dim",

// 存储函数 prefill kernel
// grid (bsz, n_kv_heads, n_block)
// block (d_head, 1, 1)
__global__ void k_cache_save_prefill_kernel(unsigned char* k_cache_first_8_data,
                                            unsigned char* k_cache_mid_4_data,
                                            unsigned char* k_cache_last_4_data,
                                            unsigned char* k_new_data,
                                            // const int bsz,
                                            // const int n_kv_heads,
                                            // const int n_blocks,
                                            // const int d_head,
                                            const int d_block,
                                            const int start_pos,
                                            const int seqlen,
                                            const int stride_batch_k_cache,
                                            const int stride_kvhead_k_cache,
                                            const int stride_block_k_cache,
                                            const int stride_dim_k_cache,
                                            const int stride_batch_k_new,
                                            const int stride_seq_k_new,
                                            const int stride_kvhead_k_new){
    int block_start_seq = blockIdx.z * d_block + start_pos;
    int start_pos_block = start_pos / d_block;
    int block_end_seq = block_start_seq + d_block < seqlen ? block_start_seq + d_block : seqlen;
    for(int seq = block_start_seq; seq < block_end_seq; ++seq){
        // k_new_data[b][seq][kvhead][dim]
        // k_cache_first_8_data[b][kvhead][block][dim][seq_in_block]
        // k_cache_mid_4_data[b][kvhead][block][dim][seq_in_block]
        // k_cache_last_4_data[b][kvhead][block][dim][seq_in_block]
        int first_8_index = seq - block_start_seq;
        int mid_4_index = first_8_index >> 1;
        int move = (first_8_index & 1) << 2;
        // k_cache_first_8_data[b][kvhead][block][dim][seq_in_block] = k_new_data[b][seq][kvhead][dim]
        k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + blockIdx.y * stride_kvhead_k_cache + (blockIdx.z + start_pos_block) * stride_block_k_cache + threadIdx.x * stride_dim_k_cache) * 2 + first_8_index]
            = k_new_data[(blockIdx.x * stride_batch_k_new + seq * stride_seq_k_new + blockIdx.y * stride_kvhead_k_new + threadIdx.x) * 2 + 1];
        // k_cache_mid_4_data[b][kvhead][block][dim][seq_in_block] = k_new_data[b][seq][kvhead][dim]
        k_cache_mid_4_data[blockIdx.x * stride_batch_k_cache + blockIdx.y * stride_kvhead_k_cache + (blockIdx.z + start_pos_block) * stride_block_k_cache + threadIdx.x * stride_dim_k_cache + mid_4_index]
            += (k_new_data[(blockIdx.x * stride_batch_k_new + seq * stride_seq_k_new + blockIdx.y * stride_kvhead_k_new + threadIdx.x) * 2] & 0xf0) >> move;
        // k_cache_last_4_data[b][kvhead][block][dim][seq_in_block] = k_new_data[b][seq][kvhead][dim]
        k_cache_last_4_data[blockIdx.x * stride_batch_k_cache + blockIdx.y * stride_kvhead_k_cache + (blockIdx.z + start_pos_block) * stride_block_k_cache + threadIdx.x * stride_dim_k_cache + mid_4_index]
            += (k_new_data[(blockIdx.x * stride_batch_k_new + seq * stride_seq_k_new + blockIdx.y * stride_kvhead_k_new + threadIdx.x) * 2] & 0x0f) << (4 - move);
    }
}

// 存储函数 prefill kernel
// grid (bsz, n_kv_heads, 1)
// block (d_head, 1, 1)
__global__ void k_cache_save_decoding_kernel(unsigned char* k_cache_first_8_data,
                                             unsigned char* k_cache_mid_4_data,
                                             unsigned char* k_cache_last_4_data,
                                             unsigned char* k_new_data,
                                             // const int bsz,
                                             // const int n_kv_heads,
                                             // const int n_blocks,
                                             // const int d_head,
                                             // const int d_block,
                                             const int start_pos,
                                             // const int seqlen,
                                             const int stride_batch_k_cache,
                                             const int stride_kvhead_k_cache,
                                             const int stride_block_k_cache,
                                             const int stride_dim_k_cache,
                                             const int stride_batch_k_new,
                                             const int stride_seq_k_new,
                                             const int stride_kvhead_k_new,
                                             const int block_pos,
                                             const int first_8_index){
    int mid_4_index = first_8_index >> 1;
    int move = (first_8_index & 1) << 2;
    // k_cache_first_8_data[b][kvhead][block][dim][seq_in_block] = k_new_data[b][seq][kvhead][dim]
    k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + blockIdx.y * stride_kvhead_k_cache + block_pos * stride_block_k_cache + threadIdx.x * stride_dim_k_cache) * 2 + first_8_index]
            = k_new_data[(blockIdx.x * stride_batch_k_new + blockIdx.y * stride_kvhead_k_new + threadIdx.x) * 2 + 1];
    // k_cache_mid_4_data[b][kvhead][block][dim][seq_in_block] = k_new_data[b][seq][kvhead][dim]
    k_cache_mid_4_data[blockIdx.x * stride_batch_k_cache + blockIdx.y * stride_kvhead_k_cache + block_pos * stride_block_k_cache + threadIdx.x * stride_dim_k_cache + mid_4_index]
            += (k_new_data[(blockIdx.x * stride_batch_k_new + blockIdx.y * stride_kvhead_k_new + threadIdx.x) * 2] & 0xf0) >> move;
    // k_cache_last_4_data[b][kvhead][block][dim][seq_in_block] = k_new_data[b][seq][kvhead][dim]
    k_cache_last_4_data[blockIdx.x * stride_batch_k_cache + blockIdx.y * stride_kvhead_k_cache + block_pos * stride_block_k_cache + threadIdx.x * stride_dim_k_cache + mid_4_index]
            += (k_new_data[(blockIdx.x * stride_batch_k_new + blockIdx.y * stride_kvhead_k_new + threadIdx.x) * 2] & 0x0f) << (4 - move);
}

// 存储函数
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
                          const int seqlen){
    unsigned char* k_cache_first_8_data = k_cache_first_8.data_ptr<unsigned char>();
    unsigned char* k_cache_mid_4_data = k_cache_mid_4.data_ptr<unsigned char>();
    unsigned char* k_cache_last_4_data = k_cache_last_4.data_ptr<unsigned char>();
    unsigned char* k_new_data = (unsigned char*)k_new.data_ptr<unsigned char>();
    int stride_dim_k_cache = d_block >> 1;
    int stride_block_k_cache = stride_dim_k_cache * d_head;
    int stride_kvhead_k_cache = stride_block_k_cache * n_max_blocks;
    int stride_batch_k_cache = stride_kvhead_k_cache * n_kv_heads;
    int stride_kvhead_k_new = d_head;
    int stride_seq_k_new = stride_kvhead_k_new * n_kv_heads;
    int stride_batch_k_new = stride_seq_k_new * seqlen;
    if (start_pos == 0 || seqlen % d_block == 0){
        // prefill
        int use_block = n_blocks;
        dim3 grid(bsz, n_kv_heads, use_block);
        dim3 block(d_head, 1, 1);
        k_cache_save_prefill_kernel<<<grid, block>>>(k_cache_first_8_data,
                                                     k_cache_mid_4_data,
                                                     k_cache_last_4_data,
                                                     k_new_data,
                                                     d_block,
                                                     start_pos,
                                                     seqlen,
                                                     stride_batch_k_cache,
                                                     stride_kvhead_k_cache,
                                                     stride_block_k_cache,
                                                     stride_dim_k_cache,
                                                     stride_batch_k_new,
                                                     stride_seq_k_new,
                                                     stride_kvhead_k_new);
    } else if (seqlen == 1) {
        // decoding
        int use_block = start_pos / d_block;
        int first_8_index = start_pos - use_block * d_block;
        dim3 grid(bsz, n_kv_heads, 1);
        dim3 block(d_head, 1, 1);
        k_cache_save_decoding_kernel<<<grid, block>>>(k_cache_first_8_data,
                                                      k_cache_mid_4_data,
                                                      k_cache_last_4_data,
                                                      k_new_data,
                                                      start_pos,
                                                      stride_batch_k_cache,
                                                      stride_kvhead_k_cache,
                                                      stride_block_k_cache,
                                                      stride_dim_k_cache,
                                                      stride_batch_k_new,
                                                      stride_seq_k_new,
                                                      stride_kvhead_k_new,
                                                      use_block,
                                                      first_8_index);
    }
//    cudaDeviceSynchronize();
}

// 计算qk^T=s,仅限于decoding,如果prefill直接用矩阵计算
// 可以使用分支，保证一个block在一个分值内即可
// (batch, 1, n_local_kv_heads, head_dim) * torch.zeros((bsz, n_local_kv_heads, tl.cdiv(max_seq_len, column_block), head_dim, column_block)
// = (batch, n_local_kv_heads, 1, cdiv * column_block)
// k_column_max = (batch, 1, n_kv_heads, dim)
// k_column_max_multiply_q_max = (batch, 1, n_kv_heads, 1)
// grid (bsz, n_kv_heads // 4, block_num)
// block (column block // 2, n_kv_heads(4), 1)
template <bool reference>
__global__ void k_cache_compute_kernel(unsigned char* k_cache_first_8_data,
                                       unsigned char* k_cache_mid_4_data,
                                       unsigned char* k_cache_last_4_data,
                                       unsigned char* k_column_max_multiply_q,
                                       unsigned char* k_column_max_multiply_q_max,
                                       half* q,
                                       half* s,
                                       const int stride_batch_k_cache,
                                       const int stride_kvhead_k_cache,
                                       const int stride_block_k_cache,
                                       const int stride_dim_k_cache,
                                       const int stride_batch_q,
                                       const int stride_kvhead_q,
                                       const int stride_batch_k_column_max_multiply_q_max,
                                       const int stride_batch_s,
                                       const int stride_kvhead_s,
                                       const float sqrt_dim,
                                       const int d_head,
                                       const int d_block){
    const int kv_head_index = blockIdx.y * 4 + threadIdx.y;
    unsigned char k_column_max_multiply_q_max_data = k_column_max_multiply_q_max[(blockIdx.x * stride_batch_k_column_max_multiply_q_max + kv_head_index)];
//    unsigned char k_column_max_multiply_q_max_data = __ldg(k_column_max_multiply_q_max + (blockIdx.x * stride_batch_k_column_max_multiply_q_max + kv_head_index));
    int first_8_index_1 = threadIdx.x << 1;
    int first_8_index_2 = first_8_index_1 + 1;
    int mid_4_index = threadIdx.x;
    half2 result = __half2half2(__short_as_half(0x0000));
    __shared__ unsigned char trick[2];
    if (threadIdx.x == 0){
        trick[0] = 0x00;
        trick[1] = 0x88;
    }
    for(int i = 0; i < d_head; ++i){
        unsigned short k_data_1;
        unsigned short k_data_2;

        if(not reference){
            unsigned char diff = k_column_max_multiply_q_max_data - k_column_max_multiply_q[(blockIdx.x * stride_batch_q + kv_head_index * stride_kvhead_q + i)];
//            unsigned char diff = k_column_max_multiply_q_max_data - __ldg(k_column_max_multiply_q + (blockIdx.x * stride_batch_q + kv_head_index * stride_kvhead_q + i));
//            switch(diff >> 2){
//                case 0:
//                    // 全部
//                    k_data_1 -= 0x0008;
//                    k_data_2 -= 0x0008;
//                    k_data_1 += ((unsigned short)k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0) >> 4;
//                    k_data_2 += (unsigned short)k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f;
//                case 1:
//                    // 截后四位
//                    k_data_1 += (0x0008 - 0x0080);
//                    k_data_2 += (0x0008 - 0x0080);
//                    k_data_1 += (unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0;
//                    k_data_2 += ((unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f) << 4;
//                default:
//                    // 截后八位
//                    k_data_1 += ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1]) << 8;
//                    k_data_2 += ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2]) << 8;
//            }
//            if (diff >= 8) {
//                k_data_1 = ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1]) << 8 + 0x0080;
//                k_data_2 = ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2]) << 8 + 0x0080;
//            } else if (diff >= 4) {
//                k_data_1 = ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1]) << 8
//                            + ((unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0) + 0x0008;
//                k_data_2 = ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2]) << 8
//                            + ((unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f) << 4 + 0x0008;
//            } else {
//                k_data_1 = ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1]) << 8
//                            + ((unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0)
//                            + ((unsigned short)k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0) >> 4;
//                k_data_2 = ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2]) << 8
//                            + ((unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f) << 4
//                            + ((unsigned short)k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f);
//            }
            unsigned char* k_data_1_first_ptr = &k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1];
            unsigned char* k_data_2_first_ptr = &k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2];
            unsigned char* k_data_mid_ptr = diff >= 8 ? &trick[1] : &k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index];
            unsigned char* k_data_last_ptr = diff >= 4 ? (diff >= 8 ? &trick[0] : &trick[1]) : &k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index];
            k_data_1 = (((unsigned short)*k_data_1_first_ptr) << 8) + (((unsigned short)*k_data_mid_ptr) & 0xf0) + ((((unsigned short)*k_data_last_ptr) & 0xf0) >> 4);
            k_data_2 = (((unsigned short)*k_data_2_first_ptr) << 8) + ((((unsigned short)*k_data_mid_ptr) & 0x0f) << 4) + (((unsigned short)*k_data_last_ptr) & 0x0f);
//            if (diff >= 8){
//                k_data_1 = ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1)) << 8 + 0x0080;
//                k_data_2 = ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2)) << 8 + 0x0080;
//            } else if (diff >= 4){
//                k_data_1 = ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1)) << 8
//                            + ((unsigned short)__ldg(k_cache_mid_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0xf0) + 0x0008;
//                k_data_2 = ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2)) << 8
//                            + ((unsigned short)__ldg(k_cache_mid_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0x0f) << 4 + 0x0008;
//            } else {
//                k_data_1 = ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1)) << 8
//                            + ((unsigned short)__ldg(k_cache_mid_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0xf0)
//                            + ((unsigned short)__ldg(k_cache_last_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0xf0) >> 4;
//                k_data_2 = ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2)) << 8
//                            + ((unsigned short)__ldg(k_cache_mid_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0x0f) << 4
//                            + ((unsigned short)__ldg(k_cache_last_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0x0f);
//            }
        } else {
            k_data_1 = 0x0000;
            k_data_2 = 0x0000;
            unsigned char* k_data_1_first_ptr = &k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1];
            unsigned char* k_data_2_first_ptr = &k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2];
            unsigned char* k_data_mid_ptr = &k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index];
            unsigned char* k_data_last_ptr = &k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index];
//            k_data_1 += ((unsigned short)k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0) >> 4;
//            k_data_2 += (unsigned short)k_cache_last_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f;
//            k_data_1 += (unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0xf0;
//            k_data_2 += ((unsigned short)k_cache_mid_4_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index] & 0x0f) << 4;
//            k_data_1 += ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1]) << 8;
//            k_data_2 += ((unsigned short)k_cache_first_8_data[(blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2]) << 8;

//            k_data_1 += ((unsigned short)*k_data_1_first_ptr) << 8;
//            k_data_2 += ((unsigned short)*k_data_2_first_ptr) << 8;
//            k_data_1 += ((unsigned short)*k_data_mid_ptr) & 0xf0;
//            k_data_2 += ((unsigned short)*k_data_mid_ptr) & 0x0f << 4;
//            k_data_1 += ((unsigned short)*k_data_last_ptr) & 0xf0;
//            k_data_2 += ((unsigned short)*k_data_last_ptr) & 0x0f;

            k_data_1 = (((unsigned short)*k_data_1_first_ptr) << 8) + (((unsigned short)*k_data_mid_ptr) & 0xf0) + ((((unsigned short)*k_data_last_ptr) & 0xf0) >> 4);
            k_data_2 = (((unsigned short)*k_data_2_first_ptr) << 8) + ((((unsigned short)*k_data_mid_ptr) & 0x0f) << 4) + (((unsigned short)*k_data_last_ptr) & 0x0f);

//            k_data_1 += ((unsigned short)__ldg(k_cache_last_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0xf0) >> 4;
//            k_data_2 += (unsigned short)__ldg(k_cache_last_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0x0f;
//            k_data_1 += (unsigned short)__ldg(k_cache_mid_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0xf0;
//            k_data_2 += ((unsigned short)__ldg(k_cache_mid_4_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) + mid_4_index) & 0x0f) << 4;
//            k_data_1 += ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_1)) << 8;
//            k_data_2 += ((unsigned short)__ldg(k_cache_first_8_data + (blockIdx.x * stride_batch_k_cache + kv_head_index * stride_kvhead_k_cache + blockIdx.z * stride_block_k_cache + i * stride_dim_k_cache) * 2 + first_8_index_2)) << 8;
        }

        half2 temp_k = __halves2half2(__ushort_as_half(k_data_1), __ushort_as_half(k_data_2));
//        half2 temp_k = __halves2half2(__ushort_as_half(0x3c00), __ushort_as_half(0x3c00));
        half2 temp_q = __half2half2(q[(blockIdx.x * stride_batch_q + kv_head_index * stride_kvhead_q + i)]);
        result = __hfma2(temp_k, temp_q, result);
    }
    result = __hmul2(result, __float2half2_rn(sqrt_dim));
    int seq = blockIdx.z * d_block + threadIdx.x * 2;
    s[(blockIdx.x * stride_batch_s + kv_head_index * stride_kvhead_s + seq)] = __low2half(result); // k_data_1
    s[(blockIdx.x * stride_batch_s + kv_head_index * stride_kvhead_s + seq + 1)] = __high2half(result); // k_data_2
}

// (batch, 1, n_local_kv_heads, head_dim) * torch.zeros((bsz, n_local_kv_heads, tl.cdiv(max_seq_len, column_block), head_dim, column_block)
// = (batch, n_local_kv_heads, 1, seqlen)
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
                             const bool reference){
    auto options = torch::TensorOptions().dtype(torch::kByte).device(q.device());
    torch::Tensor s = torch::empty({bsz, n_kv_heads, 1, (n_blocks + 1) * d_block * 2}, options);
    unsigned char* k_cache_first_8_data = k_cache_first_8.data_ptr<unsigned char>();
    unsigned char* k_cache_mid_4_data = k_cache_mid_4.data_ptr<unsigned char>();
    unsigned char* k_cache_last_4_data = k_cache_last_4.data_ptr<unsigned char>();
    unsigned char* k_column_max_multiply_q_data = k_column_max_multiply_q.data_ptr<unsigned char>();
    unsigned char* k_column_max_multiply_q_max_data = k_column_max_multiply_q_max.data_ptr<unsigned char>();
    half* q_data = (half*)q.data_ptr<unsigned char>();
    half* s_data = (half*)s.data_ptr<unsigned char>();
    int stride_dim_k_cache = d_block >> 1;
    int stride_block_k_cache = stride_dim_k_cache * d_head;
    int stride_kvhead_k_cache = stride_block_k_cache * n_max_blocks;
    int stride_batch_k_cache = stride_kvhead_k_cache * n_kv_heads;
    int stride_kvhead_q = d_head;
    int stride_batch_q = stride_kvhead_q * n_kv_heads;
    int stride_batch_k_column_max_multiply_q_max = n_kv_heads;
    int stride_kvhead_s = (n_blocks + 1) * d_block;
    int stride_batch_s = stride_kvhead_s * n_kv_heads;
    dim3 grid = dim3(bsz, n_kv_heads >> 2, n_blocks);
    dim3 block = dim3(d_block >> 1, 4, 1);
    if (reference){
        k_cache_compute_kernel<true><<<grid, block>>>(k_cache_first_8_data,
                                                      k_cache_mid_4_data,
                                                      k_cache_last_4_data,
                                                      k_column_max_multiply_q_data,
                                                      k_column_max_multiply_q_max_data,
                                                      q_data,
                                                      s_data,
                                                      stride_batch_k_cache,
                                                      stride_kvhead_k_cache,
                                                      stride_block_k_cache,
                                                      stride_dim_k_cache,
                                                      stride_batch_q,
                                                      stride_kvhead_q,
                                                      stride_batch_k_column_max_multiply_q_max,
                                                      stride_batch_s,
                                                      stride_kvhead_s,
                                                      sqrt_dim,
                                                      d_head,
                                                      d_block);
    } else {
        k_cache_compute_kernel<false><<<grid, block>>>(k_cache_first_8_data,
                                                       k_cache_mid_4_data,
                                                       k_cache_last_4_data,
                                                       k_column_max_multiply_q_data,
                                                       k_column_max_multiply_q_max_data,
                                                       q_data,
                                                       s_data,
                                                       stride_batch_k_cache,
                                                       stride_kvhead_k_cache,
                                                       stride_block_k_cache,
                                                       stride_dim_k_cache,
                                                       stride_batch_q,
                                                       stride_kvhead_q,
                                                       stride_batch_k_column_max_multiply_q_max,
                                                       stride_batch_s,
                                                       stride_kvhead_s,
                                                       sqrt_dim,
                                                       d_head,
                                                       d_block);
    }
//    cudaDeviceSynchronize();
    return s;
}