#include "AlignedKV.h"

struct KCache {
    // 3个数据
    unsigned int* __restrict__ k_cache_first_8_data; // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 4)
    unsigned int* __restrict__ k_cache_mid_4_data;    // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    unsigned int* __restrict__ k_cache_last_4_data;   // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    // shape data 以int4为准
    unsigned int stride_d_head;
    unsigned int stride_seq_per_columnblock;
    unsigned int stride_n_local_kv_heads;
    unsigned int stride_bsz;

    // 构造函数
    KCache(const torch::Tensor& first, const torch::Tensor& mid, const torch::Tensor& last) {
        // 确保输入的张量是uint32类型
        TORCH_CHECK(first.dtype() == torch::kUInt32, "first tensor must be of type torch.uint32");
        TORCH_CHECK(mid.dtype() == torch::kUInt32, "mid tensor must be of type torch.uint32");
        TORCH_CHECK(last.dtype() == torch::kUInt32, "last tensor must be of type torch.uint32");

        // 将数据指针初始化为Tensor的data指针
        k_cache_first_8_data = first.data_ptr<unsigned int>();
        k_cache_mid_4_data = mid.data_ptr<unsigned int>();
        k_cache_last_4_data = last.data_ptr<unsigned int>();

        // 初始化步幅
        stride_bsz = last.stride(0);
        stride_n_local_kv_heads = last.stride(1);
        stride_seq_per_columnblock = last.stride(2);
        stride_d_head = last.stride(3);
    }
    // 访问函数1 以int4为准
    __device__ __forceinline__ unsigned int get_bias_4(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq_per_columnblock, const unsigned int d_head, const unsigned int seq_rest_columnblock) {
        return bsz * stride_bsz + n_local_kv_heads * stride_n_local_kv_heads + seq_per_columnblock * stride_seq_per_columnblock + d_head * stride_d_head + seq_rest_columnblock / 8;
    }
    // 访问函数2 以int4为准
    __device__ __forceinline__ unsigned int get_bias_4(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq, const unsigned int d_head) {
        unsigned int seq_per_columnblock = seq / (COLUMN_BLOCK * 8);
        unsigned int seq_rest_columnblock = seq % COLUMN_BLOCK;
        return bsz * stride_bsz + n_local_kv_heads * stride_n_local_kv_heads + seq_per_columnblock * stride_seq_per_columnblock + d_head * stride_d_head + seq_rest_columnblock;
    }
    // 保存8个uint16数，相当于4个uint32数
    __device__ __forceinline__ void save_8_uint16(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq_per_columnblock, const unsigned int start_seq_columnblock, const unsigned int d_head, const unsigned int seq_rest_columnblock, const TensorNormal k_new) {
        unsigned int seq = seq_per_columnblock * COLUMN_BLOCK + seq_rest_columnblock;
        unsigned int num_1 = k_new.get_uint32(bsz, n_local_kv_heads, d_head, seq);
        unsigned int num_2 = k_new.get_uint32(bsz, n_local_kv_heads, d_head, seq + 2);
        unsigned int num_3 = k_new.get_uint32(bsz, n_local_kv_heads, d_head, seq + 4);
        unsigned int num_4 = k_new.get_uint32(bsz, n_local_kv_heads, d_head, seq + 6);
        unsigned int bias = get_bias_4(bsz, n_local_kv_heads, seq_per_columnblock + start_seq_columnblock, d_head, seq_rest_columnblock);
        // first 8
        k_cache_first_8_data[bias * 2] = ((num_1 & 0xFF00FF00) >> 8) | ((num_2 & 0xFF00FF00));
        k_cache_first_8_data[bias * 2 + 1] = ((num_3 & 0xFF00FF00) >> 8) | ((num_4 & 0xFF00FF00));
        // mid 4
        k_cache_mid_4_data[bias] = ((num_1 & 0x00F000F0) >> 4) | ((num_2 & 0x00F000F0)) | ((num_3 & 0x00F000F0) << 4) | ((num_4 & 0x00F000F0) << 8);
        // last 4
        k_cache_last_4_data[bias] = ((num_1 & 0x000F000F)) | ((num_2 & 0x000F000F) << 4) | ((num_3 & 0x000F000F) << 8) | ((num_4 & 0x000F000F) << 12);
    }
    // 读取8个uint16数，相当于4个uint32数
    __device__ __forceinline__ void load_8_uint16(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq_per_columnblock, const unsigned int d_head, const unsigned int seq_rest_columnblock, const short diff, unsigned int* shared_mem, uint4 &num) {
        // level: 0-3: all, 4-7: ignore last 4, 8-11: ignore mid 4, 11-: ignore first 4
        // shared_mem: [0]: first 8, [1]: mid 4 & last 4
        // shared_mem: 0: 0x00000000, 1: 0x88888888
        unsigned int bias = get_bias_4(bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock);
        // first 8
        unsigned int* first_8_p1 = diff >= 11 ? (shared_mem) : (k_cache_first_8_data + bias * 2);
        unsigned int* first_8_p2 = diff >= 11 ? (shared_mem) : (k_cache_first_8_data + bias * 2 + 1);
        // mid 4
        unsigned int* mid_4_p = diff >= 8 ? (shared_mem + 1) : (k_cache_mid_4_data + bias);
        // last 4
        unsigned int* last_4_p = diff >= 4 ? (shared_mem + 1) : (k_cache_last_4_data + bias);
        // 读取数据
        num.x = (((*first_8_p1) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x000F000F) << 4) | ((*last_4_p) & 0x000F000F);
        num.y = (((*first_8_p1) & 0xFF00FF00)) | (((*mid_4_p) & 0x00F000F0)) | (((*last_4_p) & 0x00F000F0) >> 4);
        num.z = (((*first_8_p2) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x0F000F00) >> 4) | (((*last_4_p) & 0x0F000F00) >> 8);
        num.w = (((*first_8_p2) & 0xFF00FF00)) | (((*mid_4_p) & 0xF000F000) >> 8) | (((*last_4_p) & 0xF000F000) >> 12);
    }
    __device__ __forceinline__ void load_8_uint16(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq_per_columnblock, const unsigned int d_head, const unsigned int seq_rest_columnblock, const short diff, unsigned int* shared_mem, unsigned int* to_mem) {
        // level: 0-3: all, 4-7: ignore last 4, 8-11: ignore mid 4, 11-: ignore first 4
        // shared_mem: [0]: first 8, [1]: mid 4 & last 4
        // shared_mem: 0: 0x00000000, 1: 0x88888888
        unsigned int bias = get_bias_4(bsz, n_local_kv_heads, seq_per_columnblock, d_head, seq_rest_columnblock);
        // first 8
        unsigned int* first_8_p1 = diff >= 11 ? (shared_mem) : (k_cache_first_8_data + bias * 2);
        unsigned int* first_8_p2 = diff >= 11 ? (shared_mem) : (k_cache_first_8_data + bias * 2 + 1);
        // mid 4
        unsigned int* mid_4_p = diff >= 8 ? (shared_mem + 1) : (k_cache_mid_4_data + bias);
        // last 4
        unsigned int* last_4_p = diff >= 4 ? (shared_mem + 1) : (k_cache_last_4_data + bias);
        // 读取数据
        to_mem[0] = (((*first_8_p1) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x000F000F) << 4) | ((*last_4_p) & 0x000F000F);
        to_mem[1] = (((*first_8_p1) & 0xFF00FF00)) | (((*mid_4_p) & 0x00F000F0)) | (((*last_4_p) & 0x00F000F0) >> 4);
        to_mem[2] = (((*first_8_p2) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x0F000F00) >> 4) | (((*last_4_p) & 0x0F000F00) >> 8);
        to_mem[3] = (((*first_8_p2) & 0xFF00FF00)) | (((*mid_4_p) & 0xF000F000) >> 8) | (((*last_4_p) & 0xF000F000) >> 12);
    }
};

// k_new: (bsz, n_local_kv_heads, d_head, seqlen)
// k_cache: (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK)
// assert seqlen % 2 == 0, expect seqlen % COLUMN_BLOCK == 0
// 线程按x连续，x维度放最连续的维度
// block (8, n_local_kv_heads, bsz)
// thread (COLUMN_BLOCK / 8, d_head / 8, seqlen // COLUMN_BLOCK)
__global__ void k_cache_save_kernel(TensorNormal k_new, KCache k_cache, const unsigned int start_block) {
    const unsigned int bsz_id = blockIdx.z;
    const unsigned int n_local_kv_heads_id = blockIdx.y;
    const unsigned int d_head_id = threadIdx.y + blockIdx.x * blockDim.y;
    const unsigned int seq_per_columnblock_id = threadIdx.z;
    const unsigned int seq_rest_columnblock_id = threadIdx.x * 8;
    // 保存数据
    k_cache.save_8_uint16(bsz_id, n_local_kv_heads_id, seq_per_columnblock_id, start_block, d_head_id, seq_rest_columnblock_id, k_new);
}

void k_cache_save(
    torch::Tensor& k_new,                  // (bsz, n_local_kv_heads, d_head, seqlen)
    torch::Tensor& k_cache_first_8,       // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 4)
    torch::Tensor& k_cache_mid_4,         // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    torch::Tensor& k_cache_last_4,        // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    const unsigned int start_block             // 开始的block
) {
    // 推出维度信息
    const unsigned int bsz = k_cache_first_8.size(0);                  // 从第一个维度获取批次大小
    const unsigned int n_local_kv_heads = k_cache_first_8.size(1);    // 从第二个维度获取局部键值头的数量
    const unsigned int d_head = k_cache_first_8.size(3);               // 从第四个维度获取每个头的维度
    const unsigned int seqlen = k_new.size(3);                         // 从第四个维度获取序列长度

    // Check if the tensors are on the GPU
    TORCH_CHECK(k_new.is_cuda(), "k_new must be a CUDA tensor");
    TORCH_CHECK(k_cache_first_8.is_cuda(), "k_cache_first_8 must be a CUDA tensor");
    TORCH_CHECK(k_cache_mid_4.is_cuda(), "k_cache_mid_4 must be a CUDA tensor");
    TORCH_CHECK(k_cache_last_4.is_cuda(), "k_cache_last_4 must be a CUDA tensor");

    // Set the device for the kernel launch
    cudaSetDevice(k_new.get_device());

    // Ensure the tensors have the correct sizes and types
    TORCH_CHECK(k_new.dtype() == torch::kHalf, "k_new must be of type torch.half");
    TORCH_CHECK(k_cache_first_8.dtype() == torch::kUInt32, "k_cache_first_8 must be of type torch.uint32");
    TORCH_CHECK(k_cache_mid_4.dtype() == torch::kUInt32, "k_cache_mid_4 must be of type torch.uint32");
    TORCH_CHECK(k_cache_last_4.dtype() == torch::kUInt32, "k_cache_last_4 must be of type torch.uint32");

    // Check if the tensors have the correct sizes
    TORCH_CHECK(seqlen % COLUMN_BLOCK == 0, "seqlen must be divisible by COLUMN_BLOCK");
    TORCH_CHECK(seqlen <= (k_cache_first_8.size(2) - start_block) * COLUMN_BLOCK, "seqlen must be less than or equal to (k_cache_first_8.size(2) - start_block) * COLUMN_BLOCK");
    TORCH_CHECK(d_head % 8 == 0, "d_head must be divisible by 8");
    TORCH_CHECK(k_cache_first_8.size(4) == COLUMN_BLOCK / 4, "k_cache_first_8.size(4) must be equal to COLUMN_BLOCK / 4");
    TORCH_CHECK(k_cache_mid_4.size(4) == COLUMN_BLOCK / 8, "k_cache_mid_4.size(4) must be equal to COLUMN_BLOCK / 8");
    TORCH_CHECK(k_cache_last_4.size(4) == COLUMN_BLOCK / 8, "k_cache_last_4.size(4) must be equal to COLUMN_BLOCK / 8");

    // Dimension check for k_new
    TORCH_CHECK(k_new.size(0) == bsz && k_new.size(1) == n_local_kv_heads && k_new.size(2) == d_head && k_new.size(3) == seqlen, "k_new has incorrect dimensions");

    // Create TensorNormal and KCache objects
    TensorNormal k_new_tensor(k_new);
    KCache k_cache(k_cache_first_8, k_cache_mid_4, k_cache_last_4);

    // Calculate grid and block dimensions
    const unsigned int threads_per_block_x = COLUMN_BLOCK / 8; // Assuming COLUMN_BLOCK is divisible by 8
    const unsigned int threads_per_block_y = d_head / 8;       // Assuming d_head is divisible by 8
    const unsigned int threads_per_block_z = seqlen / COLUMN_BLOCK; // Assuming seqlen is divisible by COLUMN_BLOCK

    dim3 grid(8, n_local_kv_heads, bsz); // Grid dimensions
    dim3 block(threads_per_block_x, threads_per_block_y, threads_per_block_z); // Block dimensions

    // Launch the kernel
    k_cache_save_kernel<<<grid, block>>>(k_new_tensor, k_cache, start_block);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in k_cache_save_kernel: " << cudaGetErrorString(err) << std::endl;
    }
}

// 计算qkT=s
// q: (bsz, n_local_kv_heads, n_gqa_group, d_head)
// k_cache: (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK / 4 or 8)
// s: (bsz, n_local_kv_heads, n_gqa_group, seqlen // COLUMN_BLOCK, COLUMN_BLOCK)
// diff_array: (1, bsz, n_local_kv_heads, d_head)
// block (seqlen // COLUMN_BLOCK, n_local_kv_heads, bsz)
// thread (COLUMN_BLOCK / 8, dim / dim_part, 1)
// set dim_part = 16, then dim_part_num = 8
template <bool reference, bool use_tensorcore, unsigned int gqa, unsigned int dim_part, unsigned int dim_part_num>
__global__ void k_cache_compute_kernel(TensorNormal q, KCache k_cache, TensorNormal s, TensorNormal diff_array) {
    // 每个线程负责计算headdim的dim_part长度
    const unsigned int bsz_id = blockIdx.z;
    const unsigned int n_local_kv_heads_id = blockIdx.y;
    const unsigned int seq_per_columnblock_id = blockIdx.x;
    const unsigned int seq_rest_columnblock_id = threadIdx.x * 8;
    const unsigned int d_head_part_begin = threadIdx.y * dim_part;
    constexpr unsigned int warp_num = (dim_part_num * COLUMN_BLOCK / 8) / WARP_SIZE;
    const unsigned int warp_id = threadIdx.y / (WARP_SIZE / (COLUMN_BLOCK / 8));
    constexpr unsigned int lest_size_notensorcore = ((dim_part_num * gqa * COLUMN_BLOCK / 2) + (TENSORCORE_M * COLUMN_BLOCK) - 1) / (TENSORCORE_M * COLUMN_BLOCK);
    constexpr unsigned int lest_size = use_tensorcore ? warp_num : lest_size_notensorcore;

    __align__(64) __shared__ half cache_key[dim_part * dim_part_num][COLUMN_BLOCK];
    __align__(64) __shared__ half cache_query[TENSORCORE_M][dim_part * dim_part_num];
    __align__(64) __shared__ float cache_result[lest_size][TENSORCORE_M][COLUMN_BLOCK];
    __shared__ unsigned int shared_mem[2];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_mem[0] = 0x00000000;
        shared_mem[1] = 0x88888888;
    }

    // load q
    // q(gqa, dim_part * dim_part_num), thread (COLUMN_BLOCK / 8, dim_part_num, 1)
    static_assert((gqa * dim_part * dim_part_num) % ((COLUMN_BLOCK / 8) * dim_part_num) == 0, "gqa * dim_part * dim_part_num must be divisible by (COLUMN_BLOCK / 8) * dim_part_num");
    #pragma unroll
    for (unsigned int bias = 0; bias < gqa * dim_part_num * dim_part; bias += (COLUMN_BLOCK * dim_part_num / 8)) {
        cache_query[0][bias + threadIdx.y * (COLUMN_BLOCK / 8) + threadIdx.x] = q.get_half(bsz_id, n_local_kv_heads_id, 0, bias + threadIdx.y * (COLUMN_BLOCK / 8) + threadIdx.x);
    }

    // init cache_key
    if constexpr (use_tensorcore) {
        static_assert((TENSORCORE_M * dim_part * dim_part_num) % ((COLUMN_BLOCK / 8) * dim_part_num) == 0, "TENSORCORE_M * dim_part * dim_part_num must be divisible by (COLUMN_BLOCK / 8) * dim_part_num");
        #pragma unroll
        for (unsigned int bias = gqa * dim_part_num * dim_part; bias < TENSORCORE_M * dim_part_num * dim_part; bias += (COLUMN_BLOCK * dim_part_num / 8)) {
            cache_query[0][bias + threadIdx.y * (COLUMN_BLOCK / 8) + threadIdx.x] = __ushort_as_half(0x0000);
        }
    }

    // init cache_result
    if constexpr (use_tensorcore) {
        static_assert((TENSORCORE_M * COLUMN_BLOCK) % (COLUMN_BLOCK * dim_part_num / 8) == 0, "TENSORCORE_M * COLUMN_BLOCK must be divisible by COLUMN_BLOCK * dim_part_num / 8");
        #pragma unroll
        for (unsigned int i = 0; i < warp_num; i++) {
            #pragma unroll
            for (unsigned int bias = 0; bias < TENSORCORE_M * COLUMN_BLOCK; bias += (COLUMN_BLOCK * dim_part_num / 8)) {
                cache_result[i][0][bias + threadIdx.y * (COLUMN_BLOCK / 8) + threadIdx.x] = 0.0;
            }
        }
    }

    // 线程级别的同步
    __syncthreads();

    half2 temp_result[gqa][4] = {};

    static_assert(dim_part % TENSORCORE_K == 0, "dim_part must be divisible by TENSORCORE_K");
    #pragma unroll
    for (unsigned int outer = 0; outer < dim_part; outer += TENSORCORE_K) {
        // d_head维
        #pragma unroll
        for (unsigned int i = outer; i < outer + TENSORCORE_K; i++) {
            // d_head维
            unsigned int d_head_id = d_head_part_begin + i;
            short diff;
            uint4 temp_num;

            // judge diff
            if constexpr (reference) {
                // reference
                diff = 0;
            } else {
                // my method
                diff = diff_array.get_short(0, bsz_id, n_local_kv_heads_id, d_head_id);
            }

            // load k
            // k_cache(bsz, n_local_kv_heads, seq_per_columnblock, d_head, COLUMN_BLOCK / 4 or 8)
            if constexpr (use_tensorcore) {
                // use tensorcore
                // save to cache_key
                unsigned int* to_mem = reinterpret_cast<unsigned int*>(&cache_key[d_head_id][seq_rest_columnblock_id]);
                k_cache.load_8_uint16(bsz_id, n_local_kv_heads_id, seq_per_columnblock_id, d_head_id, seq_rest_columnblock_id, diff, shared_mem, to_mem);
            } else {
                // not use tensorcore
                // save to temp_num
                k_cache.load_8_uint16(bsz_id, n_local_kv_heads_id, seq_per_columnblock_id, d_head_id, seq_rest_columnblock_id, diff, shared_mem, temp_num);
            }

            if (diff >= 11) {
                continue;
            }

            // compute if not use tensorcore
            if constexpr (!use_tensorcore) {
                #pragma unroll
                for (unsigned int j = 0; j < gqa; j++) {
                    temp_result[j][0] = __hfma2(__half2half2(cache_query[j][d_head_id]), reinterpret_cast<half2&>(temp_num.x), temp_result[j][0]);
                    temp_result[j][1] = __hfma2(__half2half2(cache_query[j][d_head_id]), reinterpret_cast<half2&>(temp_num.y), temp_result[j][1]);
                    temp_result[j][2] = __hfma2(__half2half2(cache_query[j][d_head_id]), reinterpret_cast<half2&>(temp_num.z), temp_result[j][2]);
                    temp_result[j][3] = __hfma2(__half2half2(cache_query[j][d_head_id]), reinterpret_cast<half2&>(temp_num.w), temp_result[j][3]);
                }
            }
        }

        // compute if use tensorcore
        if constexpr (use_tensorcore) {
            static_assert(COLUMN_BLOCK % TENSORCORE_N == 0, "COLUMN_BLOCK must be divisible by TENSORCORE_N");
            #pragma unroll
            for (unsigned int i = 0; i < COLUMN_BLOCK; i += TENSORCORE_N) {
                // COLUMN i~i+TENSORCORE_N

                // init fragment
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TENSORCORE_M, TENSORCORE_N, TENSORCORE_K, half, nvcuda::wmma::row_major> a_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TENSORCORE_M, TENSORCORE_N, TENSORCORE_K, half, nvcuda::wmma::row_major> b_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TENSORCORE_M, TENSORCORE_N, TENSORCORE_K, float> c_frag;

                // load c
                nvcuda::wmma::load_matrix_sync(c_frag, &cache_result[warp_id][0][i], COLUMN_BLOCK, nvcuda::wmma::mem_row_major);

                // warp_d_head_part_begin = warp_id * (dim_part_num / warp_num) * dim_part
                #pragma unroll
                for (unsigned int j = warp_id * (dim_part_num / warp_num) * dim_part; j < (warp_id + 1) * (dim_part_num / warp_num) * dim_part; j += dim_part) {
                    // load a
                    nvcuda::wmma::load_matrix_sync(a_frag, &cache_query[0][j + outer], dim_part * dim_part_num);

                    // load b
                    nvcuda::wmma::load_matrix_sync(b_frag, &cache_key[j + outer][i], COLUMN_BLOCK);

                    // compute
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                // store c
                nvcuda::wmma::store_matrix_sync(&cache_result[warp_id][0][i], c_frag, COLUMN_BLOCK, nvcuda::wmma::mem_row_major);
            }
        }
    }

    // merge result and store in s
    if constexpr (use_tensorcore) {
        __syncthreads();

        // reduce and store
        // cache_result[j][gqa][COLUMN_BLOCK] to s(bsz, n_local_kv_heads, n_gqa_group, seqlen // COLUMN_BLOCK, COLUMN_BLOCK)
        for (unsigned int i = threadIdx.y * (COLUMN_BLOCK / 8) + threadIdx.x; i < gqa * COLUMN_BLOCK; i += (COLUMN_BLOCK / 8) * dim_part_num) {
            float acc = 0.0;
            #pragma unroll
            for (unsigned int j = 0; j < warp_num; j++) {
                acc += cache_result[j][0][i];
            }
            unsigned int flatten_gqa_id = i / COLUMN_BLOCK;
            unsigned int flatten_seq_id = i % COLUMN_BLOCK;
            s.save_half(bsz_id, n_local_kv_heads_id, flatten_gqa_id, seq_per_columnblock_id, flatten_seq_id, __float2half(acc));
        }
    } else {
        // change cache_result to reinterpret_cast<half*>(cache_result)[dim_part_num][gqa][COLUMN_BLOCK]
        // also reinterpret_cast<half2*>(cache_result)[dim_part_num][gqa][COLUMN_BLOCK // 2]
        static_assert((dim_part_num * gqa * COLUMN_BLOCK / 2) <= sizeof(cache_result) / sizeof(half2), "dim_part_num * gqa * COLUMN_BLOCK / 2 must be less than or equal to sizeof(cache_result) / sizeof(half2)");

        // copy number
        #pragma unroll
        for (unsigned int i = 0; i < gqa; i++) {
            reinterpret_cast<half2*>(cache_result)[threadIdx.y * (gqa * COLUMN_BLOCK / 2) + i * (COLUMN_BLOCK / 2) + threadIdx.x * 4] = temp_result[i][0];
            reinterpret_cast<half2*>(cache_result)[threadIdx.y * (gqa * COLUMN_BLOCK / 2) + i * (COLUMN_BLOCK / 2) + threadIdx.x * 4 + 1] = temp_result[i][1];
            reinterpret_cast<half2*>(cache_result)[threadIdx.y * (gqa * COLUMN_BLOCK / 2) + i * (COLUMN_BLOCK / 2) + threadIdx.x * 4 + 2] = temp_result[i][2];
            reinterpret_cast<half2*>(cache_result)[threadIdx.y * (gqa * COLUMN_BLOCK / 2) + i * (COLUMN_BLOCK / 2) + threadIdx.x * 4 + 3] = temp_result[i][3];
        }

        __syncthreads();

        // reduce and store
        for (unsigned int i = threadIdx.y * (COLUMN_BLOCK / 8) + threadIdx.x; i < gqa * COLUMN_BLOCK / 2; i += (COLUMN_BLOCK / 8) * dim_part_num) {
            half2 acc = __half2half2(__ushort_as_half(0x0000));
            #pragma unroll
            for (unsigned int j = 0; j < dim_part_num * gqa * COLUMN_BLOCK / 2; j += gqa * COLUMN_BLOCK / 2) {
                acc = __hadd2(acc, reinterpret_cast<half2*>(cache_result)[j + i]);
            }
            unsigned int flatten_gqa_id = i / (COLUMN_BLOCK / 2);
            unsigned int flatten_seq_id = (i % (COLUMN_BLOCK / 2)) * 2;
            s.save_uint32(bsz_id, n_local_kv_heads_id, flatten_gqa_id, seq_per_columnblock_id, flatten_seq_id, reinterpret_cast<unsigned int&>(acc));
        }
    }

    __syncthreads();
}

torch::Tensor k_cache_compute(
    torch::Tensor& q,                  // (bsz, n_local_kv_heads, n_gqa_group, d_head)
    torch::Tensor& k_cache_first_8,    // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 4)
    torch::Tensor& k_cache_mid_4,      // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    torch::Tensor& k_cache_last_4,     // (bsz, n_local_kv_heads, seqlen // COLUMN_BLOCK, d_head, COLUMN_BLOCK // 8)
    torch::Tensor& diff_array,         // (bsz, n_local_kv_heads, d_head)
    const unsigned int column_block_num, 
    bool reference,
    bool use_tensorcore
) {
    // 推出维度信息
    const unsigned int bsz = q.size(0);                               // 从第一个维度获取批次大小
    const unsigned int n_local_kv_heads = q.size(1);                  // 从第二个维度获取局部键值头的数量
    const unsigned int n_gqa_group = q.size(2);                       // 从第三个维度获取gqa的数量
    const unsigned int d_head = q.size(3);                            // 从第四个维度获取每个头的维度

    // Check if the tensors are on the GPU
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_cache_first_8.is_cuda(), "k_cache_first_8 must be a CUDA tensor");
    TORCH_CHECK(k_cache_mid_4.is_cuda(), "k_cache_mid_4 must be a CUDA tensor");
    TORCH_CHECK(k_cache_last_4.is_cuda(), "k_cache_last_4 must be a CUDA tensor");
    TORCH_CHECK(diff_array.is_cuda(), "diff_array must be a CUDA tensor");

    // Set the device context for the current device
    cudaSetDevice(k_cache_first_8.get_device());

    // Ensure the tensors have the correct sizes and types
    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be of type torch.half");
    TORCH_CHECK(k_cache_first_8.dtype() == torch::kUInt32, "k_cache_first_8 must be of type torch.uint32");
    TORCH_CHECK(k_cache_mid_4.dtype() == torch::kUInt32, "k_cache_mid_4 must be of type torch.uint32");
    TORCH_CHECK(k_cache_last_4.dtype() == torch::kUInt32, "k_cache_last_4 must be of type torch.uint32");
    TORCH_CHECK(diff_array.dtype() == torch::kShort, "diff_array must be of type torch.short");

    // Check if the tensors have the correct sizes
    TORCH_CHECK(d_head == 128, "d_head must be equal to 128, else not supported");
    TORCH_CHECK(n_gqa_group == 1 || n_gqa_group == 2 || n_gqa_group == 3 || n_gqa_group == 4 || n_gqa_group == 8, "n_gqa_group must be one of the following values: 1, 2, 3, 4, or 8");
    TORCH_CHECK(column_block_num <= k_cache_first_8.size(2), "column_block_num must be less than or equal to k_cache_first_8.size(2)");
    TORCH_CHECK(k_cache_first_8.size(4) == COLUMN_BLOCK / 4, "k_cache_first_8.size(4) must be equal to COLUMN_BLOCK / 4");
    TORCH_CHECK(k_cache_mid_4.size(4) == COLUMN_BLOCK / 8, "k_cache_mid_4.size(4) must be equal to COLUMN_BLOCK / 8");
    TORCH_CHECK(k_cache_last_4.size(4) == COLUMN_BLOCK / 8, "k_cache_last_4.size(4) must be equal to COLUMN_BLOCK / 8");

    // Create output tensor
    torch::Tensor s = torch::empty({bsz, n_local_kv_heads, n_gqa_group, column_block_num, COLUMN_BLOCK}, torch::dtype(torch::kHalf).device(k_cache_first_8.device()));

    // Reshape diff_array to (1, bsz, n_local_kv_heads, d_head)
    diff_array = diff_array.view({1, bsz, n_local_kv_heads, d_head});

    // Create TensorNormal and KCache objects
    TensorNormal q_tensor(q);
    TensorNormal s_tensor(s);
    KCache k_cache(k_cache_first_8, k_cache_mid_4, k_cache_last_4);
    TensorNormal diff_array_tensor(diff_array);

    // Calculate grid and block dimensions
    // block (seqlen // COLUMN_BLOCK, n_local_kv_heads, bsz)
    // thread (COLUMN_BLOCK / 8, dim / dim_part, 1)
    dim3 grid(column_block_num, n_local_kv_heads, bsz); // Grid dimensions
    dim3 block(COLUMN_BLOCK / 8, d_head / 16, 1); // Block dimensions

    // Launch the kernel
    if (reference) {
        if (use_tensorcore) {
            switch (n_gqa_group) {
                case 1:
                    k_cache_compute_kernel<true, true, 1, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 2:
                    k_cache_compute_kernel<true, true, 2, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 3:
                    k_cache_compute_kernel<true, true, 3, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 4:
                    k_cache_compute_kernel<true, true, 4, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 8:
                    k_cache_compute_kernel<true, true, 8, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
            }
        } else {
            switch (n_gqa_group) {
                case 1:
                    k_cache_compute_kernel<true, false, 1, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 2:
                    k_cache_compute_kernel<true, false, 2, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 3:
                    k_cache_compute_kernel<true, false, 3, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 4:
                    k_cache_compute_kernel<true, false, 4, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 8:
                    k_cache_compute_kernel<true, false, 8, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
            }
        }
    } else {
        if (use_tensorcore) {
            switch (n_gqa_group) {
                case 1:
                    k_cache_compute_kernel<false, true, 1, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 2:
                    k_cache_compute_kernel<false, true, 2, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 3:
                    k_cache_compute_kernel<false, true, 3, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 4:
                    k_cache_compute_kernel<false, true, 4, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 8:
                    k_cache_compute_kernel<false, true, 8, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
            }
        } else {
            switch (n_gqa_group) {
                case 1:
                    k_cache_compute_kernel<false, false, 1, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 2:
                    k_cache_compute_kernel<false, false, 2, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 3:
                    k_cache_compute_kernel<false, false, 3, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 4:
                    k_cache_compute_kernel<false, false, 4, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
                case 8:
                    k_cache_compute_kernel<false, false, 8, 16, 8><<<grid, block>>>(q_tensor, k_cache, s_tensor, diff_array_tensor);
                    break;
            }
        }
    }
    
    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in k_cache_compute: " << cudaGetErrorString(err) << std::endl;
    }

    return s.view({bsz, n_local_kv_heads, n_gqa_group, column_block_num * COLUMN_BLOCK});
}