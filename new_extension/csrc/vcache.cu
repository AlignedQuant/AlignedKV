#include "AlignedKV.h"

struct VCache {
    // 3个数据
    unsigned int* __restrict__ v_cache_first_8_data; // (bsz, n_local_kv_heads, seqlen, d_head // 4)
    unsigned int* __restrict__ v_cache_mid_4_data;    // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    unsigned int* __restrict__ v_cache_last_4_data;   // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    // shape data 以int4为准
    unsigned int stride_seqlen;
    unsigned int stride_n_local_kv_heads;
    unsigned int stride_bsz;

    // 构造函数
    VCache(const torch::Tensor& first, const torch::Tensor& mid, const torch::Tensor& last) {
        // 确保输入的张量是uint32类型
        TORCH_CHECK(first.dtype() == torch::kUInt32, "first tensor must be of type torch.uint32");
        TORCH_CHECK(mid.dtype() == torch::kUInt32, "mid tensor must be of type torch.uint32");
        TORCH_CHECK(last.dtype() == torch::kUInt32, "last tensor must be of type torch.uint32");

        // 将数据指针初始化为Tensor的data指针
        v_cache_first_8_data = first.data_ptr<unsigned int>();
        v_cache_mid_4_data = mid.data_ptr<unsigned int>();
        v_cache_last_4_data = last.data_ptr<unsigned int>();

        // 初始化步幅
        stride_bsz = last.stride(0);
        stride_n_local_kv_heads = last.stride(1);
        stride_seqlen = last.stride(2);
    }
    // 访问函数 以int4为准
    __device__ __forceinline__ unsigned int get_bias_4(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq, const unsigned int d_head) {
        return bsz * stride_bsz + n_local_kv_heads * stride_n_local_kv_heads + seq * stride_seqlen + d_head / 8;
    }
    // 保存8个uint16数，相当于4个uint32数
    __device__ __forceinline__ void save_8_uint16(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq, const unsigned int start_seq, const unsigned int d_head, const TensorNormal v_new) {
        unsigned int num_1 = v_new.get_uint32(bsz, n_local_kv_heads, seq, d_head);
        unsigned int num_2 = v_new.get_uint32(bsz, n_local_kv_heads, seq, d_head + 2);
        unsigned int num_3 = v_new.get_uint32(bsz, n_local_kv_heads, seq, d_head + 4);
        unsigned int num_4 = v_new.get_uint32(bsz, n_local_kv_heads, seq, d_head + 6);
        unsigned int bias = get_bias_4(bsz, n_local_kv_heads, seq + start_seq, d_head);
        // first 8
        v_cache_first_8_data[bias * 2] = ((num_1 & 0xFF00FF00) >> 8) | ((num_2 & 0xFF00FF00));
        v_cache_first_8_data[bias * 2 + 1] = ((num_3 & 0xFF00FF00) >> 8) | ((num_4 & 0xFF00FF00));
        // mid 4
        v_cache_mid_4_data[bias] = ((num_1 & 0x00F000F0) >> 4) | ((num_2 & 0x00F000F0)) | ((num_3 & 0x00F000F0) << 4) | ((num_4 & 0x00F000F0) << 8);
        // last 4
        v_cache_last_4_data[bias] = ((num_1 & 0x000F000F)) | ((num_2 & 0x000F000F) << 4) | ((num_3 & 0x000F000F) << 8) | ((num_4 & 0x000F000F) << 12);
    }
    // 读取8个uint16数，相当于4个uint32数
    __device__ __forceinline__ void load_8_uint16(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq, const unsigned int d_head, const short diff, unsigned int* shared_mem, uint4 &num) {
        // shared_mem: [0]: first 8, [1]: mid 4 & last 4
        // shared_mem: 0: 0x00000000, 1: 0x88888888
        unsigned int bias = get_bias_4(bsz, n_local_kv_heads, seq, d_head);
        // first 8
        unsigned int* first_8_p1 = diff >= 11 ? (shared_mem) : (v_cache_first_8_data + bias * 2);
        unsigned int* first_8_p2 = diff >= 11 ? (shared_mem) : (v_cache_first_8_data + bias * 2 + 1);
        // mid 4
        unsigned int* mid_4_p = diff >= 8 ? (shared_mem + 1) : (v_cache_mid_4_data + bias);
        // last 4
        unsigned int* last_4_p = diff >= 4 ? (shared_mem + 1) : (v_cache_last_4_data + bias);
        // 读取数据
        num.x = (((*first_8_p1) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x000F000F) << 4) | ((*last_4_p) & 0x000F000F);
        num.y = (((*first_8_p1) & 0xFF00FF00)) | (((*mid_4_p) & 0x00F000F0)) | (((*last_4_p) & 0x00F000F0) >> 4);
        num.z = (((*first_8_p2) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x0F000F00) >> 4) | (((*last_4_p) & 0x0F000F00) >> 8);
        num.w = (((*first_8_p2) & 0xFF00FF00)) | (((*mid_4_p) & 0xF000F000) >> 8) | (((*last_4_p) & 0xF000F000) >> 12);
    }
    __device__ __forceinline__ void load_8_uint16(const unsigned int bsz, const unsigned int n_local_kv_heads, const unsigned int seq, const unsigned int d_head, const short diff, unsigned int* shared_mem, unsigned int* to_mem) {
        // shared_mem: [0]: first 8, [1]: mid 4 & last 4
        // shared_mem: 0: 0x00000000, 1: 0x88888888
        unsigned int bias = get_bias_4(bsz, n_local_kv_heads, seq, d_head);
        // first 8
        unsigned int* first_8_p1 = diff >= 11 ? (shared_mem) : (v_cache_first_8_data + bias * 2);
        unsigned int* first_8_p2 = diff >= 11 ? (shared_mem) : (v_cache_first_8_data + bias * 2 + 1);
        // mid 4
        unsigned int* mid_4_p = diff >= 8 ? (shared_mem + 1) : (v_cache_mid_4_data + bias);
        // last 4
        unsigned int* last_4_p = diff >= 4 ? (shared_mem + 1) : (v_cache_last_4_data + bias);
        // 读取数据
        to_mem[0] = (((*first_8_p1) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x000F000F) << 4) | ((*last_4_p) & 0x000F000F);
        to_mem[1] = (((*first_8_p1) & 0xFF00FF00)) | (((*mid_4_p) & 0x00F000F0)) | (((*last_4_p) & 0x00F000F0) >> 4);
        to_mem[2] = (((*first_8_p2) & 0x00FF00FF) << 8) | (((*mid_4_p) & 0x0F000F00) >> 4) | (((*last_4_p) & 0x0F000F00) >> 8);
        to_mem[3] = (((*first_8_p2) & 0xFF00FF00)) | (((*mid_4_p) & 0xF000F000) >> 8) | (((*last_4_p) & 0xF000F000) >> 12);
    }
};

// v_new: (bsz, n_local_kv_heads, seqlen, d_head)
// v_cache_first_8_data: (bsz, n_local_kv_heads, seqlen, d_head // 4)
// v_cache_mid_4_data: (bsz, n_local_kv_heads, seqlen, d_head // 8)
// v_cache_last_4_data: (bsz, n_local_kv_heads, seqlen, d_head // 8)
// assert seqlen % 8 == 0
// block (seqlen // 8, n_local_kv_heads, bsz)
// thread (d_head // 8, 8, 1)
// new block (seqlen // 32, n_local_kv_heads, bsz)
// new thread (d_head // 8, 32, 1)
// template <unsigned int data_transfer_mode>
__global__ void v_cache_save_kernel(TensorNormal v_new, VCache v_cache, const unsigned int start_seq) {
    const unsigned int bsz_id = blockIdx.z;
    const unsigned int n_local_kv_heads_id = blockIdx.y;
    // const unsigned int seq_id = blockIdx.x * 8 + threadIdx.y;
    const unsigned int seq_id = blockIdx.x * 32 + threadIdx.y;
    const unsigned int d_head_id = threadIdx.x * 8;
    // 保存数据
    v_cache.save_8_uint16(bsz_id, n_local_kv_heads_id, seq_id, start_seq, d_head_id, v_new);
}

void v_cache_save(
    torch::Tensor v_new,                    // (bsz, n_local_kv_heads, seqlen, d_head)
    torch::Tensor v_cache_first_8_data,    // (bsz, n_local_kv_heads, seqlen, d_head // 4)
    torch::Tensor v_cache_mid_4_data,      // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    torch::Tensor v_cache_last_4_data,    // (bsz, n_local_kv_heads, seqlen, d_head // 8)
    const unsigned int start_seq
) {
    // 推导维度信息
    const unsigned int bsz = v_cache_first_8_data.size(0);
    const unsigned int n_local_kv_heads = v_cache_first_8_data.size(1);
    const unsigned int seqlen = v_new.size(2);
    const unsigned int d_head = v_new.size(3);
    
    // Check if the tensors are on the GPU
    TORCH_CHECK(v_new.device().is_cuda(), "v_new must be a CUDA tensor");
    TORCH_CHECK(v_cache_first_8_data.device().is_cuda(), "v_cache_first_8_data must be a CUDA tensor");
    TORCH_CHECK(v_cache_mid_4_data.device().is_cuda(), "v_cache_mid_4_data must be a CUDA tensor");
    TORCH_CHECK(v_cache_last_4_data.device().is_cuda(), "v_cache_last_4_data must be a CUDA tensor");

    // Set the device for the kernel launch
    cudaSetDevice(v_new.get_device());

    // Check if the tensors have the correct type
    TORCH_CHECK(v_new.dtype() == torch::kHalf, "v_new must be of type torch.half");
    TORCH_CHECK(v_cache_first_8_data.dtype() == torch::kUInt32, "v_cache_first_8_data must be of type torch.uint32");
    TORCH_CHECK(v_cache_mid_4_data.dtype() == torch::kUInt32, "v_cache_mid_4_data must be of type torch.uint32");
    TORCH_CHECK(v_cache_last_4_data.dtype() == torch::kUInt32, "v_cache_last_4_data must be of type torch.uint32");

    // Check the size of the tensors
    TORCH_CHECK(seqlen % 8 == 0, "seqlen must be a multiple of 8");
    TORCH_CHECK(start_seq % 8 == 0, "start_seq must be a multiple of 8");
    TORCH_CHECK(d_head % 8 == 0, "d_head must be a multiple of 8");
    TORCH_CHECK(v_cache_first_8_data.size(3) == d_head / 4, "v_cache_first_8_data must have the correct last dimension");
    TORCH_CHECK(v_cache_mid_4_data.size(3) == d_head / 8, "v_cache_mid_4_data must have the correct last dimension");
    TORCH_CHECK(v_cache_last_4_data.size(3) == d_head / 8, "v_cache_last_4_data must have the correct last dimension");
    TORCH_CHECK(start_seq + seqlen <= v_cache_first_8_data.size(2), "start_seq + seqlen must be less than or equal to the size of v_cache_first_8_data");

    // Dimension check for v_new
    TORCH_CHECK(v_new.size(0) == bsz && v_new.size(1) == n_local_kv_heads, "v_new must have the same first two dimensions as v_cache_first_8_data");

    // Create TensorNormal object and VCache object
    TensorNormal v_new_tensor(v_new);
    VCache v_cache(v_cache_first_8_data, v_cache_mid_4_data, v_cache_last_4_data);

    // Create grid and block parameters
    // dim3 grid(seqlen / 8, n_local_kv_heads, bsz);
    // dim3 block(d_head / 8, 8, 1);
    dim3 grid(seqlen / 32, n_local_kv_heads, bsz);
    dim3 block(d_head / 8, 32, 1);
    
    // Launch the kernel
    v_cache_save_kernel<<<grid, block>>>(v_new_tensor, v_cache, start_seq);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in v_cache_save: " << cudaGetErrorString(error) << std::endl;
    }
}

// s (bsz, n_local_kv_heads, n_gqa_group, seqlen)
// v_cache: (bsz, n_local_kv_heads, seqlen, d_head // 4 or 8)
// o (bsz, n_local_kv_heads, expand_seq_acc, n_gqa_group, d_head)
// diff_level: 0-3: all, 4-7: ignore last 4, 8-11: ignore mid 4, 11-: ignore first 8
// diff_array: (1, bsz, n_local_kv_heads, seqlen)
// block(expand_seq_acc, n_local_kv_heads, bsz)
// thread(d_head // 8, expand_seq_inner_warp, expand_seq_inter_warp)
// seqlen = (expand_seq_acc, expand_seq_inter_warp, rest, expand_seq_inner_warp)
template <bool reference, bool use_tensorcore, unsigned int data_transfer_mode, 
          unsigned int gqa, unsigned int expand_seq_acc, unsigned int expand_seq_inter_warp, unsigned int expand_seq_inner_warp,
          unsigned int d_head, unsigned int tensorcore_mid_dim_group>
__global__ void v_cache_compute_kernel(TensorNormal s, VCache v_cache, TensorNormal o, TensorNormal diff_array, const unsigned int seqlen) {
    // 每个线程的参数
    const unsigned int bsz_id = blockIdx.z;
    const unsigned int n_local_kv_heads_id = blockIdx.y;
    const unsigned int expand_seq_acc_id = blockIdx.x;
    const unsigned int d_head_id = threadIdx.x * 8;
    const unsigned int seq_rest = seqlen / (expand_seq_acc * expand_seq_inter_warp * expand_seq_inner_warp);
    const unsigned int seq_begin = blockIdx.x * (expand_seq_inter_warp * expand_seq_inner_warp * seq_rest) + threadIdx.z * (expand_seq_inner_warp * seq_rest) + threadIdx.y;
    const unsigned int seq_end = seq_begin + (expand_seq_inner_warp * seq_rest);
    constexpr unsigned int seq_stride = expand_seq_inner_warp;

    // 每个线程的参数 for copy operation
    static_assert((d_head % 8) == 0, "d_head must be a multiple of 8");
    static_assert((d_head / 8) * expand_seq_inner_warp == WARP_SIZE, "d_head / 8 * expand_seq_inner_warp must be equal to WARP_SIZE");
    const unsigned int thread_id = threadIdx.z * (expand_seq_inner_warp * (d_head / 8)) + threadIdx.y * (d_head / 8) + threadIdx.x;
    constexpr unsigned int thread_num = expand_seq_inner_warp * expand_seq_inter_warp * (d_head / 8);
    constexpr unsigned int warp_num = thread_num / WARP_SIZE;
    const unsigned int warp_id = thread_id / WARP_SIZE;
    const unsigned int warp_rest_id = thread_id % WARP_SIZE;

    // 每个warp的参数
    const unsigned int warp_seq_begin = blockIdx.x * (expand_seq_inter_warp * expand_seq_inner_warp * seq_rest) + threadIdx.z * (expand_seq_inner_warp * seq_rest);
    const unsigned int warp_seq_end = warp_seq_begin + (expand_seq_inner_warp * seq_rest);
    const unsigned int seq_begin_bias_to_warp = threadIdx.y;

    // every thread need:
    // s[bsz_id, n_local_kv_heads_id, :, seq_begin:seq_end:seq_stride]
    // v_cache[bsz_id, n_local_kv_heads_id, seq_begin:seq_end:seq_stride, d_head_id:d_head_id + 8]
    // o[bsz_id, n_local_kv_heads_id, expand_seq_acc_id, :, d_head_id:d_head_id + 8]
    // diff_array[0, bsz_id, n_local_kv_heads_id, seq_begin:seq_end]
    // every warp need:
    // s[bsz_id, n_local_kv_heads_id, :, warp_seq_begin:warp_seq_end]
    // v_cache[bsz_id, n_local_kv_heads_id, warp_seq_begin:warp_seq_end, :]
    // o[bsz_id, n_local_kv_heads_id, expand_seq_acc_id, :, :]

    // 共享内存
    static_assert(TENSORCORE_M >= gqa, "TENSORCORE_M must be greater than or equal to gqa");
    constexpr unsigned int tensorcore_switch = use_tensorcore ? 1 : 0;
    constexpr unsigned int use_buffer = data_transfer_mode > 0 ? 1 : tensorcore_switch;
    // __align__(64) __shared__ half cache_score[use_buffer][warp_num][TENSORCORE_M][TENSORCORE_K * tensorcore_mid_dim_group]; // 2*8*32*2*2 = 2048
    // __align__(64) __shared__ half cache_value[use_buffer][warp_num][TENSORCORE_K * tensorcore_mid_dim_group][d_head]; // 2*2*32*128*2 = 32MB
    // __align__(64) __shared__ float cache_output[tensorcore_switch][warp_num][TENSORCORE_M][d_head]; // 
    // __align__(64) __shared__ half2 cache_output_half2[1 - tensorcore_switch][expand_seq_inter_warp][expand_seq_inner_warp][gqa][d_head / 2];
    constexpr unsigned int warp_num_use_buffer = use_buffer ? warp_num : 1;
    constexpr unsigned int TENSORCORE_M_use_buffer = use_buffer ? TENSORCORE_M : 1;
    constexpr unsigned int TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer = use_buffer ? TENSORCORE_K * tensorcore_mid_dim_group : 1;
    constexpr unsigned int d_head_use_buffer = use_buffer ? d_head : 1;
    constexpr unsigned int warp_num_tensorcore_switch = tensorcore_switch ? warp_num : 1;
    constexpr unsigned int TENSORCORE_M_tensorcore_switch = tensorcore_switch ? TENSORCORE_M : 1;
    constexpr unsigned int d_head_tensorcore_switch = tensorcore_switch ? d_head : 1;
    constexpr unsigned int expand_seq_inter_warp_not_tensorcore_switch = (1 - tensorcore_switch) ? expand_seq_inter_warp : 1;
    constexpr unsigned int expand_seq_inner_warp_not_tensorcore_switch = (1 - tensorcore_switch) ? expand_seq_inner_warp : 1;
    constexpr unsigned int gqa_not_tensorcore_switch = (1 - tensorcore_switch) ? gqa : 1;
    constexpr unsigned int d_head_div_2_not_tensorcore_switch = (1 - tensorcore_switch) ? d_head / 2 : 1;
    __align__(64) __shared__ half cache_score[warp_num_use_buffer][TENSORCORE_M_use_buffer][TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer];
    __align__(64) __shared__ half cache_value[warp_num_use_buffer][TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer][d_head_use_buffer];
    __align__(64) __shared__ float cache_output[warp_num_tensorcore_switch][TENSORCORE_M_tensorcore_switch][d_head_tensorcore_switch];
    __align__(64) __shared__ half2 cache_output_half2[expand_seq_inter_warp_not_tensorcore_switch][expand_seq_inner_warp_not_tensorcore_switch][gqa_not_tensorcore_switch][d_head_div_2_not_tensorcore_switch];
    __shared__ unsigned int shared_mem[2];
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 1) {
        shared_mem[0] = 0x00000000;
        shared_mem[1] = 0x88888888;
    }
    __syncthreads();

    // init cache_score
    if constexpr (use_buffer) {
        static_assert((TENSORCORE_M * TENSORCORE_K * tensorcore_mid_dim_group) % WARP_SIZE == 0, "TENSORCORE_M * TENSORCORE_N * tensorcore_mid_dim_group must be a multiple of WARP_SIZE");
        # pragma unroll
        for (unsigned int i = 0; i < TENSORCORE_M * TENSORCORE_K * tensorcore_mid_dim_group; i += WARP_SIZE) {
            cache_score[warp_id][0][warp_rest_id + i] = __ushort_as_half(0x0000);
        }
    }

    // init cache_output
    if constexpr (use_tensorcore) {
        static_assert((TENSORCORE_M * d_head) % WARP_SIZE == 0, "TENSORCORE_M * d_head must be a multiple of WARP_SIZE");
        # pragma unroll
        for (unsigned int i = 0; i < TENSORCORE_M * d_head; i += WARP_SIZE) {
            cache_output[warp_id][0][warp_rest_id + i] = 0.0f;
        }
    }

    // 初始化结果寄存器
    half2 temp_result[gqa][4] = {};

    // main loop body
    constexpr unsigned int loop_stride = use_buffer == 1 ? (TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer / expand_seq_inner_warp) : 1;
    for (unsigned int i = 0; i < seq_rest; i += loop_stride) {
        half temp_s[gqa];
        short diff[loop_stride];
        uint4 temp_num;

        // load s
        if constexpr (data_transfer_mode == 0) {
            if constexpr (use_tensorcore) {
                // single warp
                // s[bsz_id, n_local_kv_heads_id, :, warp_seq_begin + i * seq_stride : warp_seq_begin + (i + loop_stride) * seq_stride]
                // loop_stride * seq_stride = TENSORCORE_N * tensorcore_mid_dim_group
                // static_assert((TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer * gqa) % WARP_SIZE == 0, "TENSORCORE_K * tensorcore_mid_dim_group * gqa must be a multiple of WARP_SIZE");
                # pragma unroll
                for (unsigned int j = 0; j < TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer * gqa; j += WARP_SIZE) {
                    const unsigned int seq_id = warp_seq_begin + i * seq_stride + j % TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer + warp_rest_id;
                    const unsigned int gqa_id = j / TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer;
                    cache_score[warp_id][0][j + warp_rest_id] = s.get_half(bsz_id, n_local_kv_heads_id, gqa_id, seq_id);
                }
            } else {
                // single thread
                // s[bsz_id, n_local_kv_heads_id, :, seq_begin + i * seq_stride]
                # pragma unroll
                for (unsigned int j = 0; j < gqa; j++) {
                    temp_s[j] = s.get_half(bsz_id, n_local_kv_heads_id, j, seq_begin + i * seq_stride);
                }
            }
        } else {
            ;// TODO
        }

        // diff
        if constexpr (reference) {
            # pragma unroll
            for (unsigned int j = 0; j < loop_stride; j++) {
                diff[j] = 0;
            }
        } else {
            // diff_array[0, bsz_id, n_local_kv_heads_id, seq_begin + i * seq_stride:seq_begin + (i + loop_stride) * seq_stride:seq_stride]
            # pragma unroll
            for (unsigned int j = 0; j < loop_stride; j++) {
                diff[j] = diff_array.get_short(0, bsz_id, n_local_kv_heads_id, seq_begin + (i + j) * seq_stride);
            }
        }

        // load v_cache
        if constexpr (data_transfer_mode == 0) {
            if constexpr (use_tensorcore) {
                // single warp
                // v_cache[bsz_id, n_local_kv_heads_id, warp_seq_begin + i * seq_stride : warp_seq_begin + (i + loop_stride) * seq_stride, :]
                // loop_stride * seq_stride = TENSORCORE_K * tensorcore_mid_dim_group
                static_assert(TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer % seq_stride == 0, "TENSORCORE_K * tensorcore_mid_dim_group * d_head / 8 must be a multiple of WARP_SIZE");
                # pragma unroll
                for (unsigned int j = 0, k = 0; j < TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer; j += seq_stride, k += 1) {
                    const unsigned int seq_id = seq_begin + i * seq_stride + j;
                    unsigned int* to_mem = reinterpret_cast<unsigned int*>(&cache_value[warp_id][j + seq_begin_bias_to_warp][d_head_id]);
                    v_cache.load_8_uint16(bsz_id, n_local_kv_heads_id, seq_id, d_head_id, diff[k], shared_mem, to_mem);
                }
            } else {
                // single thread
                // v_cache[bsz_id, n_local_kv_heads_id, seq_begin + i * seq_stride, d_head_id:d_head_id + 8]
                v_cache.load_8_uint16(bsz_id, n_local_kv_heads_id, seq_begin + i * seq_stride, d_head_id, diff[0], shared_mem, temp_num);
            }
        } else {
            ;// TODO
        }

        // compute
        if constexpr (use_tensorcore) {
            // single warp
            // cache_score[warp_num][TENSORCORE_M][TENSORCORE_N * tensorcore_mid_dim_group];
            // cache_value[warp_num][TENSORCORE_N * tensorcore_mid_dim_group][d_head];
            // cache_output[warp_num][TENSORCORE_M][d_head];

            # pragma unroll
            for (unsigned int j = 0; j < d_head; j += TENSORCORE_N){
                // init fragment
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TENSORCORE_M, TENSORCORE_N, TENSORCORE_K, half, nvcuda::wmma::row_major> a_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TENSORCORE_M, TENSORCORE_N, TENSORCORE_K, half, nvcuda::wmma::row_major> b_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TENSORCORE_M, TENSORCORE_N, TENSORCORE_K, float> c_frag;

                // load c
                nvcuda::wmma::load_matrix_sync(c_frag, &cache_output[warp_id][0][j], d_head, nvcuda::wmma::mem_row_major);

                for (unsigned int k = 0; k < TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer; k += TENSORCORE_K) {
                    // load a
                    nvcuda::wmma::load_matrix_sync(a_frag, &cache_score[warp_id][0][k], TENSORCORE_K_mul_tensorcore_mid_dim_group_use_buffer);

                    // load b
                    nvcuda::wmma::load_matrix_sync(b_frag, &cache_value[warp_id][k][j], d_head);

                    // compute
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                // store c
                nvcuda::wmma::store_matrix_sync(&cache_output[warp_id][0][j], c_frag, d_head, nvcuda::wmma::mem_row_major);
            }
        } else {
            // single thread
            // o[bsz_id, n_local_kv_heads_id, expand_seq_acc_id, :, d_head_id:d_head_id + 8]
            # pragma unroll
            for (unsigned int j = 0; j < gqa; j++) {
                temp_result[j][0] = __hfma2(__half2half2(temp_s[j]), reinterpret_cast<half2&>(temp_num.x), temp_result[j][0]);
                temp_result[j][1] = __hfma2(__half2half2(temp_s[j]), reinterpret_cast<half2&>(temp_num.y), temp_result[j][1]);
                temp_result[j][2] = __hfma2(__half2half2(temp_s[j]), reinterpret_cast<half2&>(temp_num.z), temp_result[j][2]);
                temp_result[j][3] = __hfma2(__half2half2(temp_s[j]), reinterpret_cast<half2&>(temp_num.w), temp_result[j][3]);
            }
        }
    }

    __syncthreads();
    
    // reduce and store
    if constexpr (data_transfer_mode == 0) {
        if constexpr (use_tensorcore) {
            // multithread
            // cache_output[warp_num][TENSORCORE_M][d_head]
            // o[bsz_id, n_local_kv_heads_id, expand_seq_acc_id, :, :]
            for (unsigned int i = thread_id; i < gqa * d_head; i += thread_num) {
                half acc = __ushort_as_half(0x0000);
                # pragma unroll
                for (unsigned int j = 0; j < warp_num; j++) {
                    acc = __hadd(acc, __float2half(cache_output[j][0][i]));
                }
                o.save_half(bsz_id, n_local_kv_heads_id, expand_seq_acc_id, 0, i, acc);
            }
        } else {
            // single thread
            # pragma unroll
            for (unsigned int i = 0; i < gqa; i++) {
                cache_output_half2[threadIdx.z][threadIdx.y][i][d_head_id / 2] = temp_result[i][0];
                cache_output_half2[threadIdx.z][threadIdx.y][i][d_head_id / 2 + 1] = temp_result[i][1];
                cache_output_half2[threadIdx.z][threadIdx.y][i][d_head_id / 2 + 2] = temp_result[i][2];
                cache_output_half2[threadIdx.z][threadIdx.y][i][d_head_id / 2 + 3] = temp_result[i][3];
            }

            __syncthreads();

            // cache_output_half2[expand_seq_inter_warp][expand_seq_inner_warp][gqa][d_head / 2]
            // o[bsz_id, n_local_kv_heads_id, expand_seq_acc_id, :, :]
            for (unsigned int i = thread_id; i < gqa * d_head / 2; i += thread_num) {
                half2 acc = __half2half2(__ushort_as_half(0x0000));
                # pragma unroll
                for (unsigned int j = 0; j < expand_seq_inter_warp; j++) {
                    # pragma unroll
                    for (unsigned int k = 0; k < expand_seq_inner_warp; k++) {
                        acc = __hadd2(acc, cache_output_half2[j][k][0][i]);
                    }
                }
                o.save_uint32(bsz_id, n_local_kv_heads_id, expand_seq_acc_id, 0, i * 2, reinterpret_cast<unsigned int&>(acc));
            }
        }
    } else {
        ;// TODO
    }

    __syncthreads();
}

torch::Tensor v_cache_compute(
    torch::Tensor& s,                  // (bsz, n_local_kv_heads, n_gqa_group, seqlen)
    torch::Tensor& v_cache_first_8,    // (bsz, n_local_kv_heads, max_seqlen, d_head // 4)
    torch::Tensor& v_cache_mid_4,      // (bsz, n_local_kv_heads, max_seqlen, d_head // 8)
    torch::Tensor& v_cache_last_4,     // (bsz, n_local_kv_heads, max_seqlen, d_head // 8)
    torch::Tensor& diff_array,         // (bsz, n_local_kv_heads, seqlen)
    const unsigned int seqlen,
    const bool reference,
    const bool use_tensorcore
) {
    // 推导维度信息
    const unsigned int bsz = s.size(0);
    const unsigned int n_local_kv_heads = s.size(1);
    const unsigned int n_gqa_group = s.size(2);
    const unsigned int d_head = v_cache_first_8.size(3) * 4;

    // 配置设置
    const unsigned int expand_seq_acc = EXPAND_SEQ_ACC;
    // block(expand_seq_acc, n_local_kv_heads, bsz)
    // thead(d_head // 8, expand_seq_inner_warp, expand_seq_inter_warp)
    // if ((bsz * n_local_kv_heads) % 80 == 0 || bsz * n_local_kv_heads > 160){
    //     expand_seq_acc = 1;
    // } else if ((bsz * n_local_kv_heads) % 40 == 0 || bsz * n_local_kv_heads > 80) {
    //     expand_seq_acc = 2;
    // } else if ((bsz * n_local_kv_heads) % 20 == 0 || bsz * n_local_kv_heads > 40) {
    //     expand_seq_acc = 4;
    // } else {
    //     expand_seq_acc = 8;
    // }
    const unsigned int expand_seq_inter_warp = EXPAND_SEQ_INTER_WARP;
    const unsigned int expand_seq_inner_warp = EXPAND_SEQ_INNER_WARP;
    const unsigned int tensorcore_mid_dim_group = TENSORCORE_MID_DIM_GROUP;

    // 判断
    TORCH_CHECK(d_head == 128, "d_head must be 128, otherwise the code not implemented yet");
    TORCH_CHECK(seqlen % 64 == 0, "seqlen must be a multiple of 8");
    TORCH_CHECK((d_head / 8) * expand_seq_inner_warp == WARP_SIZE, "d_head / 8 * expand_seq_inner_warp must be equal to WARP_SIZE");
    TORCH_CHECK(seqlen % (expand_seq_acc * expand_seq_inter_warp * expand_seq_inner_warp) == 0, "seqlen must be a multiple of expand_seq_acc * expand_seq_inter_warp * expand_seq_inner_warp");
    const unsigned int seq_rest = seqlen / (expand_seq_acc * expand_seq_inter_warp * expand_seq_inner_warp);
    TORCH_CHECK(!use_tensorcore || seq_rest % (TENSORCORE_N * tensorcore_mid_dim_group / expand_seq_inner_warp) == 0, "if use_tensorcore, seqlen must be a multiple of TENSORCORE_N * tensorcore_mid_dim_group * expand_acc * expand_seq_inner_warp");
    // same as seqlen % (TENSORCORE_N * tensorcore_mid_dim_group * expand_seq_acc * expand_seq_inter_warp)
    TORCH_CHECK(n_gqa_group == 1 || n_gqa_group == 2 || n_gqa_group == 3 || n_gqa_group == 4 || n_gqa_group == 8, "n_gqa_group must be one of the following values: 1, 2, 3, 4, or 8");
    
    // Check if the tensors are on the GPU
    TORCH_CHECK(s.device().is_cuda(), "s must be a CUDA tensor");
    TORCH_CHECK(v_cache_first_8.device().is_cuda(), "v_cache_first_8 must be a CUDA tensor");
    TORCH_CHECK(v_cache_mid_4.device().is_cuda(), "v_cache_mid_4 must be a CUDA tensor");
    TORCH_CHECK(v_cache_last_4.device().is_cuda(), "v_cache_last_4 must be a CUDA tensor");
    TORCH_CHECK(diff_array.device().is_cuda(), "diff_array must be a CUDA tensor");

    // Set the device context for the current device
    cudaSetDevice(s.get_device());

    // Ensure the tensors have the correct sizes and types
    TORCH_CHECK(s.dtype() == torch::kHalf, "s must be of type torch.half");
    TORCH_CHECK(v_cache_first_8.dtype() == torch::kUInt32, "v_cache_first_8 must be of type torch.uint32");
    TORCH_CHECK(v_cache_mid_4.dtype() == torch::kUInt32, "v_cache_mid_4 must be of type torch.uint32");
    TORCH_CHECK(v_cache_last_4.dtype() == torch::kUInt32, "v_cache_last_4 must be of type torch.uint32");
    TORCH_CHECK(diff_array.dtype() == torch::kShort, "diff_array must be of type torch.short");

    // Ensure the tensors have the correct sizes and types
    TORCH_CHECK(
        v_cache_first_8.size(0) == bsz &&
        v_cache_first_8.size(1) == n_local_kv_heads &&
        v_cache_first_8.size(2) >= seqlen &&
        v_cache_first_8.size(3) == d_head / 4, 
        "v_cache_first_8 must have the correct size: expected (", 
        bsz, ", ", n_local_kv_heads, ", >= ", seqlen, ", ", d_head / 4, 
        "), but got (", 
        v_cache_first_8.size(0), ", ", 
        v_cache_first_8.size(1), ", ", 
        v_cache_first_8.size(2), ", ", 
        v_cache_first_8.size(3), ")."
    );
    TORCH_CHECK(
        v_cache_mid_4.size(0) == bsz &&
        v_cache_mid_4.size(1) == n_local_kv_heads &&
        v_cache_mid_4.size(2) >= seqlen &&
        v_cache_mid_4.size(3) == d_head / 8, 
        "v_cache_mid_4 must have the correct size: expected (", 
        bsz, ", ", n_local_kv_heads, ", >= ", seqlen, ", ", d_head / 8, 
        "), but got (", 
        v_cache_mid_4.size(0), ", ", 
        v_cache_mid_4.size(1), ", ", 
        v_cache_mid_4.size(2), ", ", 
        v_cache_mid_4.size(3), ")."
    );
    TORCH_CHECK(
        v_cache_last_4.size(0) == bsz &&
        v_cache_last_4.size(1) == n_local_kv_heads &&
        v_cache_last_4.size(2) >= seqlen &&
        v_cache_last_4.size(3) == d_head / 8, 
        "v_cache_last_4 must have the correct size: expected (", 
        bsz, ", ", n_local_kv_heads, ", >= ", seqlen, ", ", d_head / 8, 
        "), but got (", 
        v_cache_last_4.size(0), ", ", 
        v_cache_last_4.size(1), ", ", 
        v_cache_last_4.size(2), ", ", 
        v_cache_last_4.size(3), ")."
    );
    
    // Create output tensor
    torch::Tensor o = torch::ones({bsz, n_local_kv_heads, expand_seq_acc, n_gqa_group, d_head}, torch::dtype(torch::kHalf).device(v_cache_first_8.device()));

    // Reshape diff_array to (1, bsz, n_local_kv_heads, seqlen_more)
    diff_array = diff_array.view({1, bsz, n_local_kv_heads, -1});

    // Create TensorNormal and VCache objects
    TensorNormal s_tensor(s);
    VCache v_cache(v_cache_first_8, v_cache_mid_4, v_cache_last_4);
    TensorNormal o_tensor(o);
    TensorNormal diff_array_tensor(diff_array);

    // Create grid and block parameters
    dim3 grid(expand_seq_acc, n_local_kv_heads, bsz);
    dim3 block(d_head / 8, expand_seq_inner_warp, expand_seq_inter_warp);

    // Launch the kernel
    /*template <bool reference, bool use_tensorcore, unsigned int data_transfer_mode, 
          unsigned int gqa, unsigned int expand_seq_acc, unsigned int expand_seq_inter_warp, unsigned int expand_seq_inner_warp,
          unsigned int d_head, unsigned int tensorcore_mid_dim_group>*/
    // v_cache_compute_kernel<true, true, 0, 4, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
    if (reference) {
        if (use_tensorcore) {
            // v_cache_compute_kernel<true, true, 0, n_gqa_group, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
            switch (n_gqa_group) {
                case 1:
                    v_cache_compute_kernel<true, true, 0, 1, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 2:
                    v_cache_compute_kernel<true, true, 0, 2, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 3:
                    v_cache_compute_kernel<true, true, 0, 3, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 4:
                    v_cache_compute_kernel<true, true, 0, 4, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 8:
                    v_cache_compute_kernel<true, true, 0, 8, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
            }
        } else {
            // v_cache_compute_kernel<true, false, 0, n_gqa_group, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
            switch (n_gqa_group) {
                case 1:
                    v_cache_compute_kernel<true, false, 0, 1, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 2:
                    v_cache_compute_kernel<true, false, 0, 2, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 3:
                    v_cache_compute_kernel<true, false, 0, 3, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 4:
                    v_cache_compute_kernel<true, false, 0, 4, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 8:
                    v_cache_compute_kernel<true, false, 0, 8, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
            }
        }
    } else {
        if (use_tensorcore) {
            // v_cache_compute_kernel<false, true, 0, n_gqa_group, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
            switch (n_gqa_group) {
                case 1:
                    v_cache_compute_kernel<false, true, 0, 1, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 2:
                    v_cache_compute_kernel<false, true, 0, 2, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 3:
                    v_cache_compute_kernel<false, true, 0, 3, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 4:
                    v_cache_compute_kernel<false, true, 0, 4, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 8:
                    v_cache_compute_kernel<false, true, 0, 8, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
            }
        } else {
            // v_cache_compute_kernel<false, false, 0, n_gqa_group, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
            switch (n_gqa_group) {
                case 1:
                    v_cache_compute_kernel<false, false, 0, 1, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 2:
                    v_cache_compute_kernel<false, false, 0, 2, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 3:
                    v_cache_compute_kernel<false, false, 0, 3, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 4:
                    v_cache_compute_kernel<false, false, 0, 4, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
                case 8:
                    v_cache_compute_kernel<false, false, 0, 8, expand_seq_acc, expand_seq_inter_warp, expand_seq_inner_warp, 128, tensorcore_mid_dim_group><<<grid, block>>>(s_tensor, v_cache, o_tensor, diff_array_tensor, seqlen);
                    break;
            }
        }
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in v_cache_compute: " << cudaGetErrorString(error) << std::endl;
    }

    return o;
}