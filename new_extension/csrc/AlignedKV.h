#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h> // For WMMA
#include <stdio.h>
// #include <cooperative_groups.h> 
// #include <cooperative_groups∕memcpy_async.h>

// namespace cg = cooperative_groups;

#define BITS_PER_CHANNEL 256
#define COLUMN_BLOCK 64 // at least BITS_PER_CHANNEL / 4
#define WARP_SIZE 32
#define TENSORCORE_M 8
#define TENSORCORE_N 32
#define TENSORCORE_K 16

#define EXPAND_SEQ_ACC 2
#define TENSORCORE_MID_DIM_GROUP 2
#define EXPAND_SEQ_INNER_WARP 2
#define EXPAND_SEQ_INTER_WARP 2


// TODO
#define DATA_TRANSFER 0 // 0: use shared memory, 1: memcpy_async

/*
gemv v2
争取做到能把各种数据pipeline起来
*/
struct TensorNormal {
    // 数据指针
    unsigned short* __restrict__ data;
    // 一般都是4个维度或5个维度
    unsigned int stride_0;
    unsigned int stride_1;
    unsigned int stride_2;
    unsigned int stride_3;

    // 构造函数
    TensorNormal(const torch::Tensor& tensor) {
        // 检查维度是4维或5维
        TORCH_CHECK(tensor.dim() == 4 || tensor.dim() == 5, "tensor must be of dimension 4 or 5");

        // 将数据指针初始化为Tensor的data指针
        data = reinterpret_cast<unsigned short*>(tensor.data_ptr());

        // 初始化stride（假设tensor的维度为4）
        stride_0 = tensor.stride(0);
        stride_1 = tensor.stride(1);
        stride_2 = tensor.stride(2);
        if (tensor.dim() > 4) {
            stride_3 = tensor.stride(3);
        } else {
            stride_3 = 0;
        }
    }
    // 访问函数
    __device__ __forceinline__ unsigned int get_bias(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3) const {
        return dim_0 * stride_0 + dim_1 * stride_1 + dim_2 * stride_2 + dim_3;
    }
    __device__ __forceinline__ unsigned int get_bias(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3, const unsigned int dim_4) const {
        return dim_0 * stride_0 + dim_1 * stride_1 + dim_2 * stride_2 + dim_3 * stride_3 + dim_4;
    }
    // 访问数 half
    __device__ __forceinline__ half get_half(const int dim_0, const int dim_1, const int dim_2, const int dim_3) const {
        return reinterpret_cast<half*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3)];
    }
    __device__ __forceinline__ half get_half(const int dim_0, const int dim_1, const int dim_2, const int dim_3, const int dim_4) const {
        return reinterpret_cast<half*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3, dim_4)];
    }
    // 访问数 uint32
    __device__ __forceinline__ unsigned int get_uint32(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3) const {
        return reinterpret_cast<unsigned int*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3) / 2];
    }
    __device__ __forceinline__ unsigned int get_uint32(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3, const unsigned int dim_4) const {
        return reinterpret_cast<unsigned int*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3, dim_4) / 2];
    }
    // 保存数 half
    __device__ __forceinline__ void save_half(const int dim_0, const int dim_1, const int dim_2, const int dim_3, const half value) {
        reinterpret_cast<half*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3)] = value;
    }
    __device__ __forceinline__ void save_half(const int dim_0, const int dim_1, const int dim_2, const int dim_3, const int dim_4, const half value) {
        reinterpret_cast<half*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3, dim_4)] = value;
    }
    // 保存数 uint32
    __device__ __forceinline__ void save_uint32(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3, const unsigned int value) {
        reinterpret_cast<unsigned int*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3) / 2] = value;
    }
    __device__ __forceinline__ void save_uint32(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3, const unsigned int dim_4, const unsigned int value) {
        reinterpret_cast<unsigned int*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3, dim_4) / 2] = value;
    }
    // 访问数 short
    __device__ __forceinline__ short get_short(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3) const {
        return reinterpret_cast<short*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3)];
    }
    __device__ __forceinline__ short get_short(const unsigned int dim_0, const unsigned int dim_1, const unsigned int dim_2, const unsigned int dim_3, const unsigned int dim_4) const {
        return reinterpret_cast<short*>(data)[get_bias(dim_0, dim_1, dim_2, dim_3, dim_4)];
    }
};