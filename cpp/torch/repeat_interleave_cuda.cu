#include "repeat_interleave.h"
#include <ATen/cuda/CUDAContext.h>

__global__ static void compute_cuda_kernel(const int64_t * __restrict__ repeat_ptr, const int64_t * __restrict__ cumsum_ptr,
                                           int64_t * __restrict__ result_ptr, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < size; i += stride) {
        int64_t end = cumsum_ptr[i];
        int64_t repeat = repeat_ptr[i];
        int64_t start = end - repeat;
        for(int64_t j = start; j < end; j++) {
            result_ptr[j] = i;
        }
    }
}

__global__ static void compute_cuda_kernel_scope(const int64_t * __restrict__ scope_ptr, int64_t * __restrict__ result_ptr, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < size; i += stride) {
        int64_t start = scope_ptr[2 * i];
        int64_t repeat = scope_ptr[2 * i + 1];
        int64_t end = start + repeat;
        for(int64_t j = start; j < end; j++) {
            result_ptr[j] = i;
        }
    }
}

static void compute_cuda(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t block = 512;
    int64_t grid = std::min<int64_t>((size + block - 1) / block, 2048L);
    compute_cuda_kernel<<<grid, block, 0, stream>>>(repeat_ptr, cumsum_ptr, result_ptr, size);
}

static void compute_cuda_scope(int64_t *scope_ptr, int64_t *result_ptr, int64_t size) {
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t block = 512;
    int64_t grid = std::min<int64_t>((size + block - 1) / block, 2048L);
    compute_cuda_kernel_scope<<<grid, block, 0, stream>>>(scope_ptr, result_ptr, size);
}

at::Tensor &genric::repeat_interleave_gpu_out(const at::Tensor& repeats, at::Tensor &out) {
    TORCH_CHECK(out.is_contiguous(), "Output array must be contiguous.");

    auto repeats_ = repeats.contiguous();
    auto cumsum = repeats.cumsum(0);
    compute_cuda(repeats_.data<int64_t>(), cumsum.data<int64_t>(), out.data<int64_t>(),
                 repeats.size(0));
    return out;
}

at::Tensor &genric::repeat_interleave_gpu_out_scope(const at::Tensor& scope, at::Tensor &out) {
    TORCH_CHECK(out.is_contiguous(), "Output array must be contiguous");

    auto scope_ = scope.contiguous();
    compute_cuda_scope(scope_.data<int64_t>(), out.data<int64_t>(), scope_.size(0));
    return out;
}
