#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>


template<typename T>
__global__ void segment_logsumexp_backward_kernel(T * __restrict__ result, const T * __restrict__ grad_output, const T * __restrict__ input,
                                                  const T * __restrict__ logsumexp, const int * __restrict__ offsets,
                                                  const int * __restrict__ source_idx) {
    int idx = offsets[blockIdx.x] + threadIdx.x;

    if (idx >= offsets[blockIdx.x + 1]) {
        // don't run over into next block.
        return;
    }

    int idx_source = source_idx[blockIdx.x];
    result[idx] = exp(input[idx] - logsumexp[idx_source]) * grad_output[idx_source];
}


at::Tensor segment_logsumexp_backward_gpu(at::Tensor const& grad_output, at::Tensor const& input, at::Tensor const& logsumexp, at::Tensor const& lengths) {
    int threads_per_block = 256;

    auto output = at::empty_like(input);
    auto lengths_int = lengths.toType(c10::ScalarType::Int);

    // Pre-compute indexing structures
    auto blocks_per_segment = lengths_int.add(threads_per_block - 1).floor_divide_(threads_per_block);
    auto source_idx_long = at::repeat_interleave(blocks_per_segment.toType(c10::ScalarType::Long));
    auto source_idx = source_idx_long.toType(c10::ScalarType::Int);
    auto block_lengths = at::full_like(source_idx, threads_per_block);


    {
        // adjust last block of each segment to have the right length.
        auto adjust = blocks_per_segment.mul(threads_per_block).sub_(lengths_int);
        auto block_is_last =
            at::ones_like(source_idx, source_idx.options().dtype(c10::ScalarType::Byte));
        auto block_is_last_narrow = block_is_last.narrow(0, 0, source_idx.size(0) - 1);
        at::ne_out(block_is_last_narrow,
                   source_idx.narrow(0, 1, source_idx.size(0) - 1),
                   source_idx.narrow(0, 0, source_idx.size(0) - 1));
        block_lengths.sub_(adjust.index_select(0, source_idx_long)
                               .mul_(block_is_last.toType(adjust.scalar_type())));
    }


    int num_blocks = c10::checked_convert<int>(block_lengths.size(0), "int");

    auto block_offsets = at::zeros({num_blocks + 1}, block_lengths.options());

    {
        auto block_offsets_narrow = block_offsets.narrow(0, 1, num_blocks);
        at::cumsum_out(block_offsets_narrow, block_lengths, 0);
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "segment_logsumexp_backward_gpu", [&]() {
        segment_logsumexp_backward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), input.contiguous().data_ptr<scalar_t>(),
            logsumexp.contiguous().data_ptr<scalar_t>(), block_offsets.data_ptr<int>(),
            source_idx.data_ptr<int>());
    });

    return output;
}
