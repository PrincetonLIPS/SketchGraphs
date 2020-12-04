#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>

#include <utility>

#include <cub/cub.cuh>
#include "cub_utils.cuh"


template <typename ScalarT, template<typename> typename PtrTraits, typename IndexT>
struct RepeatScopesToOffsetIterator : std::iterator<std::random_access_iterator_tag, ScalarT> {

    typedef at::PackedTensorAccessor<ScalarT, 2, PtrTraits, IndexT> AccessorT;
    AccessorT scopes;
    bool is_end;
    IndexT stride_scope;
    IndexT stride_value;

    RepeatScopesToOffsetIterator(AccessorT const &scopes, bool is_end, IndexT stride_scope, IndexT stride_value)
        : scopes(scopes), is_end(is_end), stride_scope(stride_scope), stride_value(stride_value) {}

    __host__ __device__ __forceinline__ ScalarT operator[](IndexT idx) const {
        auto offset = idx / stride_scope;
        idx = idx % stride_scope;

        auto result = scopes[idx][0];

        if (is_end) {
            result += scopes[idx][1];
        }

        return result + stride_value * offset;
    }
};


std::tuple<at::Tensor, at::Tensor> segment_max_pool1d_cuda(at::Tensor const& values, at::Tensor const& scopes) {
    auto value_transpose = values.t().contiguous();

    auto result_values = at::empty({values.size(1), scopes.size(0)}, values.options());
    auto result_locations = at::empty({values.size(1), scopes.size(0)}, scopes.options().dtype(c10::ScalarType::Int));

    auto scopes_accessor =
        scopes.packed_accessor<std::int64_t, 2, at::DefaultPtrTraits, std::uint32_t>();

    auto num_segments = c10::checked_convert<std::uint32_t>(scopes.size(0), "uint32_t");
    auto num_rows = c10::checked_convert<std::uint32_t>(values.size(0), "uint32_t");

    typedef RepeatScopesToOffsetIterator<std::int64_t, at::DefaultPtrTraits, std::uint32_t>
        AccessorT;
    AccessorT offsets_start{scopes_accessor, false, num_segments, num_rows};
    AccessorT offsets_end{scopes_accessor, true, num_segments, num_rows};

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "segment_max_pool1d_cuda", [&]() {
        segment_argmax_gpu_impl(
            value_transpose.data_ptr<scalar_t>(), offsets_start, offsets_end,
            result_values.data_ptr<scalar_t>(), result_locations.data_ptr<std::int32_t>(),
            scopes.size(0) * values.size(1));
    });

    return std::make_tuple(result_values.t(), result_locations.t().toType(scopes.scalar_type()));
}


at::Tensor segment_avg_pool1d_cuda(at::Tensor const &values,
    at::Tensor const &scopes) {

    auto value_transpose = values.t().contiguous();

    auto result_values_transpose = at::empty({values.size(1), scopes.size(0)}, values.options());

    auto scopes_accessor =
        scopes.packed_accessor<std::int64_t, 2, at::DefaultPtrTraits, std::uint32_t>();

    auto num_segments = c10::checked_convert<std::uint32_t>(scopes.size(0), "uint32_t");
    auto num_rows = c10::checked_convert<std::uint32_t>(values.size(0), "uint32_t");
    auto num_segments_total = c10::checked_convert<int>(scopes.size(0) * values.size(1), "int");

    typedef RepeatScopesToOffsetIterator<std::int64_t, at::DefaultPtrTraits, std::uint32_t>
        AccessorT;
    AccessorT offsets_start{scopes_accessor, false, num_segments, num_rows};
    AccessorT offsets_end{scopes_accessor, true, num_segments, num_rows};

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "segment_max_pool1d_cuda", [&]() {
        auto value_ptr = value_transpose.data_ptr<scalar_t>();
        auto out_ptr = result_values_transpose.data_ptr<scalar_t>();

        void *temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        auto stream = at::cuda::getCurrentCUDAStream();

        THCudaCheck(cub::DeviceSegmentedReduce::Sum(
            temp_storage, temp_storage_bytes, value_ptr, out_ptr,
            num_segments_total, offsets_start, offsets_end, stream));

        THCudaCheck(cudaMalloc(&temp_storage, temp_storage_bytes));

        THCudaCheck(cub::DeviceSegmentedReduce::Sum(
            temp_storage, temp_storage_bytes, value_ptr, out_ptr,
            num_segments_total, offsets_start, offsets_end, stream));
        THCudaCheck(cudaFree(temp_storage));
    });

    result_values_transpose.div_(scopes.select(1, 1).unsqueeze(0).toType(values.scalar_type()));
    return result_values_transpose.t();
}
