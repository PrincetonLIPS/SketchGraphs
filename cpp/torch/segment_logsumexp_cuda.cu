#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>

#include <utility>
#include "cub_utils.cuh"


template <typename ScalarT, template<typename> typename PtrTraits, typename IndexT>
struct ScopesToOffsetIterator : std::iterator<std::random_access_iterator_tag, ScalarT> {
    at::PackedTensorAccessor<ScalarT, 2, PtrTraits, IndexT> scopes;
    bool is_end;

    ScopesToOffsetIterator(at::PackedTensorAccessor<ScalarT, 2, PtrTraits, IndexT> const &scopes, bool is_end)
        : scopes(scopes), is_end(is_end) {}

    __host__ __device__ __forceinline__ ScalarT operator[](IndexT idx) const {
        auto result = scopes[idx][0];

        if (is_end) {
            result += scopes[idx][1];
        }

        return result;
    }
};

template <typename ScalarT, template <typename> typename PtrTraits, typename IndexT>
ScopesToOffsetIterator<ScalarT, PtrTraits, IndexT> make_scopes_to_offset_iterator(
    at::PackedTensorAccessor<ScalarT, 2, PtrTraits, IndexT> const &scopes, bool is_end) {
    return ScopesToOffsetIterator<ScalarT, PtrTraits, IndexT>(scopes, is_end);
}


std::tuple<at::Tensor, at::Tensor> segment_argmax_gpu(at::Tensor const &values,
                                                      at::Tensor const &scopes) {
    auto result_values = at::empty({scopes.size(0)}, values.options());
    auto result_locations =
        at::empty({scopes.size(0)}, scopes.options().dtype(c10::ScalarType::Int));

    auto scopes_accessor =
        scopes.packed_accessor<std::int64_t, 2, at::DefaultPtrTraits, std::uint32_t>();
    auto offsets_start = make_scopes_to_offset_iterator(scopes_accessor, false);
    auto offsets_end = make_scopes_to_offset_iterator(scopes_accessor, true);

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "segment_argmax_gpu", [&]() {
        segment_argmax_gpu_impl<scalar_t>(
            values.data_ptr<scalar_t>(), offsets_start, offsets_end,
            result_values.data_ptr<scalar_t>(), result_locations.data_ptr<std::int32_t>(),
            scopes.size(0));
    });

    return std::make_tuple(result_values, result_locations.toType(scopes.scalar_type()));
}

struct LogAddExp {
    template <typename T> __host__ __device__ __forceinline__ T operator()(T a, T b) {
        if (a < b) {
            T c(a);
            a = b;
            b = c;
        }

        return log1p(exp(b - a)) + a;
    }
};

template <typename T, typename Accessor>
void segment_logsumexp_gpu_impl(T *values, Accessor const &scopes, T *out_values) {
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto offsets_start = make_scopes_to_offset_iterator(scopes, false);
    auto offsets_end = make_scopes_to_offset_iterator(scopes, true);
    auto init_value = std::numeric_limits<T>::lowest();

    THCudaCheck(cub::DeviceSegmentedReduce::Reduce(
        temp_storage, temp_storage_bytes, values, out_values,
        scopes.size(0), offsets_start, offsets_end, LogAddExp{}, init_value, stream));

    THCudaCheck(cudaMalloc(&temp_storage, temp_storage_bytes));

    THCudaCheck(cub::DeviceSegmentedReduce::Reduce(
        temp_storage, temp_storage_bytes, values, out_values,
        scopes.size(0), offsets_start, offsets_end, LogAddExp{}, init_value, stream));
    THCudaCheck(cudaFree(temp_storage));
}

at::Tensor segment_logsumexp_gpu(at::Tensor const &values, at::Tensor const &scopes) {
    auto result = at::empty({scopes.size(0)}, values.options());

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "segment_logsumexp_gpu", [&]() {
        segment_logsumexp_gpu_impl<scalar_t>(
            values.data_ptr<scalar_t>(),
            scopes.packed_accessor<std::int64_t, 2, at::RestrictPtrTraits, std::uint32_t>(),
            result.data_ptr<scalar_t>());
    });

    return result;
}
