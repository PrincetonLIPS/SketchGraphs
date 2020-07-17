#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <cub/cub.cuh>

#include <iterator>


/** This iterator provides a mean of iterating over two different arrays
 * as a proxy for iterating over an array of pairs.
 */
template <typename K, typename V>
struct KvpTwoArrayIterator
    : std::iterator<std::random_access_iterator_tag, cub::KeyValuePair<K, V>> {
    // This iterator lies about its value type, as it returns a proxy.
    typedef cub::KeyValuePair<K, V> value_type;

    K *ptr_key;
    V *ptr_value;

    KvpTwoArrayIterator(K *ptr_key, V *ptr_value) : ptr_key(ptr_key), ptr_value(ptr_value) {}

    struct KvpProxy {
        K *ptr_key;
        V *ptr_value;

        __host__ __device__ __forceinline__ operator cub::KeyValuePair<K, V>() const {
            return cub::KeyValuePair<K, V>(*ptr_key, *ptr_value);
        }

        __host__ __device__ __forceinline__ KvpProxy &
        operator=(cub::KeyValuePair<K, V> const &other) {
            *ptr_key = other.key;
            *ptr_value = other.value;
            return *this;
        }
    };

    template <typename Idx>
    __host__ __device__ __forceinline__ KvpProxy operator[](Idx idx) const {
        return KvpProxy{ptr_key + idx, ptr_value + idx};
    }
};

template <typename T, typename SegmentIterator>
void segment_argmax_gpu_impl(T *values, SegmentIterator const& offsets_start, SegmentIterator const& offsets_end, T *out_values,
                             std::int32_t *out_idx, int num_items) {
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    auto stream = at::cuda::getCurrentCUDAStream();

    KvpTwoArrayIterator<std::int32_t, T> out_iterator{out_idx, out_values};

    THCudaCheck(cub::DeviceSegmentedReduce::ArgMax(
        temp_storage, temp_storage_bytes, values, out_iterator,
        num_items, offsets_start, offsets_end, stream.stream()));

    THCudaCheck(cudaMalloc(&temp_storage, temp_storage_bytes));

    THCudaCheck(cub::DeviceSegmentedReduce::ArgMax(
        temp_storage, temp_storage_bytes, values, out_iterator,
        num_items, offsets_start, offsets_end, stream.stream()));

    THCudaCheck(cudaFree(temp_storage));
}

