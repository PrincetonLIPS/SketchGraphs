#include <ATen/ATen.h>
#include <ATen/Parallel.h>

namespace {

template <typename T, typename Idx>
void reduce_mean(at::TensorAccessor<T, 2, at::DefaultPtrTraits, Idx> const &data,
                 at::TensorAccessor<T, 1, at::DefaultPtrTraits, Idx> output, Idx start,
                 Idx length) {

    for (Idx i = start; i < start + length; ++i) {
        for (Idx j = 0; j < output.size(0); ++j) {
            output[j] += data[i][j];
        }
    }

    for (Idx j = 0; j < output.size(0); ++j) {
        if(length != 0) {
            output[j] /= static_cast<T>(length);
        }
    }
}

template <typename T, typename Idx>
void reduce_max(at::TensorAccessor<T, 2, at::DefaultPtrTraits, Idx> const &data,
                at::TensorAccessor<T, 1, at::DefaultPtrTraits, Idx> output,
                at::TensorAccessor<Idx, 1, at::DefaultPtrTraits, Idx> output_idx, Idx start,
                Idx length) {

    for (Idx i = start; i < start + length; ++i) {
        for (Idx j = 0; j < output.size(0); ++j) {
            if (data[i][j] > output[j]) {
                output[j] = data[i][j];
                output_idx[j] = i - start;
            }
        }
    }
}

at::Tensor segment_avg_pool1d_cpu(at::Tensor const &values, at::Tensor const &scopes) {
    auto output = at::zeros({scopes.size(0), values.size(1)}, values.options());

    if(scopes.size(0) == 0) {
        return output;
    }

    auto num_element_per_segment =
        std::max<decltype(values.size(0))>(values.size(0) / scopes.size(0) * values.size(1), 1);
    auto grain_size = std::max<decltype(at::internal::GRAIN_SIZE)>(
        at::internal::GRAIN_SIZE / num_element_per_segment, 1);

    auto scopes_accessor = scopes.accessor<int64_t, 2>();

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "segment_avgpool1d_cpu", [&]() {
        auto values_accessor = values.accessor<scalar_t, 2>();
        auto output_accessor = output.accessor<scalar_t, 2>();

        at::parallel_for(0, scopes.size(0), grain_size, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                reduce_mean(values_accessor, output_accessor[i], scopes_accessor[i][0],
                            scopes_accessor[i][1]);
            }
        });
    });

    return output;
}

std::tuple<at::Tensor, at::Tensor> segment_max_pool1d_cpu(at::Tensor const &values,
                                                          at::Tensor const &scopes) {
    at::Tensor output;
    auto output_idx = at::empty({scopes.size(0), values.size(1)}, values.options().dtype(c10::ScalarType::Long));

    if(scopes.size(0) == 0) {
        return std::make_tuple(output, output_idx);
    }

    auto num_element_per_segment =
        std::max<decltype(values.size(0))>(values.size(0) / scopes.size(0) * values.size(1), 1);
    auto grain_size = std::max<decltype(at::internal::GRAIN_SIZE)>(
        at::internal::GRAIN_SIZE / num_element_per_segment, 1);

    auto scopes_accessor = scopes.accessor<int64_t, 2>();
    auto output_idx_accessor = output_idx.accessor<int64_t, 2>();

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "segment_maxpool1d_cpu", [&]() {
        output = at::full({scopes.size(0), values.size(1)}, std::numeric_limits<scalar_t>::lowest(), values.options());
        auto values_accessor = values.accessor<scalar_t, 2>();
        auto output_accessor = output.accessor<scalar_t, 2>();

        at::parallel_for(0, scopes.size(0), grain_size, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                reduce_max(values_accessor, output_accessor[i], output_idx_accessor[i],
                           scopes_accessor[i][0], scopes_accessor[i][1]);
            }
        });
    });

    return std::make_tuple(output, output_idx);
}

} // namespace

at::Tensor segment_avg_pool1d_cuda(at::Tensor const &values, at::Tensor const &scopes);

at::Tensor segment_avg_pool1d(at::Tensor const &values, at::Tensor const &scopes) {
    if (values.is_cuda()) {
        return segment_avg_pool1d_cuda(values, scopes);
    } else {
        return segment_avg_pool1d_cpu(values, scopes);
    }
}

std::tuple<at::Tensor, at::Tensor> segment_max_pool1d_cuda(at::Tensor const &values,
                                                           at::Tensor const &scopes);

std::tuple<at::Tensor, at::Tensor> segment_max_pool1d(at::Tensor const &values,
    at::Tensor const &scopes) {

    if (values.is_cuda()) {
        return segment_max_pool1d_cuda(values, scopes);
    } else {
        return segment_max_pool1d_cpu(values, scopes);
    }
}
