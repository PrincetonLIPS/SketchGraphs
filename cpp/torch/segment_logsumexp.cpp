#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

template <typename T, typename Idx>
std::pair<T, Idx> reduce_max(at::TensorAccessor<T, 1> const &data, Idx start, Idx length) {
    Idx max_idx = start;
    T maximum = std::numeric_limits<T>::lowest();

    for (int64_t i = start; i < start + length; ++i) {
        if (data[i] > maximum) {
            maximum = data[i];
            max_idx = i;
        }
    }

    return std::make_pair(maximum, max_idx - start);
}

template <typename T>
T logsumexp(at::TensorAccessor<T, 1> const &data, int64_t start, int64_t length) {
    T maximum = std::numeric_limits<T>::lowest();

    for (int64_t i = start; i < start + length; ++i) {
        maximum = std::max(maximum, data[i]);
    }

    T scaled_sumexp = 0;

    for (int64_t i = start; i < start + length; ++i) {
        scaled_sumexp += std::exp(data[i] - maximum);
    }

    if (length == 0) {
        return std::numeric_limits<T>::lowest();
    }

    return std::log(scaled_sumexp) + maximum;
}

at::Tensor segment_logsumexp_cpu(at::Tensor const &values, at::Tensor const &scopes) {
    auto output = at::empty({scopes.size(0)}, values.options());
    auto num_element_per_segment = std::max<decltype(values.size(0))>(values.size(0) / scopes.size(0), 1);
    auto grain_size = std::max<decltype(at::internal::GRAIN_SIZE)>(at::internal::GRAIN_SIZE / (6 * num_element_per_segment), 1);
    auto scopes_accessor = scopes.accessor<int64_t, 2>();

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "logsumexp_cpu", [&]() {
        auto output_ptr = output.accessor<scalar_t, 1>();
        auto values_ptr = values.accessor<scalar_t, 1>();

        at::parallel_for(0, scopes.size(0), grain_size, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                output_ptr[i] =
                    logsumexp(values_ptr, scopes_accessor[i][0], scopes_accessor[i][1]);
            }
        });
    });

    return output;
}

std::tuple<at::Tensor, at::Tensor> segment_argmax_cpu(at::Tensor const &values, at::Tensor const &scopes) {
    at::Tensor output_values;
    auto output_index = at::empty({scopes.size(0)}, scopes.options());

    auto num_element_per_segment = std::max<decltype(values.size(0))>(values.size(0) / scopes.size(0), 1);
    auto grain_size = std::max<decltype(at::internal::GRAIN_SIZE)>(at::internal::GRAIN_SIZE / (2 * num_element_per_segment), 1);

    auto scopes_accessor = scopes.accessor<int64_t, 2>();
    auto output_index_ptr = output_index.accessor<int64_t, 1>();

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "logsumexp_cpu", [&]() {
        output_values = at::full({scopes.size(0)}, std::numeric_limits<scalar_t>::lowest(), values.options());
        auto output_values_ptr = output_values.accessor<scalar_t, 1>();
        auto values_ptr = values.accessor<scalar_t, 1>();

        at::parallel_for(0, scopes.size(0), grain_size, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                std::tie(output_values_ptr[i], output_index_ptr[i]) =
                    reduce_max(values_ptr, scopes_accessor[i][0], scopes_accessor[i][1]);
            }
        });
    });

    return std::make_tuple(output_values, output_index);
}

at::Tensor segment_logsumexp_backward_cpu(at::Tensor const &grad_output, at::Tensor const &input,
                                          at::Tensor const &logsumexp, at::Tensor const &lengths) {
    auto grad_output_repeat = at::repeat_interleave(grad_output, lengths, 0);
    auto logsumexp_rep = at::repeat_interleave(logsumexp, lengths, 0);

    return logsumexp_rep.sub_(input).neg_().exp_().mul_(std::move(grad_output_repeat));
}

} // namespace


at::Tensor segment_logsumexp_gpu(at::Tensor const &values, at::Tensor const &scopes);

at::Tensor segment_logsumexp(at::Tensor const &values, at::Tensor const &scopes) {
    if (values.is_cuda()) {
        return segment_logsumexp_gpu(values, scopes);
    } else {
        return segment_logsumexp_cpu(values, scopes);
    }
}


std::tuple<at::Tensor, at::Tensor> segment_argmax_gpu(at::Tensor const &values, at::Tensor const &scopes);

std::tuple<at::Tensor, at::Tensor> segment_argmax(at::Tensor const &values, at::Tensor const &scopes) {
    if (values.is_cuda()) {
        return segment_argmax_gpu(values, scopes);
    } else {
        return segment_argmax_cpu(values, scopes);
    }
}

at::Tensor segment_logsumexp_backward_gpu(at::Tensor const &grad_output, at::Tensor const &input,
                                          at::Tensor const &logsumexp, at::Tensor const &lengths);

at::Tensor segment_logsumexp_backward(at::Tensor const &grad_output, at::Tensor const &input,
    at::Tensor const &logsumexp, at::Tensor const &lengths) {
    if (grad_output.is_cuda()) {
        return segment_logsumexp_backward_gpu(grad_output, input, logsumexp, lengths);
    } else {
        return segment_logsumexp_backward_cpu(grad_output, input, logsumexp, lengths);
    }
}
