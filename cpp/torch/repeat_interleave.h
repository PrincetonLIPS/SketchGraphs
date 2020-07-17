#pragma once

#include <ATen/ATen.h>

namespace genric {

at::Tensor &repeat_interleave_cpu_out(const at::Tensor &repeats, at::Tensor &out);
at::Tensor &repeat_interleave_gpu_out(const at::Tensor &repeats, at::Tensor &out);
at::Tensor &repeat_interleave_out_index(const at::Tensor &repeats, at::Tensor &out);
at::Tensor &repeat_interleave_out(at::Tensor &out, const at::Tensor &self,
                                  const at::Tensor &repeats_or_scope, c10::optional<int64_t> dim);

at::Tensor repeat_interleave_out_shape(const at::Tensor &self, const at::Tensor &repeats_or_scope,
                                       int64_t out_length, c10::optional<int64_t> dim);

at::Tensor &repeat_interleave_cpu_out_scope(const at::Tensor &scope, at::Tensor &out);
at::Tensor &repeat_interleave_gpu_out_scope(const at::Tensor &scope, at::Tensor &out);
at::Tensor &repeat_interleave_out_index_scope(const at::Tensor &scope, at::Tensor &out);

} // namespace genric
