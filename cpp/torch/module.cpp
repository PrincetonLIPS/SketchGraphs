#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "repeat_interleave.h"

namespace py = pybind11;

at::Tensor segment_logsumexp(at::Tensor const &values, at::Tensor const &scopes);
std::tuple<at::Tensor, at::Tensor> segment_argmax(at::Tensor const &values,
                                                  at::Tensor const &scopes);
at::Tensor segment_avg_pool1d(at::Tensor const &values, at::Tensor const &scopes);
std::tuple<at::Tensor, at::Tensor> segment_max_pool1d(at::Tensor const &values,
                                                      at::Tensor const &scopes);
at::Tensor segment_logsumexp_backward(at::Tensor const &grad_output, at::Tensor const &input,
                                      at::Tensor const &logsumexp, at::Tensor const &lengths);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segment_logsumexp", &segment_logsumexp, py::arg("values"), py::arg("scopes"));
    m.def("segment_argmax", &segment_argmax, py::arg("values"), py::arg("scopes"));
    m.def("segment_avg_pool1d", &segment_avg_pool1d, py::arg("values"), py::arg("scopes"));
    m.def("segment_max_pool1d_with_indices", &segment_max_pool1d, py::arg("values"),
          py::arg("scopes"));
    m.def("segment_logsumexp_backward", &segment_logsumexp_backward, py::arg("grad_output"),
          py::arg("values"), py::arg("logsumexp"), py::arg("lengths"));

    m.def("repeat_interleave_out", &genric::repeat_interleave_out,
          py::arg("out"), py::arg("self"), py::arg("repeats_or_scope"), py::arg("dim") = py::none());

    m.def("repeat_interleave_out_index", &genric::repeat_interleave_out_index,
          py::arg("repeats"), py::arg("out"));

    m.def("repeat_interleave_out_shape", &genric::repeat_interleave_out_shape,
          py::arg("values"), py::arg("repeats_or_scope"), py::arg("out_length"), py::arg("dim") = py::none());
}
