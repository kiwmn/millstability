#include <torch/extension.h>

#include "millstability/ei.h"

namespace py = pybind11;
using namespace millstability;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {

    // Release the GIL while the C++/CUDA call runs.
    module.doc() = "Milling stability: CUDA matrices, cuSOLVER eigenvalues and CUDA EI reduction";
    module.def("milling_stability_ei_cuda", &milling_stability_ei_cuda,
        "Return EI [spindle speed, cutting depth] on CUDA",
        py::arg("N"), py::arg("Kt"), py::arg("Kn"), py::arg("w0x"), py::arg("w0y"),
        py::arg("zetax"), py::arg("zetay"), py::arg("m_tx"), py::arg("m_ty"),
        py::arg("aD"), py::arg("up_or_down"), py::arg("stx"), py::arg("sty"),
        py::arg("w_st"), py::arg("w_fi"), py::arg("o_st"), py::arg("o_fi"),
        py::arg("m"), py::arg("device_id") = 0, py::call_guard<py::gil_scoped_release>());
    module.def("multi_parameter_single_point_ei", &multi_parameter_single_point_ei,
        "Return one CUDA EI per physical parameter row",
        py::arg("parameters"), py::arg("N"), py::arg("aD"), py::arg("up_or_down"),
        py::arg("m"), py::arg("w"), py::arg("o"), py::arg("device_id") = 0,
        py::call_guard<py::gil_scoped_release>());
    module.def("test_ei", &test_ei, "Return the default 201 x 101 CUDA EI grid",
        py::arg("device_id") = 0, py::call_guard<py::gil_scoped_release>());
}
