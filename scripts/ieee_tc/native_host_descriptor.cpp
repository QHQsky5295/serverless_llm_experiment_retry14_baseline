// D747 diagnostic only; no serving integration, cache, or tensor retention.
// Build against the actual qualified PyTorch environment, not a stable ABI.
// Read each tensor's CURRENT metadata with one Python/native crossing. Keep
// PyTorch's is_pinned dispatcher and the GIL; no direct CUDA API substitution.
#include <torch/csrc/autograd/python_variable.h>
#include <cstdint>

static PyObject* read_descriptor(PyObject*, PyObject* args) {
    PyObject* object = nullptr;
    const char* name = nullptr;
    if (!PyArg_ParseTuple(args, "Os", &object, &name)) return nullptr;
    // Native vLLM loader produces exact Tensors. Subclasses have dispatch
    // semantics this bounded witness deliberately does not claim to preserve.
    if (Py_TYPE(object) != reinterpret_cast<PyTypeObject*>(THPVariableClass)) {
        PyErr_Format(PyExc_ValueError, "expected exact native Tensor: %s", name);
        return nullptr;
    }
    try {
        const auto& tensor = THPVariable_Unpack(object);
        if (!tensor.is_cpu() || tensor.layout() != c10::kStrided ||
            tensor.dim() != 2 || tensor.numel() <= 0) {
            PyErr_Format(PyExc_ValueError,
                         "expected nonempty dense native CPU LoRA tensor: %s", name);
            return nullptr;
        }
        const auto& storage = tensor.storage();
        const auto pointer = reinterpret_cast<uintptr_t>(storage.data());
        const auto capacity = storage.nbytes();
        if (!pointer || !capacity) {
            PyErr_Format(PyExc_ValueError, "empty native CPU LoRA storage: %s", name);
            return nullptr;
        }
        const bool pinned = tensor.is_pinned();
        return Py_BuildValue("(KKO)", static_cast<unsigned long long>(pointer),
                             static_cast<unsigned long long>(capacity),
                             pinned ? Py_True : Py_False);
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

static PyMethodDef methods[] = {
    {"read", read_descriptor, METH_VARARGS, "Fresh exact CPU tensor descriptor."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_ieee_native_host_descriptor", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr
};
PyMODINIT_FUNC PyInit__ieee_native_host_descriptor() { return PyModule_Create(&module); }
