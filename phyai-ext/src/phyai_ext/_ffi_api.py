"""FFI bindings for globally-registered ``phyai_ext.*`` functions."""

from __future__ import annotations

import tvm_ffi
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB

# Load the extension library. Its ``TVM_FFI_STATIC_INIT_BLOCK`` registers every
# ``phyai_ext.*`` global function with the FFI registry as a side effect of the
# dlopen.
LIB = _FFI_LOAD_LIB("phyai-ext", "phyai_ext_core")

tvm_ffi.init_ffi_api("phyai_ext", __name__)
