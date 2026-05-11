"""JIT compilation entry points (adapted from mllm-kernel; Highway/Ascend stripped)."""

from __future__ import annotations

import functools
import os
import pathlib
import re
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module


F = TypeVar("F", bound=Callable[..., Any])


def cache_once(fn: F) -> F:
    """Tiny memoizer that, unlike `functools.lru_cache`, plays nicely with `torch.compile`."""
    result_map: dict[tuple, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items(), key=lambda x: x[0])))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore[return-value]


def _make_tvm_ffi_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    """Resolve the path to the phyai_kernel package directory."""
    cur_dir = pathlib.Path(__file__).parent.parent.resolve()
    if (cur_dir / "cpu").exists() or (cur_dir / "include").exists():
        return cur_dir
    raise RuntimeError(f"Cannot find phyai-kernel path. Searched in: {cur_dir}")


class _PhyaiKernelTemplateArgListGenerator(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


PHYAI_KERNEL_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool, torch.dtype]
PHYAI_KERNEL_TEMPLATE_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
    torch.uint16: "uint16_t",
    torch.uint32: "uint32_t",
    torch.uint64: "uint64_t",
    torch.bool: "bool_t",
}


def make_cpp_args(
    *args: PHYAI_KERNEL_TEMPLATE_TYPE,
) -> _PhyaiKernelTemplateArgListGenerator:
    """Convert Python arguments to C++ template argument strings.

    Example:
        >>> make_cpp_args(16, torch.float16)
        ['16', 'fp16_t']
    """

    def _convert(arg: PHYAI_KERNEL_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return PHYAI_KERNEL_TEMPLATE_DTYPE_MAP[arg]
        raise TypeError(
            f"Unsupported argument type for phyai kernel template: {type(arg)}"
        )

    return _PhyaiKernelTemplateArgListGenerator(_convert(arg) for arg in args)


PHYAI_KERNEL_TOP_PATH = _resolve_kernel_path()
PHYAI_KERNEL_INCLUDE_DIR = PHYAI_KERNEL_TOP_PATH / "include"
PHYAI_KERNEL_CPU_PATH = PHYAI_KERNEL_TOP_PATH / "cpu"
PHYAI_KERNEL_CUDA_PATH = PHYAI_KERNEL_TOP_PATH / "cuda"

PHYAI_KERNEL_CPU_CSRC_DIR = PHYAI_KERNEL_CPU_PATH / "csrc"
PHYAI_KERNEL_CPU_INCLUDE_DIR = PHYAI_KERNEL_CPU_PATH / "include"

PHYAI_KERNEL_CUDA_CSRC_DIR = PHYAI_KERNEL_CUDA_PATH / "csrc"
PHYAI_KERNEL_CUDA_INCLUDE_DIR = PHYAI_KERNEL_CUDA_PATH / "include"

PHYAI_KERNEL_DEFAULT_CXX_FLAGS = ["-std=c++20", "-O3", "-fPIC"]
PHYAI_KERNEL_DEFAULT_CUDA_C_FLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
PHYAI_KERNEL_DEFAULT_LDFLAGS: list[str] = []
PHYAI_KERNEL_CACHE_ROOT = pathlib.Path(os.path.expanduser("~/.cache/phyai_kernel"))


JITKernelInfo: TypeAlias = Dict[str, Any]
_JIT_KERNEL_REGISTRY: dict[str, JITKernelInfo] = {}
_JIT_KERNEL_REGISTRY_LOCK = threading.Lock()


def _make_jit_kernel_registry_key(
    *,
    fn: Callable[..., Any],
    export_name: str,
    template_cpp_args: Tuple[str, ...],
    device: Literal["auto", "cpu", "cuda"],
) -> str:
    template_key = ",".join(template_cpp_args) if template_cpp_args else "-"
    return (
        f"{fn.__module__}.{fn.__qualname__}|"
        f"export={export_name}|device={device}|template={template_key}"
    )


def register_jit_kernel(info: JITKernelInfo) -> None:
    key = str(info["key"])
    with _JIT_KERNEL_REGISTRY_LOCK:
        _JIT_KERNEL_REGISTRY[key] = info


def get_jit_kernel_registry() -> dict[str, JITKernelInfo]:
    with _JIT_KERNEL_REGISTRY_LOCK:
        return {k: v.copy() for k, v in _JIT_KERNEL_REGISTRY.items()}


def clear_jit_kernel_registry() -> None:
    with _JIT_KERNEL_REGISTRY_LOCK:
        _JIT_KERNEL_REGISTRY.clear()


def _sanitize_cache_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return sanitized or "jit_kernel"


def _resolve_build_directory(
    build_directory: str | None,
    default_name: str,
) -> str:
    if build_directory is None:
        cache_dir = PHYAI_KERNEL_CACHE_ROOT / _sanitize_cache_name(default_name)
    else:
        cache_dir = pathlib.Path(os.path.expanduser(build_directory))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _build_cache_name(
    device_prefix: str,
    func_name: str,
    template_args: Tuple[str, ...] = (),
) -> str:
    if not template_args:
        return f"{device_prefix}_{func_name}"
    return f"{device_prefix}_{func_name}_" + "_".join(template_args)


def _normalize_jit_template_args(
    args: (
        PHYAI_KERNEL_TEMPLATE_TYPE
        | List[PHYAI_KERNEL_TEMPLATE_TYPE]
        | Tuple[PHYAI_KERNEL_TEMPLATE_TYPE, ...]
        | None
    ),
) -> Tuple[PHYAI_KERNEL_TEMPLATE_TYPE, ...]:
    if args is None:
        return ()
    if isinstance(args, list):
        return tuple(args)
    if isinstance(args, tuple):
        return args
    return (args,)


def _iter_tensors(values: Tuple[Any, ...]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            tensors.extend(_iter_tensors(tuple(value.values())))
        elif isinstance(value, (tuple, list)):
            tensors.extend(_iter_tensors(tuple(value)))
    return tensors


def _resolve_target_device(
    device: Literal["auto", "cpu", "cuda"],
    call_args: Tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    has_cuda_config: bool,
) -> Literal["cpu", "cuda"]:
    if device != "auto":
        return device

    tensors = _iter_tensors(call_args + (call_kwargs,))
    has_cuda = any(tensor.is_cuda for tensor in tensors)
    has_cpu = any(not tensor.is_cuda for tensor in tensors)
    if has_cuda and has_cpu:
        raise ValueError(
            "Cannot infer device from mixed CPU/CUDA tensors. "
            "Please move tensors to one device or pass device='cpu'/'cuda'."
        )
    if has_cuda:
        return "cuda"
    if has_cpu:
        return "cpu"
    if has_cuda_config:
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def jit(
    *,
    args: (
        PHYAI_KERNEL_TEMPLATE_TYPE
        | List[PHYAI_KERNEL_TEMPLATE_TYPE]
        | Tuple[PHYAI_KERNEL_TEMPLATE_TYPE, ...]
        | None
    ) = None,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    func_name: str | None = None,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cxx_flags: List[str] | None = None,
    extra_cuda_cxx_flags: List[str] | None = None,
    extra_ld_flags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Callable[[F], F]:
    """Decorator for JIT kernels that injects the compiled module as the first argument.

    Example:
        >>> @jit(args=[16], cuda_files=["add_constant.cuh"], device="auto")
        ... def add_constant_16(compiled_module, dst, src):
        ...     compiled_module.add_constant_16(dst, src)
    """
    template_args = _normalize_jit_template_args(args)
    template_cpp_args = make_cpp_args(*template_args)

    def decorator(fn: F) -> F:
        export_name = func_name or fn.__name__
        has_cuda_config = bool(cuda_files or cuda_wrappers)

        default_kernel_name = (
            f"{export_name}<{template_cpp_args}>" if template_cpp_args else export_name
        )

        resolved_cpp_wrappers = (
            [(export_name, default_kernel_name)]
            if cpp_wrappers is None
            else cpp_wrappers
        )
        resolved_cuda_wrappers = (
            [(export_name, default_kernel_name)]
            if cuda_wrappers is None
            else cuda_wrappers
        )
        template_cpp_args_tuple = tuple(str(arg) for arg in template_cpp_args)
        registry_key = _make_jit_kernel_registry_key(
            fn=fn,
            export_name=export_name,
            template_cpp_args=template_cpp_args_tuple,
            device=device,
        )

        register_jit_kernel(
            {
                "key": registry_key,
                "module": fn.__module__,
                "qualified_name": fn.__qualname__,
                "export_name": export_name,
                "decorator_device": device,
                "has_cuda_config": has_cuda_config,
                "template_args": template_cpp_args_tuple,
                "cpp_files": tuple(cpp_files or ()),
                "cuda_files": tuple(cuda_files or ()),
                "cpp_wrappers": tuple(resolved_cpp_wrappers),
                "cuda_wrappers": tuple(resolved_cuda_wrappers),
                "build_directory": build_directory,
            }
        )

        @cache_once
        def _load_module(target_device: Literal["cpu", "cuda"]):
            resolved_build_directory = _resolve_build_directory(
                build_directory,
                _build_cache_name(target_device, export_name, tuple(template_cpp_args)),
            )
            if target_device == "cuda":
                if not has_cuda_config:
                    raise ValueError(
                        f"JIT function '{export_name}' resolved to CUDA but no CUDA "
                        "entry was provided. Please set cuda_files/cuda_wrappers or "
                        "set device='cpu'."
                    )
                return load_cuda_jit(
                    export_name,
                    *template_cpp_args,
                    cpp_files=cpp_files,
                    cuda_files=cuda_files,
                    cpp_wrappers=resolved_cpp_wrappers,
                    cuda_wrappers=resolved_cuda_wrappers,
                    extra_cxx_flags=extra_cxx_flags,
                    extra_cuda_cxx_flags=extra_cuda_cxx_flags,
                    extra_ld_flags=extra_ld_flags,
                    extra_include_paths=extra_include_paths,
                    build_directory=resolved_build_directory,
                )
            return load_cpu_jit(
                export_name,
                *template_cpp_args,
                cpp_files=cpp_files,
                cpp_wrappers=resolved_cpp_wrappers,
                extra_cxx_flags=extra_cxx_flags,
                extra_ld_flags=extra_ld_flags,
                extra_include_paths=extra_include_paths,
                build_directory=resolved_build_directory,
            )

        @functools.wraps(fn)
        def wrapper(*call_args: Any, **call_kwargs: Any):
            target_device = _resolve_target_device(
                device,
                call_args,
                call_kwargs,
                has_cuda_config=has_cuda_config,
            )
            module = _load_module(target_device)
            return fn(module, *call_args, **call_kwargs)

        return cast(F, wrapper)

    return decorator


def load_cpu_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    extra_cxx_flags: List[str] | None = None,
    extra_ld_flags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """Load a CPU JIT kernel module via tvm-ffi `load_inline`."""
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cpp_wrappers = cpp_wrappers or []
    extra_cxx_flags = extra_cxx_flags or []
    extra_ld_flags = extra_ld_flags or []
    extra_include_paths = extra_include_paths or []

    include_paths = [str(PHYAI_KERNEL_CPU_INCLUDE_DIR), str(PHYAI_KERNEL_INCLUDE_DIR)]
    include_paths.extend(extra_include_paths)

    cxx_flags = PHYAI_KERNEL_DEFAULT_CXX_FLAGS.copy()
    cxx_flags.extend(extra_cxx_flags)

    cpp_paths = [(PHYAI_KERNEL_CPU_CSRC_DIR / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_tvm_ffi_wrapper(tup) for tup in cpp_wrappers]
    build_directory = _resolve_build_directory(
        build_directory,
        _build_cache_name(
            "cpu",
            str(args[0]) if args else "jit_kernel",
            tuple(str(arg) for arg in args[1:]),
        ),
    )

    return load_inline(
        "phyai_jit_kernel_cpu_" + "_".join(str(arg) for arg in args),
        cpp_sources=cpp_sources,
        cuda_sources=[],
        extra_cflags=cxx_flags,
        extra_cuda_cflags=[],
        extra_ldflags=PHYAI_KERNEL_DEFAULT_LDFLAGS + extra_ld_flags,
        extra_include_paths=include_paths,
        build_directory=build_directory,
    )


def load_cuda_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cxx_flags: List[str] | None = None,
    extra_cuda_cxx_flags: List[str] | None = None,
    extra_ld_flags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """Load a CUDA JIT kernel module via tvm-ffi `load_inline`."""
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cxx_flags = extra_cxx_flags or []
    extra_cuda_cxx_flags = extra_cuda_cxx_flags or []
    extra_ld_flags = extra_ld_flags or []
    extra_include_paths = extra_include_paths or []

    include_paths = [str(PHYAI_KERNEL_CUDA_INCLUDE_DIR), str(PHYAI_KERNEL_INCLUDE_DIR)]
    include_paths.extend(extra_include_paths)

    cpp_paths = [(PHYAI_KERNEL_CUDA_CSRC_DIR / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_tvm_ffi_wrapper(tup) for tup in cpp_wrappers]

    cuda_paths = [(PHYAI_KERNEL_CUDA_CSRC_DIR / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_tvm_ffi_wrapper(tup) for tup in cuda_wrappers]
    build_directory = _resolve_build_directory(
        build_directory,
        _build_cache_name(
            "cuda",
            str(args[0]) if args else "jit_kernel",
            tuple(str(arg) for arg in args[1:]),
        ),
    )

    return load_inline(
        "phyai_jit_kernel_cuda_" + "_".join(str(arg) for arg in args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=PHYAI_KERNEL_DEFAULT_CXX_FLAGS + extra_cxx_flags,
        extra_cuda_cflags=PHYAI_KERNEL_DEFAULT_CUDA_C_FLAGS + extra_cuda_cxx_flags,
        extra_ldflags=PHYAI_KERNEL_DEFAULT_LDFLAGS + extra_ld_flags,
        extra_include_paths=include_paths,
        build_directory=build_directory,
    )
