"""phyai-kernel CLI helpers."""

from __future__ import annotations

import argparse


def show_env() -> None:
    """Print runtime environment summary (OS, Python, CUDA, torch)."""
    import os
    import platform
    import shutil
    import sys

    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None

    print("=== phyai-kernel environment ===")
    print(f"system: {platform.system()}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")

    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    print(f"CUDA_HOME: {cuda_home or 'not set'}")
    print(f"nvcc: {shutil.which('nvcc') or 'not found'}")
    print(f"nvidia-smi: {shutil.which('nvidia-smi') or 'not found'}")

    if torch is None:
        print("torch_installed: no")
        return

    print(f"torch: {torch.__version__} (built with CUDA {torch.version.cuda})")
    print(f"torch_cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            print(f"gpu[{idx}]: {name}, sm_{major}{minor}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phyai-kernel",
        description="phyai-kernel helper commands.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["show-env"],
        help="Helper to run.",
    )
    args = parser.parse_args()

    if args.command == "show-env":
        show_env()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
