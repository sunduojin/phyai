"""Pure-function helpers for SM round-up and capacity validation.

The round-up table mirrors flashinfer's ``get_sm_count_constraint`` so the
phyai layer reaches the same answer as the underlying backend; that lets
us validate capacity before delegating, with messages keyed to phyai's
exception hierarchy.
"""

from __future__ import annotations

from phyai.vgpu.exceptions import VGPURuntimeError


def get_sm_count_constraint(cc: tuple[int, int]) -> tuple[int, int]:
    """Return ``(minimum, alignment)`` for a CUDA compute capability.

    Mirrors ``flashinfer.green_ctx.get_sm_count_constraint``::

        CC 6.x          : (1, 1)
        CC 7.x          : (2, 2)
        CC 8.x          : (4, 2)
        CC 9.x or later : (8, 8)
    """
    major, minor = cc
    if major == 6:
        return (1, 1)
    if major == 7:
        return (2, 2)
    if major == 8:
        return (4, 2)
    if major >= 9:
        return (8, 8)
    raise VGPURuntimeError(
        f"phyai.vgpu does not support compute capability {major}.{minor} "
        "(green ctx requires CC >= 6.0)"
    )


def round_up_sm_count(req: int, cc: tuple[int, int]) -> int:
    """Round ``req`` up to satisfy CC's minimum and alignment requirements."""
    if req <= 0:
        raise VGPURuntimeError(f"sm_count must be positive, got {req}")
    minimum, alignment = get_sm_count_constraint(cc)
    val = max(req, minimum)
    return ((val + alignment - 1) // alignment) * alignment


def validate_total(rounded: list[int], total_sms: int) -> None:
    """Raise ``VGPURuntimeError`` when ``sum(rounded) > total_sms``.

    The check matches flashinfer's capacity guard so callers see the same
    failure surface whether they invoke phyai or flashinfer directly.
    """
    s = sum(rounded)
    if s > total_sms:
        raise VGPURuntimeError(
            f"requested SM total {s} (rounded {rounded}) exceeds device "
            f"capacity {total_sms}"
        )
