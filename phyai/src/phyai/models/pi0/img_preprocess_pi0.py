"""pi0 image preprocessing helpers.

The scheduler consumes already-normalized ``pixel_values`` with shape
``(B, 3, 3, image_size, image_size)``: batch, three camera views, RGB,
height, width. These helpers cover the common PIL/NumPy/Tensor path for
small examples and tests; production deployments can feed tensors
directly when preprocessing already happens upstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SIGLIP_IMAGE_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_IMAGE_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


def image_to_chw_tensor(image: Any) -> torch.Tensor:
    """Convert PIL/NumPy/Tensor image input to float ``(3, H, W)`` in ``[0, 1]``."""

    if isinstance(image, torch.Tensor):
        t = image.detach()
        if t.dim() != 3:
            raise ValueError(f"image tensor must be 3-D, got {tuple(t.shape)}.")
        if t.shape[0] == 3:
            out = t
        elif t.shape[-1] == 3:
            out = t.permute(2, 0, 1)
        else:
            raise ValueError(
                f"image tensor must have 3 channels, got shape {tuple(t.shape)}."
            )
        out = out.to(torch.float32)
        if out.max() > 1.0:
            out = out / 255.0
        return out

    arr = np.ascontiguousarray(np.asarray(image))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"image array must be (H, W, 3), got shape {arr.shape}.")
    out = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)
    if out.max() > 1.0:
        out = out / 255.0
    return out


def preprocess_pi0_image(
    image: Any,
    *,
    image_size: int = 224,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Preprocess one RGB image to SigLIP-style ``(3, H, W)`` tensor."""

    x = image_to_chw_tensor(image).unsqueeze(0)
    if x.shape[-2:] != (image_size, image_size):
        _, _, h, w = x.shape
        ratio = max(w / image_size, h / image_size)
        resized_h = max(1, int(h / ratio))
        resized_w = max(1, int(w / ratio))
        x = F.interpolate(
            x,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        pad_h0, rem_h = divmod(image_size - resized_h, 2)
        pad_w0, rem_w = divmod(image_size - resized_w, 2)
        x = F.pad(
            x,
            (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h),
            mode="constant",
            value=0.0,
        )
    mean = torch.tensor(SIGLIP_IMAGE_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(SIGLIP_IMAGE_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.squeeze(0).to(dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def preprocess_pi0_camera_stack(
    images: Sequence[Any],
    *,
    image_size: int = 224,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Preprocess one robot's three camera images to ``(3, 3, H, W)``."""

    if len(images) != 3:
        raise ValueError(f"pi0 expects exactly 3 camera images, got {len(images)}.")
    return torch.stack(
        [
            preprocess_pi0_image(
                img,
                image_size=image_size,
                dtype=dtype,
                device=device,
            )
            for img in images
        ],
        dim=0,
    )


__all__ = [
    "SIGLIP_IMAGE_MEAN",
    "SIGLIP_IMAGE_STD",
    "image_to_chw_tensor",
    "preprocess_pi0_camera_stack",
    "preprocess_pi0_image",
]
