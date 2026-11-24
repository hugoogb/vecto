"""Loading and preprocessing raster images."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .options import Options

# Images whose longest side is below this get upscaled before tracing.
AUTO_UPSCALE_TARGET = 1024
AUTO_UPSCALE_MAX = 4.0


def load_rgba(source: str | Path | bytes) -> np.ndarray:
    """Load any Pillow-readable image (path or raw bytes) as an RGBA uint8 array."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img = ImageOps.exif_transpose(img)
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            img = img.point(lambda v: v / 256).convert("L")
        return np.array(img.convert("RGBA"))
    except Exception as exc:  # Pillow raises many different types
        name = "<bytes>" if isinstance(source, (bytes, bytearray)) else str(source)
        raise ValueError(f"Cannot read image {name}: {exc}") from exc


def has_transparency(rgba: np.ndarray) -> bool:
    return bool((rgba[..., 3] < 255).any())


def is_antialiased(gray: np.ndarray) -> bool:
    """True when edges carry intermediate grey levels (sub-pixel information).

    Upscaling only adds accuracy when there is sub-pixel information to recover;
    on hard-edged (aliased) art it just turns pixel steps into bigger steps.
    """
    hist = np.bincount(gray.ravel(), minlength=256)
    peaks = np.argsort(hist)[-2:]  # the two dominant tones (ink and paper)
    levels = np.arange(256)
    near_peak = (np.abs(levels[:, None] - peaks[None, :]) <= 8).any(axis=1)
    return bool(hist[~near_peak].sum() / gray.size > 0.002)


def upscale_factor(
    width: int, height: int, requested: float | None, antialiased: bool = True
) -> float:
    if requested is not None:
        if requested <= 0:
            raise ValueError("upscale must be > 0")
        return float(requested)
    if not antialiased:
        return 1.0
    longest = max(width, height)
    if longest >= AUTO_UPSCALE_TARGET:
        return 1.0
    return float(min(AUTO_UPSCALE_MAX, max(1.0, round(AUTO_UPSCALE_TARGET / longest))))


def _resize(arr: np.ndarray, factor: float, nearest: bool = False) -> np.ndarray:
    if factor == 1.0:
        return arr
    h, w = arr.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    return cv2.resize(arr, size, interpolation=interp)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _pad_bbox(bbox, pad: int, width: int, height: int):
    x0, y0, x1, y1 = bbox
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def _composite(rgba: np.ndarray, bg: int) -> np.ndarray:
    """Flatten RGBA onto a uniform grey level, returning RGB float32."""
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgb * alpha + bg * (1.0 - alpha)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


@dataclass
class Prepared:
    """A preprocessed image ready for tracing."""

    scale: float  # working pixels per output pixel
    width: int  # working resolution
    height: int
    mask: np.ndarray | None = None  # mono: True where ink is
    rgba: np.ndarray | None = None  # color: working RGBA
    ink: str = "#000000"  # detected foreground colour (mono)
    inverted: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def out_width(self) -> float:
        return self.width / self.scale

    @property
    def out_height(self) -> float:
        return self.height / self.scale


def _hex(rgb) -> str:
    r, g, b = (int(round(c)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _ink_color(rgba: np.ndarray, mask_small: np.ndarray) -> str:
    sel = mask_small & (rgba[..., 3] > 127)
    if sel.sum() < 4:
        return "#000000"
    rgb = np.median(rgba[..., :3][sel].astype(np.float32), axis=0)
    if rgb.max() < 48:  # practically black: keep it clean
        return "#000000"
    return _hex(rgb)


def prepare_mono(rgba: np.ndarray, opts: Options) -> Prepared:
    h0, w0 = rgba.shape[:2]
    warnings: list[str] = []

    if has_transparency(rgba):
        # Put the content on whichever background contrasts with it, so both
        # dark-on-transparent and light-on-transparent logos survive.
        opaque = rgba[..., 3] > 127
        if opaque.any():
            lum = _to_gray(rgba[..., :3][opaque].reshape(-1, 1, 3).astype(np.float32))
            bg = 255 if lum.mean() < 128 else 0
        else:
            bg = 255
            warnings.append("Image is fully transparent")
        gray = _to_gray(_composite(rgba, bg))
    else:
        gray = _to_gray(rgba[..., :3].astype(np.float32))

    scale = upscale_factor(w0, h0, opts.upscale, is_antialiased(gray))
    work = _resize(gray, scale)

    if opts.blur and opts.blur > 1:
        k = int(opts.blur) | 1  # kernel must be odd
        work = cv2.GaussianBlur(work, (k, k), 0)

    if opts.adaptive:
        block = max(3, int(round(min(work.shape) / 16)) | 1)
        binary = cv2.adaptiveThreshold(
            work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10
        )
    elif opts.threshold is None:
        _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if not 0 <= opts.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        _, binary = cv2.threshold(work, opts.threshold, 255, cv2.THRESH_BINARY)

    mask = binary == 0  # dark pixels are ink

    inverted = False
    if opts.invert is None:
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        if border.mean() > 0.5:  # the background touching the edges is dark
            mask = ~mask
            inverted = True
    elif opts.invert:
        mask = ~mask
        inverted = True

    if not mask.any():
        warnings.append("No foreground found; try --threshold or --invert")
    elif mask.mean() > 0.95:
        warnings.append("Almost everything is foreground; try --invert or --threshold")

    mask_small = _resize(mask.astype(np.uint8), 1 / scale, nearest=True).astype(bool)
    if mask_small.shape != (h0, w0):
        mask_small = cv2.resize(
            mask_small.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    ink = _ink_color(rgba, mask_small)

    if opts.crop:
        bbox = _bbox(mask)
        if bbox:
            pad = int(round(opts.padding * scale))
            x0, y0, x1, y1 = _pad_bbox(bbox, pad, mask.shape[1], mask.shape[0])
            mask = mask[y0:y1, x0:x1]

    return Prepared(
        scale=scale, width=mask.shape[1], height=mask.shape[0], mask=mask,
        ink=ink, inverted=inverted, warnings=warnings,
    )


def prepare_color(rgba: np.ndarray, opts: Options) -> Prepared:
    h0, w0 = rgba.shape[:2]
    # Colour tracing gains nothing from interpolation: the smoothed edges turn
    # into extra gradient layers (measured ~60x larger files at 4x). Only
    # upscale when explicitly asked.
    scale = upscale_factor(w0, h0, opts.upscale if opts.upscale is not None else 1.0)
    nearest = opts.trace_mode == "none"
    work = _resize(rgba, scale, nearest=nearest)

    if opts.crop:
        if has_transparency(work):
            content = work[..., 3] > 0
        else:
            edge = np.concatenate([work[0], work[-1], work[:, 0], work[:, -1]])[..., :3]
            bg = np.median(edge, axis=0)
            content = np.abs(work[..., :3].astype(int) - bg).max(axis=2) > 30
        bbox = _bbox(content)
        if bbox:
            pad = int(round(opts.padding * scale))
            x0, y0, x1, y1 = _pad_bbox(bbox, pad, work.shape[1], work.shape[0])
            work = work[y0:y1, x0:x1]

    return Prepared(
        scale=scale, width=work.shape[1], height=work.shape[0],
        rgba=np.ascontiguousarray(work),
    )


def prepare(rgba: np.ndarray, opts: Options) -> Prepared:
    return prepare_color(rgba, opts) if opts.mode == "color" else prepare_mono(rgba, opts)


def encode_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()
