"""SVG optimisation and rasterisation helpers."""

from __future__ import annotations

import io
import re

import numpy as np
import resvg_py
from PIL import Image
from scour import scour


def optimize_svg(svg: str, precision: int = 5) -> str:
    opts = scour.sanitizeOptions()
    opts.digits = precision
    opts.cdigits = precision
    opts.enable_viewboxing = False  # vecto writes width/height/viewBox itself
    opts.remove_metadata = True
    opts.remove_descriptive_elements = True
    opts.strip_comments = True
    opts.strip_xml_prolog = True
    opts.strip_ids = True
    opts.shorten_ids = True
    opts.indent_type = "none"
    opts.newlines = False
    opts.quiet = True
    out = scour.scourString(svg, opts)
    return re.sub(r"\s*\n\s*", "", out).strip()


def _png(svg: str, width: int | None = None, height: int | None = None,
         zoom: float | None = None) -> bytes:
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height, zoom=zoom))


_ROOT = re.compile(r"<svg\b[^>]*>", re.S)
_SIZE_ATTR = re.compile(r'\s(width|height)="[^"]*"')


def render_rgba(svg: str, width: int, height: int) -> np.ndarray:
    """Rasterise an SVG (which must have a viewBox) to exactly ``width`` x ``height``."""
    # Give the root integer pixel dimensions: resvg mis-scales fractional ones.
    root = _ROOT.search(svg)
    tag = _SIZE_ATTR.sub("", root.group(0))
    tag = tag[:4] + f' width="{width}" height="{height}"' + tag[4:]
    svg = svg[: root.start()] + tag + svg[root.end():]
    img = Image.open(io.BytesIO(_png(svg))).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height))
    return np.array(img)


def write_preview(svg: str, path, scale: float = 2.0) -> None:
    with open(path, "wb") as f:
        f.write(_png(svg, zoom=scale))
