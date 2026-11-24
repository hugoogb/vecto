"""Tracing backends: potrace (mono) and vtracer (colour)."""

from __future__ import annotations

import re

import potrace
import vtracer

from .image import Prepared, encode_png
from .options import Options


def _num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _pt(p) -> str:
    return f"{_num(p.x)} {_num(p.y)}"


def trace_mono(prep: Prepared, opts: Options) -> str:
    """Trace the ink mask and return a single SVG path ``d`` string."""
    assert prep.mask is not None
    if not prep.mask.any():
        return ""
    # potracer treats True as background, then inverts internally.
    bitmap = potrace.Bitmap(~prep.mask)
    # turdsize is an area in output pixels; scale it to the working resolution
    # so upscaling doesn't let specks slip through.
    plist = bitmap.trace(
        turdsize=int(round(opts.turdsize * prep.scale**2)),
        alphamax=float(opts.alphamax),
        opticurve=opts.opttolerance > 0,
        opttolerance=float(opts.opttolerance),
    )
    parts: list[str] = []
    for curve in plist:
        parts.append(f"M{_pt(curve.start_point)}")
        for seg in curve.segments:
            if seg.is_corner:
                parts.append(f"L{_pt(seg.c)}L{_pt(seg.end_point)}")
            else:
                parts.append(f"C{_pt(seg.c1)} {_pt(seg.c2)} {_pt(seg.end_point)}")
        parts.append("Z")
    return "".join(parts)


def mono_svg(
    d: str,
    prep: Prepared,
    fill: str,
    background: str | None = None,
    outline: bool = False,
    stroke_width: float = 1.0,
) -> str:
    w, h = prep.width, prep.height
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(prep.out_width)}" '
        f'height="{_num(prep.out_height)}" viewBox="0 0 {w} {h}">'
    )
    body = []
    if background:
        body.append(f'<rect width="{w}" height="{h}" fill="{background}"/>')
    if d:
        if outline:
            sw = _num(stroke_width * prep.scale)
            body.append(f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="{sw}"/>')
        else:
            body.append(f'<path d="{d}" fill="{fill}" fill-rule="evenodd"/>')
    return head + "".join(body) + "</svg>"


_SVG_ROOT = re.compile(r"<svg\b[^>]*>", re.S)


def trace_color(prep: Prepared, opts: Options) -> str:
    """Trace the colour image with vtracer and return a complete SVG document."""
    assert prep.rgba is not None
    if not 1 <= opts.colors <= 8:
        raise ValueError("colors must be between 1 and 8")
    svg = vtracer.convert_raw_image_to_svg(
        encode_png(prep.rgba),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode=opts.trace_mode,
        filter_speckle=int(opts.filter_speckle),
        color_precision=int(opts.colors),
        layer_difference=int(opts.layer_difference),
        corner_threshold=int(opts.corner_threshold),
    )
    root = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(prep.out_width)}" '
        f'height="{_num(prep.out_height)}" viewBox="0 0 {prep.width} {prep.height}">'
    )
    svg = _SVG_ROOT.sub(root, svg, count=1)
    # drop the XML prolog / generator comment vtracer emits before <svg>
    return svg[svg.index("<svg"):]


def count_paths(svg: str) -> int:
    return len(re.findall(r"<path\b", svg))


def unique_fills(svg: str) -> int:
    return len(set(re.findall(r'fill="(#[0-9A-Fa-f]{3,8})"', svg)))

