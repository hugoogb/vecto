"""MCP server exposing vecto to AI assistants (``vecto mcp``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import vectorize
from .options import PRESETS, build_options

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = (
    "vecto converts raster images (PNG, JPG, WebP, ...) to SVG. Pick a preset that matches "
    "the image (logo, icon, signature, lineart, cut, color, photo, pixel); use auto=true to "
    "let vecto search for the smallest SVG meeting a fidelity target. Every call reports "
    "fidelity (0-1): mono = IoU of ink pixels, color = 1 - mean colour error."
)

server = _Server("vecto", instructions=INSTRUCTIONS)


@server.tool()
def vectorize_image(
    input_path: str,
    output_path: str | None = None,
    preset: str | None = None,
    mode: str | None = None,
    auto: bool = False,
    target: float | None = None,
    color: str | None = None,
    background: str | None = None,
    crop: bool | None = None,
    invert: bool | None = None,
    threshold: int | None = None,
    return_svg: bool = False,
) -> dict[str, Any]:
    """Convert a raster image file to SVG and write it to disk.

    Args:
        input_path: Path of the image to convert.
        output_path: Where to write the SVG. Defaults to the input path with .svg.
        preset: logo, icon, signature, lineart, cut, color, photo or pixel.
        mode: "mono" (single colour) or "color". Overrides the preset.
        auto: Search tracing settings for the smallest SVG reaching `target`.
        target: Fidelity goal for auto (0-1).
        color: Fill colour for mono output, or "auto" to keep the ink colour.
        background: Optional solid background colour.
        crop: Trim empty margins.
        invert: Force inversion (default detects dark backgrounds).
        threshold: Manual binarisation cut-off 0-255 (default automatic).
        return_svg: Include the SVG markup in the response.
    """
    src = Path(input_path).expanduser()
    if not src.is_file():
        raise ValueError(f"File not found: {src}")
    opts = build_options(
        preset, mode=mode, color=color, background=background, crop=crop,
        invert=invert, threshold=threshold,
    )
    result = vectorize(src, opts, auto=auto, target=target)
    dst = Path(output_path).expanduser() if output_path else src.with_suffix(".svg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(result.svg, encoding="utf-8")
    data = result.to_dict(include_svg=return_svg)
    data["output_path"] = str(dst)
    return data


@server.tool()
def list_presets() -> dict[str, str]:
    """List vecto's presets and what kind of image each one is for."""
    return {p.name: p.summary for p in PRESETS.values()}


def run() -> None:
    server.run()
