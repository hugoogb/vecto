"""The vectorisation pipeline and automatic parameter tuning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fidelity import color_fidelity, despeckle, mono_metrics
from .image import Prepared, load_rgba, prepare
from .options import COLOR_TUNABLE, MONO_TUNABLE, Options
from .svgtools import optimize_svg
from .trace import count_paths, mono_svg, trace_color, trace_mono, unique_fills

DEFAULT_TARGET = {"mono": 0.95, "color": 0.98}
DEFAULT_MAX_EDGE = 0.5  # output pixels

SEARCH_SPACE: dict[str, list[Any]] = {
    "alphamax": [0.5, 0.8, 1.0, 1.2, 1.334],
    "opttolerance": [0.1, 0.2, 0.4, 0.7, 1.0, 1.5],
    "colors": [3, 4, 5, 6, 7, 8],
    "layer_difference": [8, 16, 24, 32, 48, 64],
}


@dataclass
class Result:
    svg: str
    width: float
    height: float
    fidelity: float
    options: Options
    scale: float
    edge_error: float | None = None  # mono only, output pixels
    inverted: bool = False
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    candidates_tried: int = 1

    @property
    def bytes(self) -> int:
        return len(self.svg.encode("utf-8"))

    @property
    def paths(self) -> int:
        return count_paths(self.svg)

    @property
    def colors(self) -> int:
        return unique_fills(self.svg) if self.options.mode == "color" else 1

    def to_dict(self, include_svg: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "bytes": self.bytes,
            "paths": self.paths,
            "colors": self.colors,
            "fidelity": round(self.fidelity, 4),
            "edge_error": None if self.edge_error is None else round(self.edge_error, 3),
            "mode": self.options.mode,
            "upscale": self.scale,
            "inverted": self.inverted,
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
            "candidates_tried": self.candidates_tried,
            "options": self.options.to_dict(),
        }
        if include_svg:
            d["svg"] = self.svg
        return d


@dataclass(frozen=True)
class _Scored:
    svg: str
    fidelity: float
    edge_error: float | None

    def meets(self, target: float, max_edge: float) -> bool:
        edge_ok = self.edge_error is None or self.edge_error <= max_edge
        return self.fidelity >= target and edge_ok


def _resolve_fill(opts: Options, prep: Prepared) -> str:
    return prep.ink if opts.color == "auto" else opts.color


def _render(prep: Prepared, opts: Options) -> _Scored:
    """Trace with ``opts`` and measure how faithful the result is."""
    finish = (lambda s: optimize_svg(s, opts.precision)) if opts.optimize else (lambda s: s)

    if opts.mode == "color":
        svg = finish(trace_color(prep, opts))
        fid = color_fidelity(svg, prep.rgba)
        if opts.background:
            i = svg.index(">", svg.index("<svg")) + 1
            rect = f'<rect width="{prep.width}" height="{prep.height}" fill="{opts.background}"/>'
            svg = svg[:i] + rect + svg[i:]
        return _Scored(svg, fid, None)

    d = trace_mono(prep, opts)
    probe = finish(mono_svg(d, prep, "#000"))
    reference = despeckle(prep.mask, int(round(opts.turdsize * prep.scale**2)))
    fid, edge = mono_metrics(probe, reference, prep.scale)
    plain = opts.color in ("#000", "#000000", "black") or (
        opts.color == "auto" and prep.ink == "#000000"
    )
    if plain and not opts.background and not opts.outline:
        return _Scored(probe, fid, edge)
    final = mono_svg(
        d, prep, _resolve_fill(opts, prep),
        background=opts.background, outline=opts.outline, stroke_width=opts.stroke_width,
    )
    return _Scored(finish(final), fid, edge)


def _key(c: _Scored, target: float, max_edge: float) -> tuple:
    """Sort key: the smallest file that meets the goals; otherwise the most faithful."""
    if c.meets(target, max_edge):
        return (0, len(c.svg))
    edge_excess = 0.0 if c.edge_error is None else max(0.0, c.edge_error - max_edge)
    return (1, edge_excess, -c.fidelity)


def _tune(prep: Prepared, opts: Options, target: float, max_edge: float):
    names = COLOR_TUNABLE if opts.mode == "color" else MONO_TUNABLE
    cache: dict[tuple, _Scored] = {}

    def evaluate(o: Options) -> _Scored:
        k = tuple(getattr(o, n) for n in names)
        if k not in cache:
            cache[k] = _render(prep, o)
        return cache[k]

    best_opts, best = opts, evaluate(opts)
    for _ in range(2):  # coordinate descent, two sweeps
        improved = False
        for name in names:
            for value in SEARCH_SPACE[name]:
                cand_opts = best_opts.merged(**{name: value})
                cand = evaluate(cand_opts)
                if _key(cand, target, max_edge) < _key(best, target, max_edge):
                    best_opts, best, improved = cand_opts, cand, True
        if not improved:
            break
    return best_opts, best, len(cache)


def vectorize(
    source: str | Path | bytes | np.ndarray,
    options: Options | None = None,
    *,
    auto: bool = False,
    target: float | None = None,
    max_edge: float | None = None,
) -> Result:
    """Convert a raster image to SVG.

    ``source`` may be a path, raw image bytes, or an RGB(A)/greyscale numpy array.

    With ``auto=True`` the curve-fitting settings are searched for the smallest
    SVG whose fidelity is at least ``target`` and (mono) whose edges stay within
    ``max_edge`` output pixels of the source; if nothing qualifies, the most
    faithful candidate is returned with a warning.
    """
    started = time.perf_counter()
    opts = options or Options()

    if isinstance(source, np.ndarray):
        rgba = source
        if rgba.ndim == 2:
            rgba = np.dstack([rgba] * 3 + [np.full_like(rgba, 255)])
        elif rgba.shape[2] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, rgba.dtype)])
        rgba = rgba.astype(np.uint8)
    else:
        rgba = load_rgba(source)

    prep = prepare(rgba, opts)
    tried = 1
    if auto:
        goal = DEFAULT_TARGET[opts.mode] if target is None else target
        edge_goal = DEFAULT_MAX_EDGE if max_edge is None else max_edge
        # Edges can't be measured finer than one working pixel.
        edge_goal = max(edge_goal, 1.0 / prep.scale)
        opts, best, tried = _tune(prep, opts, goal, edge_goal)
        if not best.meets(goal, edge_goal):
            msg = f"Auto-tune could not reach fidelity {goal:.2f}"
            if best.edge_error is not None:
                msg += f" with edge error <= {edge_goal:g}px"
            prep.warnings.append(f"{msg}; returning the most faithful result")
    else:
        best = _render(prep, opts)

    return Result(
        svg=best.svg,
        width=round(prep.out_width, 2),
        height=round(prep.out_height, 2),
        fidelity=best.fidelity,
        edge_error=best.edge_error,
        options=opts,
        scale=prep.scale,
        inverted=prep.inverted,
        warnings=prep.warnings,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        candidates_tried=tried,
    )
