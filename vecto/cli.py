"""Command-line interface."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from dataclasses import fields
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from . import __version__
from .core import Result, vectorize
from .options import PRESETS, Options, build_options
from .svgtools import write_preview

app = typer.Typer(
    help="Turn raster images (PNG, JPG, WebP, ...) into clean SVG vectors.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_EXTS = "png,jpg,jpeg,webp,bmp,gif,tif,tiff"

# --------------------------------------------------------------------------
# Shared options. Every one defaults to None so presets can fill the gaps and
# only flags the user actually typed override them.
# --------------------------------------------------------------------------

PresetOpt = Annotated[Optional[str], typer.Option(
    "--preset", "-p", help=f"Starting point: {', '.join(PRESETS)}. See `vecto presets`.")]
ModeOpt = Annotated[Optional[str], typer.Option(
    "--mode", "-m", help="mono (one colour, potrace) or color (vtracer).")]
AutoOpt = Annotated[bool, typer.Option(
    "--auto", help="Search tracing settings for the smallest SVG that meets --target.")]
TargetOpt = Annotated[Optional[float], typer.Option(
    "--target", min=0, max=1, help="Fidelity goal for --auto (default 0.95 mono, 0.98 color).")]
MaxEdgeOpt = Annotated[Optional[float], typer.Option(
    "--max-edge", min=0, help="[mono] Max edge drift for --auto, in pixels (default 0.5).")]
ThresholdOpt = Annotated[Optional[int], typer.Option(
    "--threshold", "-t", min=0, max=255, help="Binarisation cut-off. Default: automatic (Otsu).")]
AdaptiveOpt = Annotated[Optional[bool], typer.Option(
    "--adaptive/--no-adaptive", help="Local thresholding for uneven lighting (scans, photos).")]
InvertOpt = Annotated[Optional[bool], typer.Option(
    "--invert/--no-invert", help="Force inversion. Default: detect dark backgrounds.")]
BlurOpt = Annotated[Optional[int], typer.Option(
    "--blur", min=0, help="Denoise blur kernel size before thresholding (0 = off).")]
UpscaleOpt = Annotated[Optional[float], typer.Option(
    "--upscale", "-u", min=0.1,
    help="Enlarge before tracing. Default: auto for small anti-aliased mono images, 1 for color.")]
CropOpt = Annotated[Optional[bool], typer.Option(
    "--crop/--no-crop", help="Trim empty margins around the content.")]
PaddingOpt = Annotated[Optional[int], typer.Option(
    "--padding", min=0, help="Margin kept around cropped content, in pixels.")]
ColorOpt = Annotated[Optional[str], typer.Option(
    "--color", "-c", help="Fill colour for mono output, or 'auto' to keep the ink colour.")]
BackgroundOpt = Annotated[Optional[str], typer.Option(
    "--background", "-b", help="Add a solid background (default: transparent).")]
OutlineOpt = Annotated[Optional[bool], typer.Option(
    "--outline/--fill", help="Stroke shapes instead of filling them (laser cut lines).")]
StrokeOpt = Annotated[Optional[float], typer.Option(
    "--stroke-width", min=0, help="Stroke width for --outline, in pixels.")]
TurdOpt = Annotated[Optional[int], typer.Option(
    "--turdsize", min=0, help="[mono] Remove specks up to this many pixels.")]
AlphaOpt = Annotated[Optional[float], typer.Option(
    "--alphamax", min=0, max=1.334, help="[mono] Corner smoothing: 0 sharp ... 1.334 round.")]
OptTolOpt = Annotated[Optional[float], typer.Option(
    "--opttolerance", min=0, help="[mono] Curve simplification tolerance.")]
ColorsOpt = Annotated[Optional[int], typer.Option(
    "--colors", min=1, max=8, help="[color] Colour precision, bits per channel (1-8).")]
SpeckleOpt = Annotated[Optional[int], typer.Option(
    "--filter-speckle", min=0, help="[color] Discard patches smaller than this.")]
LayerOpt = Annotated[Optional[int], typer.Option(
    "--layer-difference", min=0, help="[color] Minimum colour difference between layers.")]
TraceModeOpt = Annotated[Optional[str], typer.Option(
    "--trace-mode", help="[color] spline (smooth), polygon or none (pixel exact).")]
OptimizeOpt = Annotated[Optional[bool], typer.Option(
    "--optimize/--no-optimize", help="Minify the SVG with Scour.")]
PreviewOpt = Annotated[bool, typer.Option(
    "--preview", help="Also write a PNG rendering next to the SVG.")]
JsonOpt = Annotated[bool, typer.Option(
    "--json", help="Print machine-readable results to stdout.")]
QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Only print errors.")]

_OPTION_NAMES = {f.name for f in fields(Options)}


def _options(preset: str | None, values: dict[str, Any]) -> Options:
    if values.get("mode") not in (None, "mono", "color"):
        raise typer.BadParameter("--mode must be 'mono' or 'color'")
    if values.get("trace_mode") not in (None, "spline", "polygon", "none"):
        raise typer.BadParameter("--trace-mode must be spline, polygon or none")
    try:
        return build_options(preset, **{k: v for k, v in values.items() if k in _OPTION_NAMES})
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _err(msg: str) -> None:
    typer.echo(msg, err=True)


def _summary(r: Result) -> str:
    parts = [f"{r.width:g}x{r.height:g}", f"{r.bytes / 1024:.1f} KB", f"{r.paths} paths"]
    if r.options.mode == "color":
        parts.append(f"{r.colors} colours")
    parts.append(f"fidelity {r.fidelity * 100:.1f}%")
    if r.edge_error is not None:
        parts.append(f"edge error {r.edge_error:.2f}px")
    if r.candidates_tried > 1:
        parts.append(f"{r.candidates_tried} settings tried")
    return ", ".join(parts)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"vecto {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option(
        "--version", callback=_version, is_eager=True, help="Show the version and exit.")] = False,
) -> None:
    pass


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------

def convert(
    input: Annotated[str, typer.Argument(help="Image file, or '-' to read from stdin.")],
    output: Annotated[Optional[str], typer.Argument(
        help="SVG file, or '-' for stdout. Default: next to the input (stdout for stdin).")] = None,
    preset: PresetOpt = None,
    mode: ModeOpt = None,
    auto: AutoOpt = False,
    target: TargetOpt = None,
    max_edge: MaxEdgeOpt = None,
    threshold: ThresholdOpt = None,
    adaptive: AdaptiveOpt = None,
    invert: InvertOpt = None,
    blur: BlurOpt = None,
    upscale: UpscaleOpt = None,
    crop: CropOpt = None,
    padding: PaddingOpt = None,
    color: ColorOpt = None,
    background: BackgroundOpt = None,
    outline: OutlineOpt = None,
    stroke_width: StrokeOpt = None,
    turdsize: TurdOpt = None,
    alphamax: AlphaOpt = None,
    opttolerance: OptTolOpt = None,
    colors: ColorsOpt = None,
    filter_speckle: SpeckleOpt = None,
    layer_difference: LayerOpt = None,
    trace_mode: TraceModeOpt = None,
    optimize: OptimizeOpt = None,
    preview: PreviewOpt = False,
    json_out: JsonOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Vectorize one image."""
    opts = _options(preset, locals())

    if input == "-":
        source: str | bytes = sys.stdin.buffer.read()
        if not source:
            raise typer.BadParameter("No data on stdin", param_hint="INPUT")
        out = output or "-"
    else:
        if not Path(input).is_file():
            raise typer.BadParameter(f"File not found: {input}", param_hint="INPUT")
        source = input
        out = output or str(Path(input).with_suffix(".svg"))

    try:
        result = vectorize(source, opts, auto=auto, target=target, max_edge=max_edge)
    except ValueError as exc:
        _err(f"Error: {exc}")
        raise typer.Exit(1)

    to_stdout = out == "-"
    if to_stdout:
        if not json_out:
            sys.stdout.write(result.svg + "\n")
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(result.svg, encoding="utf-8")

    preview_path = None
    if preview and not to_stdout:
        preview_path = Path(out).with_suffix(".preview.png")
        write_preview(result.svg, preview_path)

    if json_out:
        data = result.to_dict(include_svg=to_stdout)
        data["input"] = input
        data["output"] = None if to_stdout else out
        data["preview"] = str(preview_path) if preview_path else None
        typer.echo(json.dumps(data, indent=2))
    elif not quiet:
        for w in result.warnings:
            _err(f"warning: {w}")
        _err(f"{'stdout' if to_stdout else out}: {_summary(result)}")
        if preview_path:
            _err(f"preview: {preview_path}")


app.command("convert")(convert)


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def _batch_one(src: str, dst: str, opts: Options, auto: bool, target: float | None,
               max_edge: float | None, preview: bool) -> dict[str, Any]:
    try:
        result = vectorize(src, opts, auto=auto, target=target, max_edge=max_edge)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text(result.svg, encoding="utf-8")
        if preview:
            write_preview(result.svg, Path(dst).with_suffix(".preview.png"))
        d = result.to_dict()
        d.pop("options")
        return {"input": src, "output": dst, "ok": True, **d}
    except Exception as exc:  # keep going; report at the end
        return {"input": src, "output": dst, "ok": False, "error": str(exc)}


@app.command()
def batch(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Folder of images.")],
    output_dir: Annotated[Path, typer.Argument(file_okay=False, help="Where SVGs are written.")],
    ext: Annotated[str, typer.Option(help="Comma-separated extensions to include.")] = DEFAULT_EXTS,
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Include subfolders (structure is kept).")] = False,
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=0, help="Parallel workers (0 = one per CPU).")] = 0,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Don't redo SVGs that already exist.")] = False,
    preset: PresetOpt = None,
    mode: ModeOpt = None,
    auto: AutoOpt = False,
    target: TargetOpt = None,
    max_edge: MaxEdgeOpt = None,
    threshold: ThresholdOpt = None,
    adaptive: AdaptiveOpt = None,
    invert: InvertOpt = None,
    blur: BlurOpt = None,
    upscale: UpscaleOpt = None,
    crop: CropOpt = None,
    padding: PaddingOpt = None,
    color: ColorOpt = None,
    background: BackgroundOpt = None,
    outline: OutlineOpt = None,
    stroke_width: StrokeOpt = None,
    turdsize: TurdOpt = None,
    alphamax: AlphaOpt = None,
    opttolerance: OptTolOpt = None,
    colors: ColorsOpt = None,
    filter_speckle: SpeckleOpt = None,
    layer_difference: LayerOpt = None,
    trace_mode: TraceModeOpt = None,
    optimize: OptimizeOpt = None,
    preview: PreviewOpt = False,
    json_out: JsonOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Vectorize every image in a folder. Accepts all `convert` options."""
    opts = _options(preset, locals())
    exts = {e.strip().lower().lstrip(".") for e in ext.split(",") if e.strip()}
    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower().lstrip(".") in exts
        and output_dir.resolve() not in f.resolve().parents
    )
    if not files:
        _err(f"No {', '.join(sorted(exts))} files in {input_dir}")
        raise typer.Exit(1)

    tasks = []
    skipped = 0
    for f in files:
        dst = output_dir / f.relative_to(input_dir).with_suffix(".svg")
        if skip_existing and dst.exists():
            skipped += 1
            continue
        tasks.append((str(f), str(dst)))

    workers = jobs or os.cpu_count() or 1
    workers = max(1, min(workers, len(tasks) or 1))
    results: list[dict[str, Any]] = []

    def report(r: dict[str, Any]) -> None:
        results.append(r)
        if quiet or json_out:
            if not r["ok"] and not json_out:
                _err(f"FAIL {r['input']}: {r['error']}")
            return
        if r["ok"]:
            _err(f"ok   {r['input']} -> {r['output']} "
                 f"({r['bytes'] / 1024:.1f} KB, fidelity {r['fidelity'] * 100:.1f}%)")
        else:
            _err(f"FAIL {r['input']}: {r['error']}")

    if workers == 1:
        for src, dst in tasks:
            report(_batch_one(src, dst, opts, auto, target, max_edge, preview))
    else:
        ctx = multiprocessing.get_context("spawn")  # fork + OpenCV threads can deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_batch_one, s, d, opts, auto, target, max_edge, preview) for s, d in tasks]
            for fut in as_completed(futures):
                report(fut.result())

    results.sort(key=lambda r: r["input"])
    failed = [r for r in results if not r["ok"]]
    if json_out:
        typer.echo(json.dumps({
            "converted": len(results) - len(failed), "failed": len(failed),
            "skipped": skipped, "files": results,
        }, indent=2))
    elif not quiet:
        _err(f"\n{len(results) - len(failed)} converted, {len(failed)} failed, {skipped} skipped")
    if failed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# presets / mcp
# --------------------------------------------------------------------------

@app.command()
def presets(json_out: JsonOpt = False) -> None:
    """List the built-in presets."""
    if json_out:
        typer.echo(json.dumps({p.name: {"summary": p.summary, "options": p.options}
                               for p in PRESETS.values()}, indent=2))
        return
    width = max(len(n) for n in PRESETS)
    for p in PRESETS.values():
        typer.echo(f"{p.name:<{width}}  {p.summary}")


@app.command()
def mcp() -> None:
    """Run vecto as an MCP server (stdio) so AI assistants can call it."""
    try:
        from .mcp_server import run
    except ImportError:
        _err("The MCP server needs the optional dependency: pip install 'vecto[mcp]'")
        raise typer.Exit(1)
    run()


if __name__ == "__main__":
    app()
