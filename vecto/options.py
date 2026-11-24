"""Conversion options and presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

Mode = Literal["mono", "color"]
TraceMode = Literal["spline", "polygon", "none"]


@dataclass(frozen=True)
class Options:
    """Every knob that controls a conversion.

    ``None`` on an "auto" field means vecto decides from the image.
    """

    # --- general -----------------------------------------------------------
    mode: Mode = "mono"
    upscale: float | None = None  # None = auto (small images are enlarged)
    crop: bool = False  # trim empty margins around the content
    padding: int = 0  # margin (in output px) kept around cropped content
    optimize: bool = True  # run Scour over the result
    precision: int = 5  # significant digits kept by Scour

    # --- mono (potrace) ----------------------------------------------------
    threshold: int | None = None  # None = Otsu
    adaptive: bool = False  # local threshold, for uneven lighting (scans)
    invert: bool | None = None  # None = detect a dark background
    blur: int = 3  # Gaussian kernel size before thresholding, 0 disables
    turdsize: int = 5  # drop specks up to this many (output) pixels in area
    alphamax: float = 0.9  # 0 = sharp corners, 1.334 = round everything
    opttolerance: float = 0.2  # curve merging tolerance
    color: str = "auto"  # fill colour; "auto" = the ink colour of the input
    background: str | None = None  # optional background fill
    outline: bool = False  # stroke shapes instead of filling (laser cutters)
    stroke_width: float = 1.0

    # --- color (vtracer) ---------------------------------------------------
    colors: int = 6  # color precision in bits per channel (1-8)
    filter_speckle: int = 4
    layer_difference: int = 16
    corner_threshold: int = 60
    trace_mode: TraceMode = "spline"

    def merged(self, **overrides: Any) -> "Options":
        """Return a copy with every non-None override applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"Unknown option(s): {', '.join(sorted(unknown))}")
        return replace(self, **clean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Settings --auto searches. Speck filters (turdsize, filter_speckle) are left
# alone on purpose: removing noise always "costs" fidelity, so a fidelity-driven
# search would happily keep every speck. They stay a preset / user decision.
MONO_TUNABLE = ("alphamax", "opttolerance")
COLOR_TUNABLE = ("colors", "layer_difference")


@dataclass(frozen=True)
class Preset:
    name: str
    summary: str
    options: dict[str, Any]


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in [
        Preset(
            "logo",
            "Flat logos and wordmarks. Keeps the ink colour, smooth curves, tight crop.",
            dict(mode="mono", turdsize=2, alphamax=1.0, opttolerance=0.2, crop=True, padding=2),
        ),
        Preset(
            "icon",
            "Small icons and glyphs. Upscales aggressively before tracing.",
            dict(mode="mono", upscale=8, turdsize=2, alphamax=1.0, opttolerance=0.1, blur=0),
        ),
        Preset(
            "signature",
            "Scanned or photographed signatures and handwriting. Handles uneven paper, "
            "removes dust, transparent background.",
            dict(
                mode="mono", adaptive=True, blur=5, turdsize=15, alphamax=1.1,
                opttolerance=0.3, crop=True, padding=4,
            ),
        ),
        Preset(
            "lineart",
            "Ink drawings, sketches and comics in black and white.",
            dict(mode="mono", turdsize=4, alphamax=1.0, opttolerance=0.2, color="black"),
        ),
        Preset(
            "cut",
            "Cutting machines (Cricut, Silhouette, laser, plotter). One colour, no specks, "
            "simplified smooth paths.",
            dict(
                mode="mono", color="black", turdsize=25, alphamax=1.2, opttolerance=0.6,
                crop=True, padding=2,
            ),
        ),
        Preset(
            "color",
            "Colour illustrations, flat artwork and stickers.",
            dict(mode="color", colors=6, filter_speckle=4, layer_difference=16),
        ),
        Preset(
            "photo",
            "Photographs (posterised look). Fewer colour layers to keep files reasonable.",
            dict(mode="color", colors=5, filter_speckle=8, layer_difference=32),
        ),
        Preset(
            "pixel",
            "Pixel art. Keeps hard square pixels, no smoothing or upscaling.",
            dict(mode="color", trace_mode="none", filter_speckle=0, colors=8,
                 layer_difference=1, upscale=1),
        ),
    ]
}


def build_options(preset: str | None = None, **overrides: Any) -> Options:
    """Start from defaults (or a preset) and apply explicit overrides."""
    base = Options()
    if preset:
        key = preset.lower()
        if key not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}")
        base = base.merged(**PRESETS[key].options)
    return base.merged(**overrides)
