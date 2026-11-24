"""vecto: turn raster images into clean SVG vectors."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vecto-svg")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

from .core import Result, vectorize  # noqa: E402
from .options import PRESETS, Options, build_options  # noqa: E402

__all__ = ["Options", "PRESETS", "Result", "build_options", "vectorize", "__version__"]
