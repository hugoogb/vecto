import subprocess
import tempfile
from pathlib import Path

import typer
from PIL import Image
import cv2
import numpy as np
import cairosvg

app = typer.Typer(help="Vectorize images into clean SVGs using Potrace")

# -------------------------
# Preprocessing
# -------------------------

def load_grayscale(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img

def apply_threshold(img, threshold: int | None):
    if threshold is None:
        # Otsu's method
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def preprocess(input_path: Path, output_path: Path, threshold: int | None, invert: bool):
    img = load_grayscale(input_path)

    # slight blur to remove noise (important)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    binary = apply_threshold(img, threshold)

    if invert:
        binary = cv2.bitwise_not(binary)

    Image.fromarray(binary).save(output_path)


# -------------------------
# Potrace + Optimization
# -------------------------

def run_potrace(pbm: Path, svg: Path, turdsize: int, alphamax: float, opt: float):
    subprocess.run([
        "potrace",
        str(pbm),
        "-s",
        "-o", str(svg),
        "--turdsize", str(turdsize),
        "--alphamax", str(alphamax),
        "--opttolerance", str(opt),
    ], check=True)

def optimize(svg: Path):
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run([
            "scour",
            "-i", str(svg),
            "-o", str(tmp_path),
            "--enable-viewboxing",
            "--remove-metadata",
            "--strip-xml-prolog",
        ], check=True)
        tmp_path.replace(svg)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# -------------------------
# Preview (SVG → PNG)
# -------------------------

def generate_preview(svg_path: Path, png_path: Path, scale: int = 2):
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        scale=scale
    )


# -------------------------
# Core pipeline
# -------------------------

def process(
    input_path: Path,
    output_path: Path,
    threshold: int | None,
    invert: bool,
    turdsize: int,
    alphamax: float,
    opt: float,
    preview: bool
):
    with tempfile.TemporaryDirectory() as tmp:
        pbm = Path(tmp) / "temp.pbm"

        preprocess(input_path, pbm, threshold, invert)
        run_potrace(pbm, output_path, turdsize, alphamax, opt)
        optimize(output_path)

        if preview:
            preview_path = output_path.with_suffix(".preview.png")
            generate_preview(output_path, preview_path)
            typer.echo(f"🖼 Preview: {preview_path}")


# -------------------------
# CLI Commands
# -------------------------

@app.command()
def single(
    input: Path,
    output: Path,
    threshold: int = typer.Option(None, help="Threshold (0-255). Default: auto (Otsu)"),
    invert: bool = typer.Option(False, help="Invert colors"),
    turdsize: int = 5,
    alphamax: float = 0.9,
    opttolerance: float = 0.2,
    preview: bool = False,
):
    """Vectorize a single image"""

    process(
        input,
        output,
        threshold,
        invert,
        turdsize,
        alphamax,
        opttolerance,
        preview
    )

    typer.echo(f"✅ SVG: {output}")


@app.command()
def batch(
    input_dir: Path,
    output_dir: Path,
    ext: str = "png",
    preview: bool = False,
):
    """Batch vectorization"""

    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob(f"*.{ext}"):
        out = output_dir / f"{file.stem}.svg"

        process(
            file,
            out,
            threshold=None,
            invert=False,
            turdsize=5,
            alphamax=0.9,
            opt=0.2,
            preview=preview
        )

        typer.echo(f"✔ {file.name}")


if __name__ == "__main__":
    app()