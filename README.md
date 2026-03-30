# vecto

CLI tool that converts raster images (PNG, JPG, etc.) into clean, optimized SVG vector graphics. It uses Potrace for bitmap tracing and Scour for SVG optimization, with OpenCV handling image preprocessing (grayscale, denoising, thresholding).

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Potrace](http://potrace.sourceforge.net/) (system dependency)

### Install Potrace

```bash
# Debian / Ubuntu
sudo apt install potrace
```

## Installation

```bash
uv sync
```

## Usage

### Vectorize a single image

```bash
vecto single input.png output.svg
```

With options:

```bash
vecto single input.png output.svg --threshold 128 --invert --preview
```

### Batch vectorize a directory

```bash
vecto batch ./images ./output
```

Process only JPEG files with preview:

```bash
vecto batch ./images ./output --ext jpg --preview
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--threshold` | auto (Otsu) | Binarization threshold (0-255) |
| `--invert` | `False` | Invert colors before tracing |
| `--turdsize` | `5` | Suppress speckles up to this size |
| `--alphamax` | `0.9` | Corner threshold (0 = sharp, 1.334 = smooth) |
| `--opttolerance` | `0.2` | Curve optimization tolerance |
| `--preview` | `False` | Generate a PNG preview of the result |

## Pipeline

1. **Preprocess** - grayscale conversion, Gaussian blur, threshold binarization (Otsu or manual)
2. **Trace** - Potrace converts the binary bitmap to SVG paths
3. **Optimize** - Scour strips metadata, enables viewboxing, cleans up the SVG
4. **Preview** _(optional)_ - CairoSVG renders a 2x PNG for quick visual check
