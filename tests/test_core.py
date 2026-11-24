import re

import numpy as np
import pytest

from vecto import Options, build_options, vectorize
from vecto.image import is_antialiased, load_rgba, prepare_mono

from .conftest import ring


def svg_size(svg):
    w = float(re.search(r'width="([\d.]+)"', svg).group(1))
    h = float(re.search(r'height="([\d.]+)"', svg).group(1))
    return w, h


def test_mono_ring_is_faithful(ring_png):
    r = vectorize(ring_png)
    assert r.fidelity > 0.97
    assert r.edge_error is not None and r.edge_error < 1.0
    assert r.paths == 1
    assert 'fill-rule="evenodd"' in r.svg
    assert svg_size(r.svg) == (200, 200)


def test_transparent_white_logo_is_traced_in_white(transparent_white_logo):
    r = vectorize(transparent_white_logo)
    assert r.fidelity > 0.95
    assert r.inverted  # light content was flipped to become the ink
    assert 'fill="#fff' in r.svg.lower()


def test_dark_background_is_inverted_automatically():
    img = ring(fg=(255, 255, 255), bg=(0, 0, 0))
    r = vectorize(img)
    assert r.inverted
    assert r.fidelity > 0.97
    forced = vectorize(img, Options(invert=False))
    assert not forced.inverted


def test_small_antialiased_images_are_upscaled_but_aliased_ones_are_not():
    small_aa = ring(64, aa=True)
    small_hard = ring(64, aa=False)
    assert is_antialiased(small_aa[..., 0])
    assert not is_antialiased(small_hard[..., 0])
    assert vectorize(small_aa).scale > 1
    assert vectorize(small_hard).scale == 1
    # output keeps the original dimensions regardless of the working scale
    assert svg_size(vectorize(small_aa).svg) == (64, 64)


def test_crop_trims_margins(ring_png):
    full = vectorize(ring_png)
    cropped = vectorize(ring_png, Options(crop=True))
    assert cropped.width < full.width and cropped.height < full.height
    assert abs(cropped.width - 140) <= 3


def test_explicit_color_background_and_outline(ring_png):
    r = vectorize(ring_png, Options(color="#ff0000", background="#ffffff"))
    assert "#f00" in r.svg.lower() and "<rect" in r.svg
    o = vectorize(ring_png, Options(outline=True))
    assert 'fill="none"' in o.svg and "stroke=" in o.svg


def test_turdsize_removes_specks():
    img = ring()
    rng = np.random.default_rng(0)
    for y, x in rng.integers(5, 195, (40, 2)):
        if img[y, x, 0] == 255:
            img[y : y + 2, x : x + 2] = 0  # 2x2 dust specks on the paper
    clean = vectorize(img, Options(turdsize=10))
    dusty = vectorize(img, Options(turdsize=0))
    assert clean.svg.count("m") < dusty.svg.count("m")
    # speck removal is intentional, so it must not be scored as a loss
    assert clean.fidelity > 0.97


def test_auto_is_never_larger_and_meets_goals(ring_png):
    base = vectorize(ring_png, Options(alphamax=0.5, opttolerance=0.1))
    tuned = vectorize(ring_png, Options(alphamax=0.5, opttolerance=0.1), auto=True)
    assert tuned.candidates_tried > 1
    assert tuned.bytes <= base.bytes
    assert tuned.fidelity >= 0.95
    assert not tuned.warnings


def test_color_mode(color_png):
    r = vectorize(color_png, build_options("color"))
    assert r.colors >= 3
    assert r.fidelity > 0.98
    assert r.edge_error is None


def test_pixel_preset_keeps_hard_edges():
    sprite = np.zeros((8, 8, 4), np.uint8)
    sprite[2:6, 2:6] = (255, 0, 0, 255)
    sprite[3:5, 3:5] = (0, 0, 255, 255)
    r = vectorize(sprite, build_options("pixel"))
    assert r.scale == 1
    assert r.fidelity > 0.99


def test_presets_and_overrides():
    o = build_options("cut", turdsize=3)
    assert o.color == "black" and o.turdsize == 3 and o.crop
    with pytest.raises(ValueError):
        build_options("nope")
    with pytest.raises(ValueError):
        Options().merged(bogus=1)


def test_bad_input_raises(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot read image"):
        load_rgba(bad)


def test_blank_image_warns():
    r = vectorize(np.full((50, 50, 3), 255, np.uint8))
    assert any("No foreground" in w for w in r.warnings)
    assert "<path" not in r.svg


def test_adaptive_threshold_handles_uneven_lighting():
    img = ring(300).astype(np.float32)
    grad = np.linspace(0.35, 1.0, 300)[None, :, None]
    img = (img * grad).astype(np.uint8)  # strong shadow on the left
    assert vectorize(img, Options(adaptive=True, turdsize=20)).fidelity > 0.9


def test_prepare_mono_grayscale_ink_detection():
    img = ring(fg=(20, 60, 200))
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    assert prepare_mono(rgba, Options()).ink != "#000000"
