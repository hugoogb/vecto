"""Generate the sample inputs used in the README (synthetic, reproducible).

    uv run python examples/make_examples.py
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
rng = np.random.default_rng(7)


def supersample(draw, w, h, channels, ss=8, bg=0):
    """Draw at ss x resolution and downsample, giving realistic anti-aliasing."""
    big = np.full((h * ss, w * ss, channels), bg, np.uint8)
    draw(big, ss)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def logo():
    # Small transparent logo (anti-aliased), the kind people grab from a website.
    def draw(img, s):
        navy = (32, 46, 120, 255)
        cv2.circle(img, (40 * s, 40 * s), 30 * s, navy, -1, cv2.LINE_AA)
        cv2.circle(img, (40 * s, 40 * s), 17 * s, (0, 0, 0, 0), -1, cv2.LINE_AA)
        pts = np.array([[28, 62], [40, 18], [52, 62]], np.int32) * s
        cv2.fillPoly(img, [pts], navy, cv2.LINE_AA)
        cv2.putText(img, "vecto", (78 * s, 54 * s), cv2.FONT_HERSHEY_DUPLEX, 1.25 * s,
                    navy, 3 * s, cv2.LINE_AA)
    rgba = supersample(draw, 200, 80, 4)
    Image.fromarray(rgba, "RGBA").save(HERE / "logo.png")


def signature():
    # Blue ink on off-white paper, photographed: uneven light, grain, JPEG.
    w, h = 600, 260

    def draw(img, s):
        t = np.linspace(0, 1, 400)
        x = 60 + 470 * t
        y = 140 + 45 * np.sin(t * 19) * np.exp(-t * 1.2) - 30 * np.sin(t * 5)
        pts = np.stack([x, y], 1) * s
        cv2.polylines(img, [pts.astype(np.int32)], False, (25, 40, 150), 4 * s, cv2.LINE_AA)
        loop = np.stack([140 + 40 * np.cos(np.linspace(0, 6.3, 120)),
                         120 + 70 * np.sin(np.linspace(0, 6.3, 120))], 1) * s
        cv2.polylines(img, [loop.astype(np.int32)], False, (25, 40, 150), 3 * s, cv2.LINE_AA)
        cv2.line(img, (90 * s, 205 * s), (520 * s, 195 * s), (25, 40, 150), 3 * s, cv2.LINE_AA)

    ink = supersample(draw, w, h, 3, bg=255).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.62 + 0.38 * np.clip(1 - ((xx - 150) ** 2 + (yy - 40) ** 2) / 420_000, 0, 1)
    paper = np.array([242, 236, 222], np.float32) * light[..., None]
    img = paper * (ink / 255.0) + rng.normal(0, 6, (h, w, 3))
    for _ in range(40):  # dust
        cv2.circle(img, (int(rng.integers(w)), int(rng.integers(h))), 1, (90, 90, 90), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(HERE / "signature.jpg", quality=80)


def art():
    # Flat colour illustration.
    def draw(img, s):
        img[:] = (170, 215, 245)
        cv2.circle(img, (230 * s, 70 * s), 38 * s, (250, 200, 60), -1, cv2.LINE_AA)
        cv2.ellipse(img, (90 * s, 260 * s), (190 * s, 110 * s), 0, 0, 360, (90, 170, 90), -1, cv2.LINE_AA)
        cv2.ellipse(img, (260 * s, 280 * s), (170 * s, 100 * s), 0, 0, 360, (60, 130, 70), -1, cv2.LINE_AA)
        cv2.rectangle(img, (70 * s, 150 * s), (84 * s, 200 * s), (110, 75, 45), -1)
        cv2.circle(img, (77 * s, 135 * s), 30 * s, (40, 110, 55), -1, cv2.LINE_AA)
    rgb = supersample(draw, 320, 240, 3)
    Image.fromarray(rgb).save(HERE / "art.png")


def pixel():
    # 16x16 sprite.
    palette = {".": (0, 0, 0, 0), "k": (30, 30, 40, 255), "r": (220, 50, 60, 255),
               "w": (255, 255, 255, 255), "p": (250, 150, 160, 255)}
    rows = [
        "................", "...kk......kk...", "..krrk....krrk..", ".krrrrk..krrrrk.",
        ".krwrrrkkrrrrrk.", ".krwrrrrrrrrrrk.", ".krrrrrrrrrrrrk.", ".krrrrrrrrrrprk.",
        "..krrrrrrrrrpk..", "...krrrrrrrrk...", "....krrrrrrk....", ".....krrrrk.....",
        "......krrk......", ".......kk.......", "................", "................",
    ]
    img = np.array([[palette[c] for c in row] for row in rows], np.uint8)
    Image.fromarray(img, "RGBA").save(HERE / "pixel.png")


if __name__ == "__main__":
    logo()
    signature()
    art()
    pixel()
    print("wrote", ", ".join(p.name for p in sorted(HERE.glob("*.*")) if p.suffix != ".py"))
