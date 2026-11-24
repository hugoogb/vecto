from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


def ring(size=200, aa=True, fg=(0, 0, 0), bg=(255, 255, 255)) -> np.ndarray:
    line = cv2.LINE_AA if aa else cv2.LINE_8
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = bg
    c, r = size // 2, int(size * 0.35)
    cv2.circle(img, (c, c), r, fg, -1, line)
    cv2.circle(img, (c, c), r // 2, bg, -1, line)
    return img


@pytest.fixture
def ring_png(tmp_path) -> Path:
    p = tmp_path / "ring.png"
    Image.fromarray(ring()).save(p)
    return p


@pytest.fixture
def transparent_white_logo(tmp_path) -> Path:
    rgba = np.zeros((96, 96, 4), np.uint8)
    cv2.circle(rgba, (48, 48), 30, (255, 255, 255, 255), -1, cv2.LINE_AA)
    p = tmp_path / "white.png"
    Image.fromarray(rgba, "RGBA").save(p)
    return p


@pytest.fixture
def color_png(tmp_path) -> Path:
    img = np.full((120, 160, 3), 255, np.uint8)
    cv2.circle(img, (50, 60), 35, (210, 40, 40), -1, cv2.LINE_AA)
    cv2.rectangle(img, (95, 20), (150, 100), (40, 110, 220), -1)
    p = tmp_path / "color.png"
    Image.fromarray(img).save(p)
    return p
