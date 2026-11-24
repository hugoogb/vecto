"""How closely does a traced SVG match its source?

* mono:  intersection-over-union of ink pixels (1.0 = identical shapes).
         Unlike plain pixel accuracy this is not inflated by empty background.
         Area scores barely notice rounded corners, so mono also reports
         edge error: how far outlines drift (99th percentile, output pixels).
* color: 1 - mean absolute RGB difference after flattening both images on white.
"""

from __future__ import annotations

import cv2
import numpy as np

from .svgtools import render_rgba


def _drop_small(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Remove connected components of ``mask`` with area <= ``max_area``."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    keep = stats[:, cv2.CC_STAT_AREA] > max_area
    keep[0] = False  # background label
    return keep[labels]


def despeckle(mask: np.ndarray, max_area: int) -> np.ndarray:
    """What the mask looks like once specks and pinholes up to ``max_area`` are gone.

    Speck removal is a deliberate choice, so fidelity is measured against this
    cleaned reference rather than the raw (noisy) threshold.
    """
    if max_area <= 0:
        return mask
    cleaned = _drop_small(mask, max_area)
    holes = _drop_small(~cleaned, max_area)
    return ~holes


def _edges(mask: np.ndarray) -> np.ndarray:
    m8 = mask.astype(np.uint8)
    return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))).astype(bool)


def _edge_error(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """99th-percentile distance between the outlines of two masks, in output px."""
    ea, eb = _edges(a), _edges(b)
    if not ea.any() or not eb.any():
        return 0.0 if ea.any() == eb.any() else float("inf")
    # distance from every pixel to the nearest edge pixel of the other mask
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.concatenate([db[ea], da[eb]])
    return float(np.percentile(d, 99) / scale)


def mono_metrics(svg: str, mask: np.ndarray, scale: float = 1.0) -> tuple[float, float]:
    """Return (IoU of ink pixels, edge error in output pixels)."""
    h, w = mask.shape
    rendered = render_rgba(svg, w, h)[..., 3] > 127
    union = np.logical_or(rendered, mask).sum()
    iou = 1.0 if union == 0 else float(np.logical_and(rendered, mask).sum() / union)
    return iou, _edge_error(rendered, mask, scale)


def _flatten(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgb * a + 255.0 * (1 - a)


def color_fidelity(svg: str, rgba: np.ndarray) -> float:
    h, w = rgba.shape[:2]
    rendered = render_rgba(svg, w, h)
    diff = np.abs(_flatten(rendered) - _flatten(rgba)).mean()
    return float(1.0 - diff / 255.0)
