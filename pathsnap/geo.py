"""Geometry helpers.

Coordinates are projected to UTM (metres) before any distance is computed. UTM is conformal and its scale
error is below 0.1 % inside a zone, which is far smaller than GPS noise, so Euclidean distances in the
projected plane are valid at the scale of a single trip. Degrees are never used as distances.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    """lon/lat (degrees, x=lon) -> metres, in the UTM zone containing (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._t = Transformer.from_crs(4326, self.epsg, always_xy=True)

    def xy(self, lon, lat) -> np.ndarray:
        x, y = self._t.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return np.column_stack([x, y])


def point_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from points P to segments A->B and the clamped fractional position along each segment."""
    ab = B - A
    denom = np.maximum((ab * ab).sum(axis=1), 1e-9)
    frac = np.clip(((P - A) * ab).sum(axis=1) / denom, 0.0, 1.0)
    return np.linalg.norm(P - (A + frac[:, None] * ab), axis=1), frac
