"""Angles, compass statistics and a local flat-earth frame.

The race area is about 2 km across, so a local tangent plane (metres east /
metres north of a reference point) is accurate to well under a metre and keeps
all geometry in plain 2-D vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8
KMH_PER_MS = 3.6


# --------------------------------------------------------------------------
# angles
# --------------------------------------------------------------------------
def wrap360(deg):
    """Map any angle to [0, 360)."""
    return np.mod(deg, 360.0)


def wrap180(deg):
    """Map any angle to [-180, 180)."""
    return np.mod(np.asarray(deg, dtype=float) + 180.0, 360.0) - 180.0


def circular_mean(deg, weights=None):
    """Mean of compass directions and the mean resultant length R.

    Each direction is treated as a unit vector (or a vector of length
    ``weights``); the mean direction is the direction of the vector sum.
    R is 1 when all samples agree and 0 when they cancel out, so it doubles
    as a quality flag for the mean.
    """
    deg = np.asarray(deg, dtype=float)
    w = np.ones_like(deg) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(deg) & np.isfinite(w)
    if not ok.any() or w[ok].sum() == 0:
        return np.nan, np.nan
    rad = np.radians(deg[ok])
    s = np.sum(w[ok] * np.sin(rad))
    c = np.sum(w[ok] * np.cos(rad))
    r = np.hypot(s, c) / np.sum(np.abs(w[ok]))
    return float(wrap360(np.degrees(np.arctan2(s, c)))), float(r)


def resample_direction(series, rule="10s", weights=None, min_count=1, min_r=0.0):
    """Downsample a compass direction to a lower frequency without wrap errors.

    Parameters
    ----------
    series : pd.Series of degrees with a DatetimeIndex (any sample rate, gaps allowed).
    rule : pandas offset alias of the output bins, e.g. ``"10s"``.
    weights : optional pd.Series on the same index (e.g. wind speed) to get a
        vector average instead of a unit-vector average.
    min_count : bins with fewer valid samples become NaN.
    min_r : bins whose resultant length is below this become NaN (the
        direction was not stable enough for a mean to be meaningful).

    Returns
    -------
    DataFrame with ``mean`` (deg, 0-360), ``r`` (0-1), ``std`` (circular
    standard deviation, deg) and ``n`` (samples in the bin). Bins are
    left-closed and labelled by their start time.
    """
    s = pd.Series(series, dtype=float)
    w = pd.Series(1.0, index=s.index) if weights is None else pd.Series(weights, dtype=float).reindex(s.index)
    ok = s.notna() & w.notna()
    rad = np.radians(s.where(ok))
    parts = pd.DataFrame(
        {"sin": w * np.sin(rad), "cos": w * np.cos(rad), "w": w.abs().where(ok), "n": ok.astype(float)}
    ).resample(rule).sum(min_count=1)
    r = np.hypot(parts["sin"], parts["cos"]) / parts["w"]
    out = pd.DataFrame(
        {
            "mean": wrap360(np.degrees(np.arctan2(parts["sin"], parts["cos"]))),
            "r": r,
            "std": np.degrees(np.sqrt(-2.0 * np.log(r.clip(lower=1e-12, upper=1.0)))),
            "n": parts["n"].fillna(0).astype(int),
        }
    )
    bad = (out["n"] < min_count) | ~(out["r"] >= min_r)
    out.loc[bad, ["mean", "std"]] = np.nan
    return out


# --------------------------------------------------------------------------
# local frame
# --------------------------------------------------------------------------
class LocalFrame:
    """Equirectangular projection around a reference point (x east, y north, metres)."""

    def __init__(self, lat0, lon0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._kx = np.radians(1.0) * EARTH_RADIUS_M * np.cos(np.radians(self.lat0))
        self._ky = np.radians(1.0) * EARTH_RADIUS_M

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return np.stack([(lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky], axis=-1)

    def to_latlon(self, xy):
        xy = np.asarray(xy, dtype=float)
        return self.lat0 + xy[..., 1] / self._ky, self.lon0 + xy[..., 0] / self._kx


def bearing_of(vec):
    """Compass bearing (deg, 0 = north, 90 = east) of an (east, north) vector."""
    vec = np.asarray(vec, dtype=float)
    return wrap360(np.degrees(np.arctan2(vec[..., 0], vec[..., 1])))


def unit_from_bearing(deg):
    """(east, north) unit vector pointing along a compass bearing."""
    rad = np.radians(deg)
    return np.stack([np.sin(rad), np.cos(rad)], axis=-1)


def unit(vec):
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.divide(vec, n, out=np.zeros_like(vec), where=n > 0)


def cross2(a, b):
    """z-component of the 2-D cross product."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def closest_point_on_segment(p, a, b):
    """Closest point to ``p`` (N,2) on the segment a-b."""
    p = np.asarray(p, dtype=float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return a + t[..., None] * ab


def segment_hits(p0, p1, a, b):
    """Where does each step p0[i] -> p1[i] cross the segment a-b?

    Returns (hit, t, s): ``t`` is the fraction along the step, ``s`` the
    fraction along a-b (0 at a, 1 at b), both NaN where there is no crossing.
    """
    p0 = np.asarray(p0, dtype=float)
    d = np.asarray(p1, dtype=float) - p0
    a = np.asarray(a, dtype=float)  # one segment, or one per step
    e = np.asarray(b, dtype=float) - a
    den = cross2(d, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross2(a - p0, e) / den
        s = cross2(a - p0, d) / den
    hit = np.isfinite(t) & (t >= 0) & (t < 1) & (s >= 0) & (s <= 1)
    return hit, np.where(hit, t, np.nan), np.where(hit, s, np.nan)


def points_in_polygon(pts, poly):
    """Even-odd test for points (N,2) against a polygon (M,2)."""
    pts = np.asarray(pts, dtype=float)
    x, y = pts[..., 0], pts[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        straddles = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddles & (x < x_cross)
    return inside
