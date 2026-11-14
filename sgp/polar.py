"""Polar lookups and the minimum number of manoeuvres on a leg."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from . import geo
from .course import Course


class Polar:
    """Boat speed as a function of true wind angle and true wind speed (km/h in, km/h out)."""

    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.twa = table.index.to_numpy(float)
        self.tws = table.columns.to_numpy(float)

    def curve(self, tws: float) -> PchipInterpolator:
        """Boat speed vs TWA at one wind speed (linear between wind speed columns, monotone spline over angle)."""
        tws = float(np.clip(tws, self.tws[0], self.tws[-1]))
        speeds = np.array([np.interp(tws, self.tws, row) for row in self.table.to_numpy(float)])
        return PchipInterpolator(self.twa, speeds)

    def speed(self, twa, tws: float):
        return self.curve(tws)(np.abs(geo.wrap180(twa)))

    def targets(self, tws: float) -> dict:
        """Best-VMG angles and speeds upwind and downwind at one wind speed."""
        curve = self.curve(tws)
        up = np.arange(max(self.twa[0], 20.0), 90.0, 0.1)
        dn = np.arange(90.0, self.twa[-1] + 0.05, 0.1)
        vmg_up = curve(up) * np.cos(np.radians(up))
        vmg_dn = curve(dn) * np.cos(np.radians(dn))
        i, j = int(np.argmax(vmg_up)), int(np.argmin(vmg_dn))
        return {
            "tws_kmh": float(tws),
            "up_twa": float(up[i]),
            "up_bsp": float(curve(up[i])),
            "up_vmg": float(vmg_up[i]),
            "down_twa": float(dn[j]),
            "down_bsp": float(curve(dn[j])),
            "down_vmg": float(vmg_dn[j]),
        }


def min_manoeuvres(start, target, twd, tws, polar: Polar, boundary=None, margin=0.0, n=120) -> dict:
    """Fewest manoeuvres on a finite grid of the two best-VMG headings.

    Method
    ------
    1. The polar gives the best-VMG angles for this wind speed. If the direct
       course is at or between them the leg is a fetch: 0 manoeuvres.
    2. Otherwise the boat sails on its two best-VMG headings only. Any such
       route is a staircase in the skewed coordinate system spanned by those
       two headings, running from (0, 0) to the target, and each corner of the
       staircase is one manoeuvre.
    3. The staircase with the fewest corners that stays inside the course
       boundary (shrunk by ``margin`` metres) is found with a dynamic
       programme over an ``n`` x ``n`` grid. Complete edges must remain inside
       the boundary and meet the margin. This is a restricted grid optimum,
       not a continuous minimum over all positive-speed polar headings.

    Positions are (east, north) metres. Returns a dict with ``mode``
    ("beat", "run" or "fetch"), ``manoeuvres`` (None when no route fits inside
    the boundary), ``twa`` sailed, and the route as ``path`` (k, 2).
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if n < 2 or margin < 0 or not np.isfinite([*start, *target, twd, tws]).all():
        raise ValueError("Finite inputs, n >= 2 and a nonnegative margin are required")
    tg = polar.targets(tws)
    direct_twa = abs(float(geo.wrap180(twd - geo.bearing_of(target - start))))
    if tg["up_twa"] <= direct_twa <= tg["down_twa"]:
        feasible = boundary is None or bool(geo.segments_in_polygon(start, target, boundary, margin))
        return {"mode": "fetch", "manoeuvres": 0 if feasible else None,
                "twa": direct_twa, "path": np.stack([start, target]) if feasible else None}

    mode, angle = ("beat", tg["up_twa"]) if direct_twa < tg["up_twa"] else ("run", tg["down_twa"])
    # heading = TWD + |TWA| has the wind on the port side, TWD - |TWA| on starboard
    e_port, e_stbd = geo.unit_from_bearing(twd + angle), geo.unit_from_bearing(twd - angle)
    a_max, b_max = np.linalg.solve(np.stack([e_port, e_stbd], axis=1), target - start)

    a = np.linspace(0.0, a_max, n)
    b = np.linspace(0.0, b_max, n)
    pts = start + a[:, None, None] * e_port + b[None, :, None] * e_stbd
    ok = np.ones((n, n), dtype=bool)
    edge_a = np.ones((n - 1, n), bool)
    edge_b = np.ones((n, n - 1), bool)
    if boundary is not None:
        edge_a = geo.segments_in_polygon(pts[:-1], pts[1:], boundary, margin)
        edge_b = geo.segments_in_polygon(pts[:, :-1], pts[:, 1:], boundary, margin)
        ok = geo.segments_in_polygon(pts, pts, boundary, margin)
    if not ok[0, 0] or not ok[-1, -1]:
        return {"mode": mode, "manoeuvres": None, "twa": angle, "path": None}

    # turns[i, j, d]: fewest manoeuvres to reach node (i, j) arriving on board d (0 = port, 1 = starboard)
    # came[i, j, d]: the board sailed into the previous node, to walk the route back
    inf = np.iinfo(np.int32).max // 2
    turns = np.full((n, n, 2), inf, dtype=np.int32)
    came = np.zeros((n, n, 2), dtype=np.int8)
    turns[0, 0] = 0
    came[0, 0] = (0, 1)
    for i in range(n):
        for j in range(n):
            if not ok[i, j] or (i == 0 and j == 0):
                continue
            if i > 0 and edge_a[i - 1, j]:
                same, other = turns[i - 1, j, 0], turns[i - 1, j, 1] + 1
                turns[i, j, 0], came[i, j, 0] = (same, 0) if same <= other else (other, 1)
            if j > 0 and edge_b[i, j - 1]:
                same, other = turns[i, j - 1, 1], turns[i, j - 1, 0] + 1
                turns[i, j, 1], came[i, j, 1] = (same, 1) if same <= other else (other, 0)
    best = int(turns[-1, -1].min())
    if best >= inf:
        return {"mode": mode, "manoeuvres": None, "twa": angle, "path": None}

    i, j, d = n - 1, n - 1, int(np.argmin(turns[-1, -1]))
    corners = [pts[i, j]]
    while (i, j) != (0, 0):
        prev_d = int(came[i, j, d])
        i, j = (i - 1, j) if d == 0 else (i, j - 1)
        if prev_d != d and (i, j) != (0, 0):
            corners.append(pts[i, j])
        d = prev_d
    corners.append(pts[0, 0])
    return {"mode": mode, "manoeuvres": best, "twa": angle, "path": np.array(corners[::-1])}


def line_min_manoeuvres(start, a, b, twd, tws, polar: Polar, boundary=None, margin=0.0, n=120, samples=21) -> dict:
    """Fewest manoeuvres to any sampled point of the line a-b.

    A finish line is crossed, not rounded, so every point of it is a valid
    target. ``samples`` evenly spaced points are tried, kept ``margin`` metres
    inside both ends; the route with the fewest manoeuvres wins and ties go to
    the shortest one. The result of ``min_manoeuvres`` is returned with the
    chosen ``target`` added (the line midpoint when no sampled point is
    feasible).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    inset = min(margin, length / 2) / length if length > 0 else 0.5
    best, best_key = None, None
    for f in np.linspace(inset, 1.0 - inset, samples):
        target = a + f * (b - a)
        res = min_manoeuvres(start, target, twd, tws, polar, boundary, margin, n)
        if res["manoeuvres"] is None:
            continue
        key = (res["manoeuvres"], float(np.linalg.norm(np.diff(res["path"], axis=0), axis=1).sum()))
        if best_key is None or key < best_key:
            best, best_key = {**res, "target": target}, key
    if best is None:
        middle = (a + b) / 2
        best = {**min_manoeuvres(start, middle, twd, tws, polar, boundary, margin, n), "target": middle}
    return best


def course_min_manoeuvres(course: Course, twd, tws, polar: Polar, margin=0.0, n=120) -> pd.DataFrame:
    """Minimum manoeuvres for every leg, from every mark of the previous waypoint to every mark of the next.

    For a gate each of its marks is a separate target (and a separate start
    for the following leg), because the cheaper side of a gate is exactly
    what a tactician wants to know. The finish line is one target: any point
    of it at least ``margin`` metres inside its ends (``line_min_manoeuvres``).
    """
    rows = []
    for leg in range(1, course.n_legs + 1):
        prev, nxt = course.waypoints[leg - 1], course.waypoints[leg]
        starts = [("mid " + prev.name[:2], prev.centre)] if leg == 1 else list(zip(prev.mark_names, prev.xy))
        for s_name, s_xy in starts:
            if leg == course.n_legs and nxt.is_gate:
                results = [(nxt.name[:2] + " line", line_min_manoeuvres(s_xy, nxt.xy[0], nxt.xy[1], twd, tws, polar, course.boundary, margin, n))]
            else:
                results = [(t_name, min_manoeuvres(s_xy, t_xy, twd, tws, polar, course.boundary, margin, n)) for t_name, t_xy in zip(nxt.mark_names, nxt.xy)]
            for t_name, res in results:
                rows.append({"leg": leg, "from": s_name, "to": t_name, "mode": res["mode"], "manoeuvres": res["manoeuvres"], "path": res["path"]})
    return pd.DataFrame(rows)
