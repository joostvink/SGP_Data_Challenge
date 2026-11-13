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


def _distance_to_polygon_edges(pts, poly):
    best = np.full(len(pts), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        best = np.minimum(best, np.linalg.norm(pts - geo.closest_point_on_segment(pts, a, b), axis=1))
    return best


def min_manoeuvres(start, target, twd, tws, polar: Polar, boundary=None, margin=0.0, n=120) -> dict:
    """Fewest tacks (or gybes) needed to sail from ``start`` to ``target``.

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
       programme over an ``n`` x ``n`` grid in those coordinates. This handles
       boundaries of any shape, not just 'sail to the layline'.

    Positions are (east, north) metres. Returns a dict with ``mode``
    ("beat", "run" or "fetch"), ``manoeuvres`` (None when no route fits inside
    the boundary), ``twa`` sailed, and the route as ``path`` (k, 2).
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    tg = polar.targets(tws)
    direct_twa = abs(float(geo.wrap180(twd - geo.bearing_of(target - start))))
    if tg["up_twa"] <= direct_twa <= tg["down_twa"]:
        return {"mode": "fetch", "manoeuvres": 0, "twa": direct_twa, "path": np.stack([start, target])}

    mode, angle = ("beat", tg["up_twa"]) if direct_twa < tg["up_twa"] else ("run", tg["down_twa"])
    # heading = TWD + |TWA| has the wind on the port side, TWD - |TWA| on starboard
    e_port, e_stbd = geo.unit_from_bearing(twd + angle), geo.unit_from_bearing(twd - angle)
    a_max, b_max = np.linalg.solve(np.stack([e_port, e_stbd], axis=1), target - start)

    a = np.linspace(0.0, a_max, n)
    b = np.linspace(0.0, b_max, n)
    pts = start + a[:, None, None] * e_port + b[None, :, None] * e_stbd
    ok = np.ones((n, n), dtype=bool)
    if boundary is not None:
        flat = pts.reshape(-1, 2)
        inside = geo.points_in_polygon(flat, boundary)
        if margin > 0:
            inside &= _distance_to_polygon_edges(flat, boundary) >= margin
        ok = inside.reshape(n, n)
        ok[0, 0] = ok[-1, -1] = True  # the marks themselves are always reachable

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
            if i > 0:
                same, other = turns[i - 1, j, 0], turns[i - 1, j, 1] + 1
                turns[i, j, 0], came[i, j, 0] = (same, 0) if same <= other else (other, 1)
            if j > 0:
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


def course_min_manoeuvres(course: Course, twd, tws, polar: Polar, margin=0.0, n=120) -> pd.DataFrame:
    """Minimum manoeuvres for every leg, from every mark of the previous waypoint to every mark of the next.

    For a gate each of its marks is a separate target (and a separate start
    for the following leg), because the cheaper side of a gate is exactly
    what a tactician wants to know.
    """
    rows = []
    for leg in range(1, course.n_legs + 1):
        prev, nxt = course.waypoints[leg - 1], course.waypoints[leg]
        starts = [("mid " + prev.name[:2], prev.centre)] if leg == 1 else list(zip(prev.mark_names, prev.xy))
        for s_name, s_xy in starts:
            for t_name, t_xy in zip(nxt.mark_names, nxt.xy):
                res = min_manoeuvres(s_xy, t_xy, twd, tws, polar, course.boundary, margin, n)
                rows.append({"leg": leg, "from": s_name, "to": t_name, "mode": res["mode"], "manoeuvres": res["manoeuvres"], "path": res["path"]})
    return pd.DataFrame(rows)
