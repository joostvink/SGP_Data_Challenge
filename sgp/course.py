"""Race course model: XML parsing, leg detection, progress, VMC and distance to leader.

Conventions
-----------
* Waypoint 0 is the start line, the last waypoint is the finish line.
* Leg ``k`` (1-based) runs from waypoint ``k-1`` to waypoint ``k``.
* A boat's leg number is 0 before it starts and ``n_legs + 1`` once it has finished.
* All geometry is done in a local metric frame (see :mod:`sgp.geo`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import geo


@dataclass(frozen=True)
class Waypoint:
    seq: int
    name: str
    rounding: str  # "SP"/"PS" for a gate or line, "Port"/"Stbd" for a single mark
    mark_names: tuple
    xy: np.ndarray  # (1, 2) for a mark, (2, 2) for a gate / line

    @property
    def is_gate(self) -> bool:
        return len(self.xy) == 2

    @property
    def centre(self) -> np.ndarray:
        return self.xy.mean(axis=0)


@dataclass(frozen=True)
class Course:
    race_id: str
    start_time: pd.Timestamp  # UTC, timezone-naive (same convention as the boat logs)
    frame: geo.LocalFrame
    waypoints: tuple
    limits: dict  # name -> (M, 2) polygon in the local frame
    mark_tracks: dict | None = None  # optional {mark name: (time_ns, xy)} of measured positions

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_xml(cls, path, frame=None) -> "Course":
        root = ET.parse(path).getroot()
        start = pd.Timestamp(root.find("RaceStartTime").attrib["Start"]).tz_convert("UTC").tz_localize(None)

        compounds = {}
        for cm in root.find("Course").findall("CompoundMark"):
            marks = [(m.attrib["Name"], float(m.attrib["TargetLat"]), float(m.attrib["TargetLng"])) for m in cm.findall("Mark")]
            compounds[cm.attrib["CompoundMarkID"]] = (cm.attrib["Name"], marks)

        if frame is None:
            lat = np.mean([m[1] for _, marks in compounds.values() for m in marks])
            lon = np.mean([m[2] for _, marks in compounds.values() for m in marks])
            frame = geo.LocalFrame(lat, lon)

        waypoints = []
        corners = sorted(root.find("CompoundMarkSequence").findall("Corner"), key=lambda c: int(c.attrib["SeqID"]))
        for corner in corners:
            name, marks = compounds[corner.attrib["CompoundMarkID"]]
            xy = frame.to_xy([m[1] for m in marks], [m[2] for m in marks])
            waypoints.append(Waypoint(len(waypoints), name, corner.attrib["Rounding"], tuple(m[0] for m in marks), xy))

        limits = {}
        for lim in root.findall("CourseLimit"):
            pts = sorted(lim.findall("Limit"), key=lambda p: int(p.attrib["SeqID"]))
            limits[lim.attrib["name"]] = frame.to_xy([float(p.attrib["Lat"]) for p in pts], [float(p.attrib["Lon"]) for p in pts])

        return cls(root.find("RaceID").text, start, frame, tuple(waypoints), limits)

    def with_marks(self, positions: dict) -> "Course":
        """Copy of the course with mark positions replaced by ``{name: (lat, lon)}``.

        ``positions`` may also be keyed by ``(waypoint_seq, name)`` to move a mark
        for one waypoint only (the same buoy can sit in a different place when it
        is used as start pin and later as something else).
        """
        new = []
        for wp in self.waypoints:
            xy = wp.xy.copy()
            for i, name in enumerate(wp.mark_names):
                pos = positions.get((wp.seq, name), positions.get(name))
                if pos is not None and np.all(np.isfinite(pos)):
                    xy[i] = self.frame.to_xy(pos[0], pos[1])
            new.append(replace(wp, xy=xy))
        return replace(self, waypoints=tuple(new))

    def with_mark_tracks(self, marks: pd.DataFrame, time="DATETIME") -> "Course":
        """Copy of the course that knows the measured position of every mark over time.

        Only :meth:`detect_legs` uses these: a mark is tested where it actually
        was at the moment a boat went past, not where the XML says it should be.
        """
        tracks = {}
        for name, g in marks.sort_values(time).groupby("MARK"):
            tracks[name] = (g[time].to_numpy("datetime64[ns]").astype("int64"), self.frame.to_xy(g["LATITUDE_deg"], g["LONGITUDE_deg"]))
        return replace(self, mark_tracks=tracks)

    def _wp_xy(self, k: int, t_ns) -> np.ndarray:
        """Positions of the marks of waypoint ``k`` at times ``t_ns`` -> (N, n_marks, 2)."""
        wp = self.waypoints[k]
        out = np.broadcast_to(wp.xy, (len(t_ns),) + wp.xy.shape).copy()
        for i, name in enumerate(wp.mark_names):
            if self.mark_tracks and name in self.mark_tracks:
                tm, xy = self.mark_tracks[name]
                out[:, i, 0] = np.interp(t_ns, tm, xy[:, 0])
                out[:, i, 1] = np.interp(t_ns, tm, xy[:, 1])
        return out

    # ---- static geometry -------------------------------------------------
    @property
    def n_legs(self) -> int:
        return len(self.waypoints) - 1

    @property
    def boundary(self) -> np.ndarray:
        return self.limits["Boundary"]

    def leg_axis(self, leg: int) -> np.ndarray:
        """Unit vector from the centre of waypoint ``leg-1`` to the centre of waypoint ``leg``."""
        return geo.unit(self.waypoints[leg].centre - self.waypoints[leg - 1].centre)

    def leg_length(self, leg: int) -> float:
        return float(np.linalg.norm(self.waypoints[leg].centre - self.waypoints[leg - 1].centre))

    def leg_table(self) -> pd.DataFrame:
        rows = []
        for leg in range(1, self.n_legs + 1):
            rows.append(
                {
                    "leg": leg,
                    "from": self.waypoints[leg - 1].name,
                    "to": self.waypoints[leg].name,
                    "bearing_deg": float(geo.bearing_of(self.leg_axis(leg))),
                    "length_m": self.leg_length(leg),
                }
            )
        return pd.DataFrame(rows).set_index("leg")

    # ---- track geometry --------------------------------------------------
    def detect_legs(self, t, xy, gate_margin=60.0) -> np.ndarray:
        """Leg number per sample, from positions and course geometry only.

        The track is walked waypoint by waypoint. A gate or line counts as
        passed at the first step that crosses it in the direction of the leg
        (the line is extended by ``gate_margin`` metres at both ends to absorb
        mark drift and GPS antenna offset). A single mark counts as rounded at
        the first step that crosses its rounding ray: the half-line that
        starts at the mark and points away from the inside of the turn. The
        start only counts from the start time onwards.
        """
        t_ns = pd.DatetimeIndex(t).to_numpy("datetime64[ns]").astype("int64")
        xy = np.asarray(xy, dtype=float)
        leg = np.zeros(len(xy), dtype=int)
        p0, p1 = xy[:-1], xy[1:]
        step = p1 - p0
        wps = [self._wp_xy(k, t_ns[:-1]) for k in range(len(self.waypoints))]
        centres = [w.mean(axis=1) for w in wps]
        axes = [None] + [geo.unit(centres[k] - centres[k - 1]) for k in range(1, len(wps))]
        i_from = max(int(np.searchsorted(t_ns, self.start_time.value)) - 1, 0)
        for k, wp in enumerate(self.waypoints):
            if wp.is_gate:
                a, b = wps[k][:, 0], wps[k][:, 1]
                e = geo.unit(b - a)
                hit, _, _ = geo.segment_hits(p0, p1, a - gate_margin * e, b + gate_margin * e)
                hit &= np.einsum("ij,ij->i", step, axes[max(k, 1)]) > 0
            elif k < self.n_legs:
                mark = wps[k][:, 0]
                outward = geo.unit(axes[k] - axes[k + 1])
                hit, _, _ = geo.segment_hits(p0, p1, mark, mark + 800.0 * outward)
            else:  # course ending on a single mark: abeam of it along the last leg
                rem0 = np.einsum("ij,ij->i", wps[k][:, 0] - p0, axes[k])
                rem1 = np.einsum("ij,ij->i", wps[k][:, 0] - p1, axes[k])
                hit = (rem0 > 0) & (rem1 <= 0)
            idx = np.flatnonzero(hit[i_from:])
            if len(idx) == 0:
                break
            i_cross = i_from + idx[0] + 1  # first sample on the far side
            leg[i_cross:] = k + 1
            i_from = i_cross - 1
        return leg

    def _remaining(self, k: int, xy) -> np.ndarray:
        """Distance still to sail on leg ``k``, measured along the leg axis.

        For a gate or line this is the distance along the axis to the line
        through both marks, so it is exactly zero wherever the boat crosses.
        For a single mark it is zero when the mark is abeam.
        """
        wp = self.waypoints[k]
        axis = self.leg_axis(max(k, 1))
        if wp.is_gate:
            e = geo.unit(wp.xy[1] - wp.xy[0])
            normal = np.array([-e[1], e[0]])
            den = float(axis @ normal)
            if abs(den) > 0.3:  # gate reasonably square to the leg
                return ((wp.centre - xy) @ normal) / den
        return (wp.centre - xy) @ axis

    def progress(self, xy, leg) -> np.ndarray:
        """Metres sailed along the course centre line (the 'ladder' position).

        Completed legs count with their full length; the current leg counts
        with its length minus what remains along the leg axis. Before the
        start the value is the (negative) distance behind the start line, and
        it stays at the full course length after the finish.
        """
        xy = np.asarray(xy, dtype=float)
        leg = np.asarray(leg)
        lengths = np.array([self.leg_length(k) for k in range(1, self.n_legs + 1)])
        done = np.concatenate([[0.0], np.cumsum(lengths)])
        out = np.full(len(xy), np.nan)
        pre = leg == 0
        out[pre] = -np.maximum(self._remaining(0, xy[pre]), 0.0)
        for k in range(1, self.n_legs + 1):
            m = leg == k
            if m.any():
                out[m] = done[k - 1] + np.clip(lengths[k - 1] - self._remaining(k, xy[m]), 0.0, lengths[k - 1])
        out[leg > self.n_legs] = done[-1]
        return out

    def target_point(self, xy, leg) -> np.ndarray:
        """The point each sample is racing towards: the next mark, or the closest point of the next gate."""
        xy = np.asarray(xy, dtype=float)
        leg = np.asarray(leg)
        out = np.full(xy.shape, np.nan)
        for k in range(0, self.n_legs + 1):
            m = leg == k
            if not m.any():
                continue
            wp = self.waypoints[k]
            out[m] = geo.closest_point_on_segment(xy[m], wp.xy[0], wp.xy[1]) if wp.is_gate else wp.xy[0]
        return out


# --------------------------------------------------------------------------
# track helpers
# --------------------------------------------------------------------------
def velocity_from_track(t, xy, smooth_s=3) -> np.ndarray:
    """Velocity (m/s, east/north) from positions by central differences.

    ``smooth_s`` is the width in samples of a centred moving average applied
    to the result; GPS positions at 1 Hz carry ~0.1 m of noise, which is
    several km/h when differentiated raw.
    """
    seconds = (pd.DatetimeIndex(t) - pd.DatetimeIndex(t)[0]).total_seconds().to_numpy()
    xy = np.asarray(xy, dtype=float)
    v = np.stack([np.gradient(xy[:, 0], seconds), np.gradient(xy[:, 1], seconds)], axis=1)
    if smooth_s and smooth_s > 1:
        v = pd.DataFrame(v).rolling(int(smooth_s), center=True, min_periods=1).mean().to_numpy()
    return v


def add_course_metrics(track: pd.DataFrame, course: Course, lat="LATITUDE_GPS_unk", lon="LONGITUDE_GPS_unk", time="DATETIME", smooth_s=3) -> pd.DataFrame:
    """Add leg, progress and VMC columns to one boat's track, using only time and position.

    New columns: ``x_m``, ``y_m``, ``leg`` (detected), ``progress_m``,
    ``sog_kmh`` and ``cog_deg`` (from positions), ``brg_target_deg``,
    ``dist_target_m`` and ``vmc_kmh`` (velocity made good towards the next
    mark; NaN before the start and after the finish).
    """
    out = track.sort_values(time).reset_index(drop=True).copy()
    xy = course.frame.to_xy(out[lat], out[lon])
    leg = course.detect_legs(out[time], xy)
    v = velocity_from_track(out[time], xy, smooth_s)
    to_target = course.target_point(xy, leg) - xy
    racing = (leg >= 1) & (leg <= course.n_legs)
    out["x_m"], out["y_m"] = xy[:, 0], xy[:, 1]
    out["leg"] = leg
    out["progress_m"] = course.progress(xy, leg)
    out["sog_kmh"] = np.linalg.norm(v, axis=1) * geo.KMH_PER_MS
    out["cog_deg"] = geo.bearing_of(v)
    out["brg_target_deg"] = np.where(racing, geo.bearing_of(to_target), np.nan)
    out["dist_target_m"] = np.where(racing, np.linalg.norm(to_target, axis=1), np.nan)
    out["vmc_kmh"] = np.where(racing, np.einsum("ij,ij->i", v, geo.unit(to_target)) * geo.KMH_PER_MS, np.nan)
    return out


def distance_to_leader(tracks: dict, course: Course, freq="1s", max_gap_s=5, **kwargs):
    """Distance to leader for every boat on a common time base.

    ``tracks`` maps boat name to a DataFrame with time and position columns.
    Each boat's course progress is put on a regular grid (short gaps are
    interpolated, longer ones stay empty), the leader at each instant is the
    boat with the most progress, and the distance to leader is the
    difference in progress. Boats that have not started yet are left out.

    Returns (dtl, progress, leg) as DataFrames indexed by time, one column per boat.
    """
    time = kwargs.get("time", "DATETIME")
    prog, legs = {}, {}
    for boat, track in tracks.items():
        m = add_course_metrics(track, course, **kwargs).set_index(time)
        m = m[~m.index.duplicated()]
        grid = pd.date_range(m.index.min().ceil(freq), m.index.max().floor(freq), freq=freq)
        both = m[["progress_m", "leg"]].reindex(m.index.union(grid))
        limit = int(max_gap_s / pd.Timedelta(freq).total_seconds())
        prog[boat] = both["progress_m"].interpolate("time", limit=limit, limit_area="inside").reindex(grid)
        legs[boat] = both["leg"].ffill(limit=limit).reindex(grid)
    progress = pd.DataFrame(prog)
    leg = pd.DataFrame(legs)
    started = progress.where(leg >= 1)
    dtl = (-started).add(started.max(axis=1), axis=0)
    return dtl, progress, leg


def measured_marks(marks: pd.DataFrame, t0, t1, time="DATETIME") -> dict:
    """Median measured position of every mark between ``t0`` and ``t1`` -> {name: (lat, lon)}."""
    ts = marks[time]
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    w = marks[(ts >= pd.Timestamp(t0)) & (ts <= pd.Timestamp(t1))]
    med = w.groupby("MARK")[["LATITUDE_deg", "LONGITUDE_deg"]].median()
    return {name: (row.LATITUDE_deg, row.LONGITUDE_deg) for name, row in med.iterrows()}


def as_raced_course(course: Course, marks: pd.DataFrame, tracks: dict, pad_s=20, **kwargs) -> Course:
    """Course with every mark frozen where it was measured when the fleet went past it.

    The XML holds *target* positions. Marks get moved between and during
    races, and a start-line buoy is somewhere else a few minutes after the
    gun, so each waypoint is placed at the median measured position of its
    marks between the first and the last boat passing it (start line: at the
    gun). Waypoints nobody reached keep their XML position.
    """
    time, lat, lon = kwargs.get("time", "DATETIME"), kwargs.get("lat", "LATITUDE_GPS_unk"), kwargs.get("lon", "LONGITUDE_GPS_unk")
    live = course.with_mark_tracks(marks)
    passed = {k: [] for k in range(len(course.waypoints))}
    for track in tracks.values():
        track = track.sort_values(time)
        leg = live.detect_legs(track[time], course.frame.to_xy(track[lat], track[lon]))
        for k in passed:
            if (leg > k).any():
                passed[k].append(track[time].to_numpy()[np.argmax(leg > k)])
    pad = pd.Timedelta(seconds=pad_s)
    positions = {}
    for k, wp in enumerate(course.waypoints):
        if k == 0:
            t0, t1 = course.start_time - pad / 2, course.start_time + pad / 2
        elif passed[k]:
            t0, t1 = pd.Timestamp(min(passed[k])) - pad, pd.Timestamp(max(passed[k])) + pad
        else:
            continue
        for name, pos in measured_marks(marks, t0, t1).items():
            if name in wp.mark_names:
                positions[(k, name)] = pos
    return course.with_marks(positions)
