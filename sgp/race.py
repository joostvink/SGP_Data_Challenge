"""Race-level summaries: who was where when, and where the time went."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .course import Course, add_course_metrics

FOILING_RH_MM = 300.0  # ride height above which the boat is counted as flying


def race_tracks(logs: pd.DataFrame, course: Course) -> pd.DataFrame:
    """Boat logs of one race with leg, progress and VMC added for every boat."""
    parts = [add_course_metrics(g, course) for _, g in logs[logs["race"] == int(course.race_id)].groupby("BOAT")]
    out = pd.concat(parts, ignore_index=True)
    # Exclude the known invalid channel before aggregating available sensors.
    height = out[["LENGTH_RH_P_mm", "LENGTH_RH_S_mm", "LENGTH_RH_BOW_mm"]].copy()
    height.loc[out["BOAT"] == "USA", "LENGTH_RH_S_mm"] = np.nan
    out["ride_height_mm"] = height.median(axis=1)
    out["foiling"] = out["ride_height_mm"] > FOILING_RH_MM
    return out


def _passages(g, course, finish_tolerance_m=60.0):
    xy = g[["x_m", "y_m"]].to_numpy() if "x_m" in g else course.frame.to_xy(g["LATITUDE_GPS_unk"], g["LONGITUDE_GPS_unk"])
    events = course.passage_times(g["DATETIME"], xy)
    finish, status = course.finish_time(g["DATETIME"], xy, tolerance_m=finish_tolerance_m, events=events)
    if pd.notna(finish):
        events[course.n_legs] = finish
    return events, status, xy


def leg_times(tracks: pd.DataFrame, course: Course, finish_tolerance_m=60.0) -> pd.DataFrame:
    """Interpolated leg-entry times in seconds after the gun.

    The final column may include a short extrapolation of an approaching,
    truncated log. The finish_status attribute identifies those estimates;
    missing finishes are not treated as official retirements.
    """
    rows, statuses = {}, {}
    for boat, g in tracks.groupby("BOAT"):
        g = g.sort_values("DATETIME")
        events, statuses[boat], _ = _passages(g, course, finish_tolerance_m)
        rows[boat] = {k + 1: (events[k] - course.start_time).total_seconds() if k in events else np.nan
                      for k in range(course.n_legs + 1)}
    out = pd.DataFrame(rows).T
    out.index.name = "BOAT"
    out.attrs["finish_status"] = statuses
    return out


def _position_at(seconds, xy, query):
    """Linear interpolation; short finish extrapolation uses recent velocity."""
    values = np.column_stack([np.interp(query, seconds, xy[:, j]) for j in [0, 1]])
    later = query > seconds[-1]
    if later.any():
        recent = seconds >= seconds[-1] - 5
        velocity = np.array([np.polyfit(seconds[recent] - seconds[-1], xy[recent, j], 1)[0] for j in [0, 1]])
        values[later] = xy[-1] + (query[later] - seconds[-1])[:, None] * velocity
    return values


def leg_summary(tracks: pd.DataFrame, course: Course, max_gap_s=5.0) -> pd.DataFrame:
    """Integrate complete leg intervals using the same crossings as leg_times.

    Transition segments are split at the interpolated passage. A long gap
    preserves elapsed time but leaves distance/speed/foil duration undefined.
    Off-foil time uses the most recent observed sensor state per interval.
    """
    rows = []
    for boat, g in tracks.groupby("BOAT"):
        g = g.sort_values("DATETIME")
        events, status, xy = _passages(g, course)
        seconds = (g["DATETIME"] - course.start_time).dt.total_seconds().to_numpy()
        for k in range(1, course.n_legs + 1):
            if k - 1 not in events or k not in events:
                continue
            a = (events[k - 1] - course.start_time).total_seconds()
            b = (events[k] - course.start_time).total_seconds()
            if b <= a:
                continue
            query = np.r_[a, seconds[(seconds > a) & (seconds < b)], b]
            positions = _position_at(seconds, xy, query)
            duration = np.diff(query)
            observed_dt = np.diff(seconds)
            overlaps = (seconds[:-1] < b) & (seconds[1:] > a)
            covered = not (observed_dt[overlaps] > max_gap_s).any() and np.isfinite(positions).all()
            previous = np.clip(np.searchsorted(seconds, query[:-1], side="right") - 1, 0, len(g) - 1)
            sailed = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() if covered else np.nan
            off = duration[~g["foiling"].to_numpy(bool)[previous]].sum() if covered else np.nan
            within = g[(seconds >= a) & (seconds <= min(b, seconds[-1]))]
            rows.append({"BOAT": boat, "leg": k, "time_s": b - a, "sailed_m": sailed,
                         "speed_kmh": sailed / (b - a) * 3.6, "off_foil_s": off,
                         "vmc_kmh": within["vmc_kmh"].mean(), "tws_kmh": within["TWS_SGP_km_h_1"].mean(),
                         "coverage_complete": covered, "finish_status": status})
    if not rows:
        return pd.DataFrame(columns=["BOAT", "leg", "time_s", "sailed_m", "speed_kmh", "off_foil_s", "vmc_kmh", "tws_kmh", "coverage_complete", "finish_status"]).set_index(["BOAT", "leg"])
    return pd.DataFrame(rows).set_index(["BOAT", "leg"])


def gap_to_winner(times: pd.DataFrame) -> pd.DataFrame:
    """Split every finisher's gap to the race winner into the start and the legs.

    ``start`` is how much later the boat crossed the start line than the
    winner did; each leg column is how much longer the boat took on that leg.
    The columns add up exactly to ``total``, the gap at the finish.
    """
    n = times.shape[1] - 1
    finished = times.dropna(subset=[n + 1])
    if finished.empty:
        raise ValueError("No complete inferred finish times are available")
    winner = finished[n + 1].idxmin()
    out = pd.DataFrame({"start": finished[1] - finished.loc[winner, 1]})
    for k in range(1, n + 1):
        out[f"leg {k}"] = (finished[k + 1] - finished[k]) - (finished.loc[winner, k + 1] - finished.loc[winner, k])
    out["total"] = finished[n + 1] - finished.loc[winner, n + 1]
    return out.sort_values("total")


def distance_speed_split(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Logarithmic-mean allocation of leg-time differences to distance/speed.

    This is an accounting convention, not a causal attribution. Components
    can be negative. Their sum equals the elapsed-time difference, including
    the continuous equal-time case with offsetting component effects.
    """
    ref = summary.xs(reference, level="BOAT")
    rows = []
    for (boat, leg), r in summary.iterrows():
        if leg not in ref.index:
            continue
        b = ref.loc[leg]
        required = np.array([r["time_s"], b["time_s"], r["sailed_m"], b["sailed_m"]])
        if not np.isfinite(required).all() or (required <= 0).any():
            raise ValueError(f"Positive, complete time and distance required for {boat}, leg {leg}")
        dt = r["time_s"] - b["time_s"]
        log_time = np.log1p(dt / b["time_s"])
        mean_time = dt / log_time if abs(log_time) > 1e-10 else (r["time_s"] + b["time_s"]) / 2
        further = mean_time * np.log(r["sailed_m"] / b["sailed_m"])
        speed = r["sailed_m"] / r["time_s"] * 3.6
        base_speed = b["sailed_m"] / b["time_s"] * 3.6
        rows.append({"BOAT": boat, "leg": leg, "dt_s": dt, "further_s": further,
                     "slower_s": dt - further, "extra_m": r["sailed_m"] - b["sailed_m"],
                     "speed_diff_kmh": speed - base_speed})
    return pd.DataFrame(rows, columns=["BOAT", "leg", "dt_s", "further_s", "slower_s", "extra_m", "speed_diff_kmh"]).set_index(["BOAT", "leg"])


def start_table(tracks: pd.DataFrame, course: Course) -> pd.DataFrame:
    """Start quality per boat: speed and distance to the line at the gun, time of crossing, speed at crossing."""
    rows = {}
    for boat, g in tracks.groupby("BOAT"):
        t = (g["DATETIME"] - course.start_time).dt.total_seconds()
        if not (g["leg"] >= 1).any():
            continue
        gun = g.loc[t.abs().idxmin()]
        g = g.sort_values("DATETIME")
        events, _, _ = _passages(g, course)
        if 0 not in events:
            continue
        crossing = (events[0] - course.start_time).total_seconds()
        seconds = (g["DATETIME"] - course.start_time).dt.total_seconds().to_numpy()
        rows[boat] = {
            "at_gun_behind_line_m": -gun["progress_m"] if gun["leg"] == 0 else 0.0,
            "at_gun_speed_kmh": gun["BOAT_SPEED_km_h_1"],
            "cross_s": crossing,
            "cross_speed_kmh": np.interp(crossing, seconds, g["BOAT_SPEED_km_h_1"]),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "BOAT"
    return out.sort_values("cross_s")
