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
    # median of the three sensors: one of them can fail (USA's starboard sensor reads about -800 mm all day)
    out["ride_height_mm"] = out[["LENGTH_RH_P_mm", "LENGTH_RH_S_mm", "LENGTH_RH_BOW_mm"]].median(axis=1)
    out["foiling"] = out["ride_height_mm"] > FOILING_RH_MM
    return out


def leg_times(tracks: pd.DataFrame, course: Course, finish_tolerance_m=60.0) -> pd.DataFrame:
    """Time each boat entered each leg, seconds after the start signal.

    Columns 1..n are the leg entry times (1 = crossed the start line) and
    ``n + 1`` is the finish. The logs of most boats stop at the finish line,
    so a boat whose last sample is within ``finish_tolerance_m`` of the full
    course length is counted as finished at that sample.
    """
    total = sum(course.leg_length(k) for k in range(1, course.n_legs + 1))
    rows = {}
    for boat, g in tracks.groupby("BOAT"):
        t = (g["DATETIME"] - course.start_time).dt.total_seconds().to_numpy()
        leg = g["leg"].to_numpy()
        row = {k: (t[np.argmax(leg >= k)] if (leg >= k).any() else np.nan) for k in range(1, course.n_legs + 2)}
        if np.isnan(row[course.n_legs + 1]) and leg.max() == course.n_legs and g["progress_m"].iloc[-1] >= total - finish_tolerance_m:
            row[course.n_legs + 1] = t[-1]
        rows[boat] = row
    out = pd.DataFrame(rows).T
    out.index.name = "BOAT"
    return out


def leg_summary(tracks: pd.DataFrame, course: Course) -> pd.DataFrame:
    """Per boat and leg: elapsed time, distance sailed, average speed, VMC, time off the foils."""
    racing = tracks[(tracks["leg"] >= 1) & (tracks["leg"] <= course.n_legs)].copy()
    step = np.hypot(racing.groupby("BOAT")["x_m"].diff(), racing.groupby("BOAT")["y_m"].diff())
    dt = racing.groupby("BOAT")["DATETIME"].diff().dt.total_seconds()
    same_leg = racing.groupby("BOAT")["leg"].diff() == 0
    racing["step_m"] = step.where(same_leg & (dt <= 5))
    racing["dt_s"] = dt.where(same_leg & (dt <= 5))
    racing["off_foil_s"] = racing["dt_s"].where(~racing["foiling"])
    g = racing.groupby(["BOAT", "leg"])
    out = g.agg(time_s=("dt_s", "sum"), sailed_m=("step_m", "sum"), off_foil_s=("off_foil_s", "sum"), vmc_kmh=("vmc_kmh", "mean"), tws_kmh=("TWS_SGP_km_h_1", "mean"))
    out["speed_kmh"] = out["sailed_m"] / out["time_s"] * 3.6
    return out


def gap_to_winner(times: pd.DataFrame) -> pd.DataFrame:
    """Split every finisher's gap to the race winner into the start and the legs.

    ``start`` is how much later the boat crossed the start line than the
    winner did; each leg column is how much longer the boat took on that leg.
    The columns add up exactly to ``total``, the gap at the finish.
    """
    n = times.shape[1] - 1
    finished = times.dropna(subset=[n + 1])
    winner = finished[n + 1].idxmin()
    out = pd.DataFrame({"start": finished[1] - finished.loc[winner, 1]})
    for k in range(1, n + 1):
        out[f"leg {k}"] = (finished[k + 1] - finished[k]) - (finished.loc[winner, k + 1] - finished.loc[winner, k])
    out["total"] = finished[n + 1] - finished.loc[winner, n + 1]
    return out.sort_values("total")


def distance_speed_split(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Split each boat's time difference to ``reference`` on a leg into 'sailed further' and 'sailed slower'.

    time = distance / speed, so log(time ratio) = log(distance ratio) - log(speed ratio).
    The time difference is shared out in proportion to those two log terms,
    which makes the two parts add up exactly to the difference.
    """
    ref = summary.xs(reference, level="BOAT")
    rows = []
    for (boat, leg), r in summary.iterrows():
        if leg not in ref.index or r["time_s"] <= 0:
            continue
        b = ref.loc[leg]
        dt = r["time_s"] - b["time_s"]
        ld, lv = np.log(r["sailed_m"] / b["sailed_m"]), -np.log(r["speed_kmh"] / b["speed_kmh"])
        share = ld / (ld + lv) if abs(ld + lv) > 1e-9 else 0.5
        rows.append({"BOAT": boat, "leg": leg, "dt_s": dt, "further_s": dt * share, "slower_s": dt * (1 - share), "extra_m": r["sailed_m"] - b["sailed_m"], "speed_diff_kmh": r["speed_kmh"] - b["speed_kmh"]})
    return pd.DataFrame(rows).set_index(["BOAT", "leg"])


def start_table(tracks: pd.DataFrame, course: Course) -> pd.DataFrame:
    """Start quality per boat: speed and distance to the line at the gun, time of crossing, speed at crossing."""
    rows = {}
    for boat, g in tracks.groupby("BOAT"):
        t = (g["DATETIME"] - course.start_time).dt.total_seconds()
        if not (g["leg"] >= 1).any():
            continue
        gun = g.loc[t.abs().idxmin()]
        cross = g.loc[(g["leg"] >= 1).idxmax()]
        rows[boat] = {
            "at_gun_behind_line_m": -gun["progress_m"] if gun["leg"] == 0 else 0.0,
            "at_gun_speed_kmh": gun["BOAT_SPEED_km_h_1"],
            "cross_s": (cross["DATETIME"] - course.start_time).total_seconds(),
            "cross_speed_kmh": cross["BOAT_SPEED_km_h_1"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "BOAT"
    return out.sort_values("cross_s")
