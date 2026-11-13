"""Wind triangle, fleet-referenced calibration check and tack-normalised ("tacked") channels.

Sign conventions of the boat logs (verified against the data):
``TWA = TWD - HEADING``, so a negative TWA / AWA means the wind comes over
the port side (port tack) and a positive one means starboard tack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo


# --------------------------------------------------------------------------
# wind triangle (boat frame: x forward, y to starboard, angles = where the wind comes from)
# --------------------------------------------------------------------------
def apparent_from_true(twa_deg, tws, bsp):
    twa = np.radians(twa_deg)
    ax, ay = tws * np.cos(twa) + bsp, tws * np.sin(twa)
    return np.degrees(np.arctan2(ay, ax)), np.hypot(ax, ay)


def true_from_apparent(awa_deg, aws, bsp):
    awa = np.radians(awa_deg)
    tx, ty = aws * np.cos(awa) - bsp, aws * np.sin(awa)
    return np.degrees(np.arctan2(ty, tx)), np.hypot(tx, ty)


# --------------------------------------------------------------------------
# sample selection and fleet reference
# --------------------------------------------------------------------------
def steady_sailing(df, max_yaw=3.0, min_speed=30.0) -> pd.Series:
    """Samples where the boat is racing, moving and not turning."""
    return (df["RATE_YAW_deg_s_1"].abs() < max_yaw) & (df["BOAT_SPEED_km_h_1"] > min_speed) & (df["TRK_LEG_NUM_unk"] >= 1)


def point_of_sail(twa_deg, up=70.0, down=110.0):
    a = np.abs(np.asarray(twa_deg, dtype=float))
    return np.where(a < up, "upwind", np.where(a > down, "downwind", "reach"))


def tack_of(twa_deg):
    return np.where(np.asarray(twa_deg, dtype=float) < 0, "port", "stbd")


def leave_one_out_reference(df, twd="twd", tws="tws", time="DATETIME", boat="BOAT", min_boats=4) -> pd.DataFrame:
    """For every sample: mean TWD and TWS of the *other* boats at the same second.

    Using the rest of the fleet as the reference avoids a boat being compared
    with itself. Rows with fewer than ``min_boats`` other boats get NaN.
    """
    rad = np.radians(df[twd])
    parts = pd.DataFrame({"s": np.sin(rad), "c": np.cos(rad), "v": df[tws], "n": 1.0})
    tot = parts.groupby(df[time]).transform("sum")
    others = tot - parts
    ok = others["n"] >= min_boats
    ref = pd.DataFrame(index=df.index)
    ref["twd_ref"] = np.where(ok, geo.wrap360(np.degrees(np.arctan2(others["s"], others["c"]))), np.nan)
    ref["tws_ref"] = np.where(ok, others["v"] / others["n"].where(ok), np.nan)
    return ref


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def instantaneous_twd(df):
    """TWD rebuilt from heading and TWA (the logged TWD is a filtered version of this)."""
    return geo.wrap360(df["HEADING_deg"] + df["TWA_SGP_deg"])


def tack_flip(df, twd, tws="TWS_SGP_km_h_1", by=("race",)) -> pd.DataFrame:
    """Fleet-wide difference between the wind seen on port and on starboard tack.

    The real wind does not know which tack a boat is on, so over a race the
    two tacks must report the same direction and speed. ``flip_deg`` is the
    mean TWD on port minus the mean TWD on starboard; ``tws_ratio`` is the
    mean TWS on port divided by the mean on starboard.
    """
    x = pd.DataFrame({"twd": df[twd], "tws": df[tws], "sail": point_of_sail(df["TWA_SGP_deg"]), "tack": tack_of(df["TWA_SGP_deg"])})
    for c in by:
        x[c] = df[c].to_numpy()
    g = x.groupby(list(by) + ["sail", "tack"]).agg(twd=("twd", lambda s: circular_mean_deg(s)), tws=("tws", "mean"), n=("twd", "size")).unstack("tack")
    out = pd.DataFrame({"flip_deg": geo.wrap180(g["twd"]["port"] - g["twd"]["stbd"]), "tws_ratio": g["tws"]["port"] / g["tws"]["stbd"], "n_port": g["n"]["port"], "n_stbd": g["n"]["stbd"]})
    return out


def circular_mean_deg(values) -> float:
    return geo.circular_mean(values)[0]


def fit_calibration(df, shrink_n=60) -> pd.DataFrame:
    """Empirical wind calibration table: boat x point of sail x tack.

    For every cell the table holds

    * ``dtwa_deg``: what to subtract from the logged TWA (and therefore from
      TWD). It is the sum of a fleet-wide part - half the port/starboard
      flip of that point of sail, which makes the two tacks agree - and a
      boat part: the boat's mean TWD difference to the other boats at the
      same second, after the fleet-wide part has been removed.
    * ``tws_factor``: what to divide the logged TWS by: the boat's mean TWS
      ratio to the other boats at the same second.

    Boat parts are shrunk towards zero with weight ``n / (n + shrink_n)`` so
    that thinly populated cells do not get a noisy correction. ``df`` must be
    restricted to steady sailing. A method that only compares boats with each
    other cannot see an error that all boats share on both tacks.
    """
    x = df[["BOAT", "DATETIME"]].copy()
    x["sail"], x["tack"] = point_of_sail(df["TWA_SGP_deg"]), tack_of(df["TWA_SGP_deg"])
    x["twd"], x["tws"] = instantaneous_twd(df), df["TWS_SGP_km_h_1"].to_numpy()
    sign = np.where(x["tack"] == "port", 1.0, -1.0)

    pooled = tack_flip(df.assign(_twd=x["twd"], _all=0), "_twd", by=("_all",)).droplevel(0)
    half_flip = (pooled["flip_deg"] / 2.0).reindex(["upwind", "downwind"]).fillna(0.0)
    x["fleet"] = sign * x["sail"].map(half_flip).fillna(0.0).to_numpy()
    x["twd"] = geo.wrap360(x["twd"] - x["fleet"])

    ref = leave_one_out_reference(x)
    x["d_twd"] = geo.wrap180(x["twd"] - ref["twd_ref"])
    x["r_tws"] = x["tws"] / ref["tws_ref"]
    cells = x.dropna(subset=["d_twd"]).groupby(["BOAT", "sail", "tack"]).agg(boat_deg=("d_twd", "mean"), tws_ratio=("r_tws", "mean"), fleet_deg=("fleet", "mean"), n=("d_twd", "size"))
    w = cells["n"] / (cells["n"] + shrink_n)
    cells["dtwa_deg"] = cells["fleet_deg"] + w * cells["boat_deg"]
    cells["tws_factor"] = 1.0 + w * (cells["tws_ratio"] - 1.0)
    return cells[["dtwa_deg", "tws_factor", "fleet_deg", "boat_deg", "n"]]


def apply_calibration(df, table) -> pd.DataFrame:
    """Add ``TWA_cal_deg``, ``TWD_cal_deg`` and ``TWS_cal_km_h_1`` using a table from :func:`fit_calibration`.

    Cells that are missing from the table are left uncorrected.
    """
    out = df.copy()
    keys = pd.MultiIndex.from_arrays([out["BOAT"], point_of_sail(out["TWA_SGP_deg"]), tack_of(out["TWA_SGP_deg"])])
    dtwa = table["dtwa_deg"].reindex(keys).fillna(0.0).to_numpy()
    factor = table["tws_factor"].reindex(keys).fillna(1.0).to_numpy()
    out["TWA_cal_deg"] = geo.wrap180(out["TWA_SGP_deg"] - dtwa)
    out["TWD_cal_deg"] = geo.wrap360(out["HEADING_deg"] + out["TWA_cal_deg"])
    out["TWS_cal_km_h_1"] = out["TWS_SGP_km_h_1"] / factor
    return out


def calibration_residuals(df, twd, tws) -> pd.DataFrame:
    """Per sample: TWD difference (deg) and TWS ratio against the rest of the fleet, with point of sail and tack."""
    x = df[["BOAT", "DATETIME", "race"]].copy()
    x["twd"], x["tws"] = df[twd].to_numpy(), df[tws].to_numpy()
    ref = leave_one_out_reference(x)
    x["d_twd"] = geo.wrap180(x["twd"] - ref["twd_ref"])
    x["r_tws"] = x["tws"] / ref["tws_ref"]
    x["sail"] = point_of_sail(df["TWA_SGP_deg"])
    x["tack"] = tack_of(df["TWA_SGP_deg"])
    return x.dropna(subset=["d_twd"])


# --------------------------------------------------------------------------
# tacked channels
# --------------------------------------------------------------------------
# Channels whose sign flips with the tack: multiplied by +1 on starboard, -1 on port,
# so that the same number always means the same thing relative to the wind.
SIGNED_CHANNELS = [
    "TWA_SGP_deg",
    "AWA_SGP_deg",
    "LEEWAY_deg",
    "HEEL_deg",
    "RATE_YAW_deg_s_1",
    "ANGLE_RUDDER_deg",
    "ANGLE_WING_ROT_deg",
    "ANGLE_WING_TWIST_deg",
    "ANGLE_CA1_deg",
    "ANGLE_CA2_deg",
    "ANGLE_CA4_deg",
    "ANGLE_CA5_deg",
    "ANGLE_CA6_deg",
]
# Port / starboard pairs that become windward / leeward.
PAIRED_CHANNELS = {
    "LENGTH_RH_{}_mm": ("LENGTH_RH_P_mm", "LENGTH_RH_S_mm"),
    "ANGLE_DB_RAKE_{}_deg": ("ANGLE_DB_RAKE_P_deg", "ANGLE_DB_RAKE_S_deg"),
    "ANGLE_DB_CANT_{}_deg": ("ANGLE_DB_CANT_P_deg", "ANGLE_DB_CANT_S_deg"),
}


def add_tacked(df, twa="TWA_SGP_deg", hold_s=3) -> pd.DataFrame:
    """Add tack-normalised copies of the tack-dependent channels.

    * ``tack``: "port" / "stbd" from the sign of the TWA, held through
      ``hold_s`` samples so it does not flicker while the bow passes through
      the wind.
    * ``<channel>_tk`` for every signed channel: the starboard-tack value on
      both tacks (port-tack samples are mirrored).
    * ``..._WW_...`` / ``..._LW_...`` for every port/starboard pair: the
      windward and the leeward side.
    """
    out = df.copy()
    sign = np.sign(out[twa]).replace(0, np.nan)
    if hold_s and hold_s > 1:  # majority vote over a short window, per boat
        sign = sign.groupby(out["BOAT"]).transform(lambda s: np.sign(s.rolling(hold_s, center=True, min_periods=1).sum()))
    sign = sign.replace(0, np.nan).groupby(out["BOAT"]).ffill().fillna(1.0)
    out["tack"] = np.where(sign < 0, "port", "stbd")
    port = (sign < 0).to_numpy()
    for c in SIGNED_CHANNELS:
        if c in out:
            out[c + "_tk"] = out[c] * sign
    for pattern, (p, s) in PAIRED_CHANNELS.items():
        if p in out and s in out:
            out[pattern.format("WW")] = np.where(port, out[p], out[s])
            out[pattern.format("LW")] = np.where(port, out[s], out[p])
    return out
