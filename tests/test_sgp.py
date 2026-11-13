import numpy as np
import pandas as pd
import pytest

from sgp import geo, wind
from sgp.course import Course, Waypoint, add_course_metrics, distance_to_leader
from sgp.polar import Polar, min_manoeuvres


# ---- compass statistics ---------------------------------------------------
def test_circular_mean_handles_north_wrap():
    mean, r = geo.circular_mean([350, 10])
    assert mean == pytest.approx(0.0, abs=1e-9) or mean == pytest.approx(360.0, abs=1e-9)
    assert r == pytest.approx(np.cos(np.radians(10)))


def test_circular_mean_of_opposite_directions_has_no_confidence():
    _, r = geo.circular_mean([0, 180])
    assert r == pytest.approx(0.0, abs=1e-12)


def test_resample_direction_bins_and_wraps():
    idx = pd.date_range("2025-01-19 03:00:00", periods=20, freq="1s")
    s = pd.Series([355.0, 5.0] * 10, index=idx)
    out = geo.resample_direction(s, "10s")
    assert len(out) == 2
    assert list(out["n"]) == [10, 10]
    assert np.allclose(geo.wrap180(out["mean"]), 0.0, atol=1e-9)
    assert s.resample("10s").mean().iloc[0] == pytest.approx(180.0)  # the naive mean points the wrong way


def test_resample_direction_respects_gaps_and_min_count():
    idx = pd.to_datetime(["2025-01-19 03:00:00", "2025-01-19 03:00:01", "2025-01-19 03:00:25"])
    out = geo.resample_direction(pd.Series([90.0, 92.0, 200.0], index=idx), "10s", min_count=2)
    assert out["mean"].iloc[0] == pytest.approx(91.0)
    assert np.isnan(out["mean"].iloc[1])  # empty bin
    assert np.isnan(out["mean"].iloc[2])  # one sample only


# ---- course geometry --------------------------------------------------------
def _square_course():
    """Start line on y=0, windward gate at y=1000, finish line back at y=0."""
    frame = geo.LocalFrame(-36.83, 174.76)
    wps = (
        Waypoint(0, "SL", "SP", ("SL1", "SL2"), np.array([[100.0, 0.0], [-100.0, 0.0]])),
        Waypoint(1, "WG", "SP", ("WG1", "WG2"), np.array([[100.0, 1000.0], [-100.0, 1000.0]])),
        Waypoint(2, "FL", "PS", ("FL1", "FL2"), np.array([[100.0, 0.0], [-100.0, 0.0]])),
    )
    return Course("1", pd.Timestamp("2025-01-19 03:00:00"), frame, wps, {})


def _track(course, xy, t0="2025-01-19 02:59:50"):
    lat, lon = course.frame.to_latlon(np.asarray(xy, dtype=float))
    return pd.DataFrame({"DATETIME": pd.date_range(t0, periods=len(xy), freq="1s"), "LATITUDE_GPS_unk": lat, "LONGITUDE_GPS_unk": lon})


def test_legs_progress_and_vmc_on_a_straight_up_and_down():
    course = _square_course()
    up = [(0.0, y) for y in np.arange(-195.0, 1010.0, 10.0)]  # 10 m/s straight up the middle
    down = [(0.0, y) for y in np.arange(1005.0, -20.0, -10.0)]
    m = add_course_metrics(_track(course, up + down), course, smooth_s=1)
    assert m["leg"].iloc[0] == 0 and m["leg"].max() == 3
    racing_up = m[m["leg"] == 1].iloc[2:-2]
    assert np.allclose(racing_up["vmc_kmh"], 36.0, atol=0.1)
    assert np.allclose(racing_up["progress_m"], racing_up["y_m"], atol=0.5)
    assert m["progress_m"].iloc[-1] == pytest.approx(2000.0)


def test_start_before_the_gun_does_not_count():
    course = _square_course()
    early = [(0.0, y) for y in np.arange(-50.0, 300.0, 10.0)]  # crosses the line 5 s before the start time
    m = add_course_metrics(_track(course, early, t0="2025-01-19 02:59:50"), course)
    assert m["leg"].max() == 0


def test_distance_to_leader_is_the_progress_gap():
    course = _square_course()
    fast = _track(course, [(0.0, y) for y in np.arange(-195.0, 600.0, 10.0)])
    slow = _track(course, [(50.0, y) for y in np.arange(-195.0, 400.0, 7.5)])
    dtl, progress, leg = distance_to_leader({"A": fast, "B": slow}, course)
    last = dtl.dropna().iloc[-1]
    assert last["A"] == pytest.approx(0.0)
    assert last["B"] == pytest.approx(progress["A"].dropna().iloc[-1] - progress["B"].dropna().iloc[-1])
    assert (dtl.min(axis=1).dropna() == 0).all()


# ---- polar and manoeuvres ---------------------------------------------------
@pytest.fixture
def polar():
    twa = np.array([30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    table = pd.DataFrame({20.0: [20, 40, 46, 50, 52, 50, 25], 40.0: [24, 46, 52, 58, 62, 60, 30]}, index=twa, dtype=float)
    return Polar(table)


def test_fetch_needs_no_manoeuvre(polar):
    assert min_manoeuvres([0, 0], [1000, 0], twd=0, tws=30, polar=polar)["manoeuvres"] == 0


def test_open_water_beat_and_run_need_one(polar):
    assert min_manoeuvres([0, 0], [0, 1000], twd=0, tws=30, polar=polar)["manoeuvres"] == 1
    assert min_manoeuvres([0, 0], [0, -1000], twd=0, tws=30, polar=polar)["manoeuvres"] == 1


def test_narrow_corridor_forces_more_tacks(polar):
    box = np.array([[-100.0, -50.0], [100.0, -50.0], [100.0, 1050.0], [-100.0, 1050.0]])
    res = min_manoeuvres([0, 0], [0, 1000], twd=0, tws=30, polar=polar, boundary=box)
    assert res["manoeuvres"] >= 4
    assert geo.points_in_polygon(res["path"][1:-1], box).all()
    assert len(res["path"]) == res["manoeuvres"] + 2


# ---- wind -------------------------------------------------------------------
def test_wind_triangle_round_trip():
    twa, tws, bsp = np.array([-45.0, 50.0, 150.0]), np.array([35.0, 40.0, 38.0]), np.array([55.0, 52.0, 75.0])
    awa, aws = wind.apparent_from_true(twa, tws, bsp)
    twa2, tws2 = wind.true_from_apparent(awa, aws, bsp)
    assert np.allclose(twa2, twa) and np.allclose(tws2, tws)
    assert np.all(np.abs(awa) < np.abs(twa))  # apparent wind is always further forward


def test_tacked_channels_mirror_port_onto_starboard():
    df = pd.DataFrame(
        {
            "BOAT": "X",
            "TWA_SGP_deg": [-45.0, -44.0, -46.0, 45.0, 44.0, 46.0],
            "HEEL_deg": [2.0, 2.0, 2.0, -2.0, -2.0, -2.0],
            "LENGTH_RH_P_mm": [900.0] * 3 + [700.0] * 3,
            "LENGTH_RH_S_mm": [700.0] * 3 + [900.0] * 3,
        }
    )
    out = wind.add_tacked(df, hold_s=1)
    assert list(out["tack"]) == ["port"] * 3 + ["stbd"] * 3
    assert (out["TWA_SGP_deg_tk"] > 0).all()
    assert out["HEEL_deg_tk"].nunique() == 1
    assert (out["LENGTH_RH_WW_mm"] == 900.0).all() and (out["LENGTH_RH_LW_mm"] == 700.0).all()
