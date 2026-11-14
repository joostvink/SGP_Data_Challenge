"""Known-answer cases for geometry, timing and missing observations."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sgp import geo, race, wind
from sgp.course import Course, Waypoint, add_course_metrics, distance_to_leader
from sgp.polar import Polar, course_min_manoeuvres, line_min_manoeuvres, min_manoeuvres


@pytest.fixture
def course():
    wps = (
        Waypoint(0, "start", "SP", ("a", "b"), np.array([[-100., 0], [100., 0]])),
        Waypoint(1, "gate", "SP", ("c", "d"), np.array([[-100., 1000], [100., 1000]])),
        Waypoint(2, "finish", "PS", ("e", "f"), np.array([[-100., 0], [100., 0]])),
    )
    return Course("1", pd.Timestamp("2025-01-19 03:00:00"), geo.LocalFrame(0, 0), wps, {})


def track(course, points, seconds, boat="A"):
    lat, lon = course.frame.to_latlon(np.asarray(points, float))
    return pd.DataFrame({"BOAT": boat, "DATETIME": course.start_time + pd.to_timedelta(seconds, unit="s"),
                         "LATITUDE_GPS_unk": lat, "LONGITUDE_GPS_unk": lon,
                         "BOAT_SPEED_km_h_1": 36., "TWS_SGP_km_h_1": 30.})


def test_crossing_fraction_before_gun_is_rejected(course):
    t = course.start_time + pd.to_timedelta([-1, 0], unit="s")
    assert course.detect_legs(t, [[0, -5], [0, 5]]).tolist() == [0, 0]


def test_start_outside_actual_segment_is_not_extended(course):
    t = course.start_time + pd.to_timedelta([0, 1], unit="s")
    assert course.detect_legs(t, [[150, -5], [150, 5]]).tolist() == [0, 0]


def test_terminal_finish_endpoint_counts(course):
    t = course.start_time + pd.to_timedelta([0, 1, 2, 3, 4], unit="s")
    assert course.detect_legs(t, [[0, -5], [0, 5], [0, 1005], [0, 5], [0, 0]])[-1] == 3


def test_single_mark_respects_directed_rounding(course):
    mark = Waypoint(1, "mark", "Port", ("m",), np.array([[0., 1000.]]))
    c = replace(course, waypoints=(course.waypoints[0], mark, course.waypoints[-1]))
    other = replace(c, waypoints=(c.waypoints[0], replace(mark, rounding="Stbd"), c.waypoints[-1]))
    t = course.start_time + pd.to_timedelta(np.arange(5), unit="s")
    points = [[10, -5], [10, 5], [10, 1005], [-10, 1005], [-10, -5]]
    assert c.detect_legs(t, points).max() == 3
    assert other.detect_legs(t, points).max() == 1


def test_finished_leader_survives_terminal_log(course):
    fast = track(course, [[0, -5], [0, 5], [0, 1005], [0, 5], [0, -5]], [-.5, .5, 1.5, 2.5, 3.5])
    slow = track(course, [[0, -5], [0, 5], [0, 505], [0, 1005], [0, 505], [0, 5], [0, -5]], [-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5], "B")
    dtl, progress, _ = distance_to_leader({"A": fast, "B": slow}, course)
    when = course.start_time + pd.Timedelta(seconds=4)
    assert dtl.loc[when, "A"] == 0
    assert dtl.loc[when, "B"] > 0
    assert progress.loc[when, "A"] == pytest.approx(2000.)


def test_whole_long_gap_stays_missing(course):
    g = track(course, [[0, -5], [0, 5], [0, 505]], [-1, 0, 20])
    _, progress, _ = distance_to_leader({"A": g}, course)
    assert pd.isna(progress.loc[course.start_time + pd.Timedelta(seconds=1), "A"])
    assert pd.isna(progress.loc[course.start_time + pd.Timedelta(seconds=19), "A"])


def test_skew_gate_progress_is_continuous(course):
    gate = replace(course.waypoints[1], xy=np.array([[-100., 900], [100., 1100]]))
    end = replace(course.waypoints[2], xy=np.array([[1000., 900], [1000., 1100]]))
    c = replace(course, waypoints=(course.waypoints[0], gate, end))
    assert c.progress(np.array([[100., 1100], [100., 1100]]), [1, 2]).tolist() == pytest.approx([1000., 1000.])


def test_interpolated_leg_summaries_telescope(course):
    g = track(course, [[0, -5], [0, 5], [0, 995], [0, 1005], [0, 995], [0, 5], [0, -5]], [0, 1, 2, 3, 4, 5, 6])
    m = add_course_metrics(g, course)
    m["foiling"] = True
    entries = race.leg_times(m, course).loc["A"]
    summary = race.leg_summary(m, course).loc["A"]
    assert entries[1] == pytest.approx(.5)
    assert entries[2] == pytest.approx(2.5)
    assert entries[3] == pytest.approx(5.5)
    assert summary["time_s"].tolist() == pytest.approx([2., 3.])
    assert summary["time_s"].sum() == pytest.approx(entries[3] - entries[1])
    assert summary["sailed_m"].sum() == pytest.approx(2010.)


def test_short_terminal_approach_is_estimated_and_labelled(course):
    # A constant closing rate in the terminal five-second fit is not assumed:
    # exercise the method using a longer, steady final approach.
    points = [[0, -5], [0, 5], [0, 1005]] + [[0, y] for y in [65, 55, 45, 35, 25, 15, 5]]
    t = course.start_time + pd.to_timedelta(np.arange(len(points)), unit="s")
    finish, status = course.finish_time(t, points)
    assert status == "estimated"
    assert (finish - t[-1]).total_seconds() == pytest.approx(.5)


def test_stopped_short_log_is_incomplete(course):
    points = [[0, -5], [0, 5], [0, 1005]] + [[0, 50]] * 7
    t = course.start_time + pd.to_timedelta(np.arange(len(points)), unit="s")
    finish, status = course.finish_time(t, points)
    assert pd.isna(finish) and status == "incomplete"


def test_equal_time_distance_speed_effects_cancel():
    table = pd.DataFrame({"BOAT": ["A", "B"], "leg": [1, 1], "time_s": [100., 100.],
                          "sailed_m": [1000., 2000.], "speed_kmh": [36., 72.]}).set_index(["BOAT", "leg"])
    row = race.distance_speed_split(table, "A").loc[("B", 1)]
    assert row.further_s == pytest.approx(100 * np.log(2))
    assert row.slower_s == pytest.approx(-100 * np.log(2))
    assert row.further_s + row.slower_s == pytest.approx(0.)


def test_invalid_reference_counts_neither_duplicates_nor_missing():
    t = pd.Timestamp("2025-01-19")
    d = pd.DataFrame({"BOAT": ["A", "B", "B", "B", "B"], "DATETIME": [t] * 5, "twd": [10.] * 5, "tws": [30.] * 5})
    assert wind.leave_one_out_reference(d).isna().all().all()
    d["BOAT"] = list("ABCDE")
    d.loc[1:, ["twd", "tws"]] = np.nan
    assert wind.leave_one_out_reference(d).isna().all().all()


def test_reference_excludes_own_boat_and_keeps_finite_counts():
    d = pd.DataFrame({"BOAT": list("ABCDE"), "DATETIME": pd.Timestamp("2025-01-19"),
                      "twd": [90., 10., 10., 10., 10.], "tws": [90., 30., 30., 30., 30.]})
    ref = wind.leave_one_out_reference(d)
    assert ref.iloc[0].to_dict() == pytest.approx({"twd_ref": 10., "tws_ref": 30.})


def test_infinite_heading_is_not_counted_and_zero_resultant_is_undefined():
    idx = pd.date_range("2025-01-19", periods=2, freq="s")
    out = geo.resample_direction(pd.Series([10., np.inf], index=idx)).iloc[0]
    assert out["n"] == 1 and out["r"] == pytest.approx(1.)
    assert np.isnan(geo.circular_mean([0., 180.])[0])
    with pytest.raises(ValueError):
        geo.circular_mean([10., 20.], [-1., 1.])


def test_tacked_vote_is_causal_and_resets_after_gap():
    d = pd.DataFrame({"BOAT": "A", "DATETIME": pd.to_datetime(["2025-01-19 00:00:00", "2025-01-19 00:00:01", "2025-01-19 00:01:00"]),
                      "TWA_SGP_deg": [-45., 45., 45.]})
    result = wind.add_tacked(d)
    assert result.tack.tolist() == ["port", "port", "stbd"]
    assert wind.add_tacked(d.iloc[:1]).tack.iloc[0] == result.tack.iloc[0]


@pytest.fixture
def polar():
    return Polar(pd.DataFrame({20.: [20, 40, 46, 50, 52, 50, 25], 40.: [24, 46, 52, 58, 62, 60, 30]}, index=[30., 45, 60, 90, 120, 150, 180]))


def test_fetch_cannot_cross_boundary_notch(polar):
    poly = np.array([[-100., -100], [1100., -100], [1100., 100], [600., 100], [600., -50], [400., -50], [400., 100], [-100., 100]])
    assert min_manoeuvres([0, 0], [1000, 0], 0, 30, polar, boundary=poly)["path"] is None


def test_default_grid_edges_cannot_bridge_thin_notches(polar):
    poly = np.array([[-1000., -100], [1000., -100], [1000., 249.95], [20., 249.95], [20., 250.05], [1000., 250.05],
                     [1000., 1100], [-1000., 1100], [-1000., 250.05], [-20., 250.05], [-20., 249.95], [-1000., 249.95]])
    result = min_manoeuvres([0, 0], [0, 1000], 0, 30, polar, boundary=poly)
    assert result["manoeuvres"] is None or result["manoeuvres"] > 1


def test_margin_applies_to_fetch_endpoints(polar):
    poly = np.array([[-10., -10], [1010., -10], [1010., 10], [-10., 10]])
    assert min_manoeuvres([0, 0], [1000, 0], 0, 30, polar, boundary=poly, margin=20)["path"] is None


def test_finish_line_is_a_segment_target_not_two_marks(polar):
    # One end of the line needs a tack, the other sits inside the boundary margin; in between is a fetch.
    box = np.array([[-200., -200], [1005., -200], [1005., 1200], [-200., 1200]])
    a, b = np.array([300., 1000]), np.array([1000., 300])
    assert min_manoeuvres([0, 0], a, 0, 30, polar, boundary=box, margin=20)["manoeuvres"] == 1
    assert min_manoeuvres([0, 0], b, 0, 30, polar, boundary=box, margin=20)["manoeuvres"] is None
    result = line_min_manoeuvres([0, 0], a, b, 0, 30, polar, boundary=box, margin=20)
    assert result["manoeuvres"] == 0
    assert np.allclose(geo.closest_point_on_segment(result["target"][None, :], a, b)[0], result["target"])
    assert min(np.linalg.norm(result["target"] - a), np.linalg.norm(result["target"] - b)) >= 20


def test_course_table_targets_the_finish_line_once(course, polar):
    # Finish mark "f" is 10 m from the boundary: as a point target it is infeasible at a 20 m margin.
    bounded = replace(course, limits={"Boundary": np.array([[-700., -100], [110., -100], [110., 1100], [-700., 1100]])})
    table = course_min_manoeuvres(bounded, 0, 30, polar, margin=20)
    last = table[table["leg"] == bounded.n_legs]
    assert last["to"].unique().tolist() == ["fi line"]
    assert last["manoeuvres"].min() == 1
    assert set(table[table["leg"] == 1]["to"]) == {"c", "d"}


def test_usa_invalid_height_channel_is_removed(course):
    g = track(course, [[0, -5], [0, 5]], [0, 1], "USA")
    g["race"] = 1
    g["LENGTH_RH_P_mm"], g["LENGTH_RH_S_mm"], g["LENGTH_RH_BOW_mm"] = 200., -800., 600.
    result = race.race_tracks(g, course)
    assert result.ride_height_mm.tolist() == [400., 400.]
    assert result.foiling.all()
