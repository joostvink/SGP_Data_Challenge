"""Loading the challenge data set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .course import Course

DATA = Path(__file__).resolve().parent.parent / "Data"


def load_boat_logs(data_dir=DATA) -> pd.DataFrame:
    """All boat logs in one frame (1 Hz, UTC timestamps), with an integer ``race`` column."""
    frames = [pd.read_csv(f, parse_dates=["DATETIME", "TIME_LOCAL_unk"]) for f in sorted(Path(data_dir, "Boat_logs").glob("data_*.csv"))]
    logs = pd.concat(frames, ignore_index=True).sort_values(["BOAT", "DATETIME"]).reset_index(drop=True)
    logs["race"] = logs["TRK_RACE_NUM_unk"].astype(int)
    return logs


def load_courses(data_dir=DATA) -> dict:
    """{race id: Course}, all in the same local frame so positions are comparable between races."""
    files = sorted(Path(data_dir, "Race_XMLs").glob("*.xml"))
    first = Course.from_xml(files[0])
    courses = [first] + [Course.from_xml(f, frame=first.frame) for f in files[1:]]
    return {int(c.race_id): c for c in courses}


def load_marks(data_dir=DATA) -> pd.DataFrame:
    """Mark positions and mark wind readings (1 Hz), timestamps as naive UTC like the boat logs."""
    marks = pd.read_csv(Path(data_dir, "Course_Marks_2025-01-19.csv"), parse_dates=["DATETIME"])
    marks["DATETIME"] = marks["DATETIME"].dt.tz_convert("UTC").dt.tz_localize(None)
    return marks


def load_manoeuvres(data_dir=DATA) -> pd.DataFrame:
    man = pd.read_csv(Path(data_dir, "2025-01-19_man_summary.csv"))
    man["DATETIME"] = pd.to_datetime(man["DATETIME"], format="ISO8601").dt.tz_convert("UTC").dt.tz_localize(None)
    man["race"] = 25011900 + man["race"].astype(int)
    return man


def load_straight_lines(data_dir=DATA) -> pd.DataFrame:
    sl = pd.read_csv(Path(data_dir, "2025-01-19_straight_lines.csv"), low_memory=False, parse_dates=["DATETIME"])
    return pd.concat([sl, sl["TRK_RACE_NUM_unk"].astype(int).rename("race")], axis=1)


def load_polar(data_dir=DATA) -> pd.DataFrame:
    """Boat speed polar (km/h): index = TWA in degrees, columns = TWS in km/h."""
    polar = pd.read_csv(Path(data_dir, "2502 m8_APW_HSB2_HSRW.kph.csv"), index_col=0)
    polar.index = polar.index.astype(float)
    polar.columns = polar.columns.astype(float)
    polar.index.name, polar.columns.name = "twa_deg", "tws_kmh"
    return polar
