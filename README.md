# SailGP Data Analyst Challenge — Auckland, 19 January 2025

An answer to the eight questions of the [SailGP Data Analyst Challenge](https://github.com/DrSailGP/SGP_Data_Challenge).
The original task description is in [CHALLENGE.md](CHALLENGE.md).

Everything is in [main.ipynb](main.ipynb), which runs top to bottom in a few minutes. The reusable code sits in a
small package, `sgp/`, with unit tests in `tests/`.

## What the data says

| # | Question | Result |
|---|---|---|
| 1 | Mean of a compass direction | Vector mean per time bin, with a quality flag. The plain mean is wrong by more than 5° in 5 % of the ten-second heading bins of this data set, mostly on the beats, where port tack is sailed close to north. |
| 2 | VMC from XML + lat/lon | Works from positions alone. The XML is not where the marks were: the start line is ~200 m off at the gun, the leeward gate 75–110 m, and the course marks of the race 8 file 550–790 m. Marks are therefore taken from their measured positions. Leg changes match the logged leg counter to 0–1 s at the start line and to a median of 1–5 s at gates and marks. |
| 3 | Calibration | Upwind, the fleet reports a true wind direction 3.5° further left on port tack than on starboard: TWA reads ~1.8° too wide. A table per boat, point of sail and tack removes that flip on a held-out race. USA has a faulty bow wind sensor and a faulty starboard ride-height sensor. |
| 4 | Distance to leader | Progress along the course centre line; the leader is the boat with the most progress. |
| 5 | Minimum tacks / gybes | Dynamic programme in layline coordinates, inside the course boundary: beats need 1–2 tacks depending on the gate marks chosen, runs 1 gybe, reaches 0. |
| 6 | Tacked variables | Signed channels mirrored on port tack; port / starboard pairs turned into windward / leeward. |
| 7 | Tack model | Staying on the foils decides a tack: one that stays up loses nothing against target (median −7 m in 40 s), one that comes down 62 m. Entry speed and peak turn rate predict it on teams the model has not seen (R² 0.78, grouped by boat). |
| 8 | Race insights | Australia won race 5 and the final from the front; Denmark won races 6 and 7. The first beat is the most expensive leg (37 % of all time lost to winners), and 70 % of the lost time is lower speed, not extra distance. |

Three things in the data that are worth knowing before using it:

* **The course XML is written about six minutes after the start.** Start buoys and the leeward gate have moved by then, and
  the race 8 file does not describe the course that was sailed. `sgp.course.as_raced_course` freezes every mark where it
  was measured while the fleet went past it.
* **The wind does not know which tack a boat is on.** Comparing port and starboard tack over a race is a check on the
  wind numbers that needs no reference instrument. Here it shows a 3–4° flip upwind in every fleet race.
* **One boat has two faulty sensors** (USA: bow wind, starboard ride height). Anything that averages port and starboard
  ride height is wrong for that boat; the code uses the median of the three ride-height sensors.

## How to run

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/jupyter notebook main.ipynb            # run from the repository root
.venv/bin/python -m pytest tests                 # 12 tests, about a minute
```

The notebook writes one extra file, `figures/dtl_race5.html`: the distance-to-leader chart of race 5 as an interactive
Bokeh plot.

## Structure

| Path | What it holds |
|---|---|
| `main.ipynb` | The answers, with tables and figures |
| `sgp/geo.py` | Circular statistics, local metric frame, small 2-D geometry helpers |
| `sgp/course.py` | Course from XML, as-raced marks, leg detection, progress, VMC, distance to leader |
| `sgp/wind.py` | Wind triangle, tack-flip check, fleet-referenced calibration table, tacked channels |
| `sgp/polar.py` | Polar targets and the minimum number of tacks / gybes inside the boundary |
| `sgp/race.py` | Leg times, gap to the winner per phase, split into "sailed further" and "sailed slower" |
| `sgp/plots.py` | Plot style, team colours, course plot |
| `sgp/data.py` | Loading the challenge files |
| `tests/` | Unit tests on small synthetic cases |
| `Data/`, `Boat_Logs_Columns_Description.txt` | The challenge data, unchanged |

## Limits

One afternoon, three fleet races and a three-boat final, in 32–44 km/h of wind. The calibration is relative to the fleet
and cannot see an error all boats share on both tacks. The tack model is observational: it says which tacks were good,
not what happens when a crew forces a faster turn. The manoeuvre summary has no column description; what its columns
mean was read from the numbers and is stated where it is used.

## AI assistance

The challenge asks which AI tools were used. Claude (Anthropic) was used throughout, for writing the code and for
drafting the text. Every number and statement in the notebook was afterwards checked against the executed outputs.

## Data

The data belongs to SailGP and comes from the challenge repository, which carries no licence.
