# SailGP Data Analyst Challenge — Auckland, 19 January 2025

Eight answers to the [SailGP Data Analyst Challenge](https://github.com/DrSailGP/SGP_Data_Challenge).
Questions: [CHALLENGE.md](CHALLENGE.md). Executed analysis: [main.ipynb](main.ipynb).

## Findings

| # | Question | Result and qualification |
|---|---|---|
| 1 | Compass means | Arithmetic averages differ from circular averages by over 5° in 95 of 1,855 racing ten-second heading bins (5.12%). |
| 2 | VMC | The XML sequence is usable but its targets do not match the sailed course. This analysis additionally requires measured mark telemetry; XML-only detection does not start a boat in these files. |
| 3 | Wind calibration | Upwind tack differences are 3.1–4.1°. A fleet-relative adjustment reduces the held-out difference from −3.14° to +0.22°. Spatial/time sampling remains a competing explanation. |
| 4 | Distance to leader | A course-ladder proxy with explicit missing-data handling and retained completed boats. Near-line finish estimates are labelled. |
| 5 | Manoeuvre counts | Boundary-constrained dynamic programming at fixed best-VMG headings: a finite-grid optimum, not an unrestricted minimum over all polar angles. The fleet's fewest turns equal it on 25 of 28 legs; the other three are bookkeeping in the manoeuvre summary. |
| 6 | Tacked variables | Signed channels and windward/leeward pairs, using a trailing vote that resets across race/data gaps. |
| 7 | Tack model | On 107 race tacks, nested boat-grouped R² is 0.55 for retrospective explanation and 0.34 for entry-only prediction. Pooling with 56 pre-start tacks yields 0.78 and is reported separately. |
| 8 | Race differences | Reconstructed winners: AUS in races 5/8, DEN in 6/7. Phase durations and distance/speed allocations use the same interpolated passages and labelled finish estimates. |

## Assumptions

- XML creation times are 359, 377 and 398 seconds after fleet starts, and 577 seconds after the final start. These timestamps do not establish when individual targets were updated. Measured marks determine the reconstruction.
- Fleet-relative wind corrections cannot identify shared errors or separate sensor bias from environmental sampling without further controls.
- USA's implausible starboard ride-height channel is excluded before aggregation. Flight thresholds and aggregation are operational definitions, differing between logs and manoeuvre summary.
- Finishes are inferred from telemetry, not an official results feed. A short approaching log may be extrapolated by at most five seconds and is labelled estimated. This applies to 24 of 32 finishes: those logs stop 2–17 m short of the line and are extrapolated by 0.2–1.3 s. That would fit logs cut at the official finish with the antenna behind the bow, which is an assumption, not a confirmed fact. Missing finish telemetry is not automatically a retirement.

## Run

Use Python 3.11 or newer; the development environment used Python 3.14.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook main.ipynb
.venv/bin/python -m pytest -q tests
```

On Windows replace `.venv/bin/` with `.venv/Scripts/`.
The notebook also writes [figures/dtl_race5.html](figures/dtl_race5.html), an interactive race-5 chart.

## Structure

| Path | Contents |
|---|---|
| `main.ipynb` | Executed answers, figures, comparisons and limitations |
| `sgp/geo.py` | Circular statistics, projection and segment geometry |
| `sgp/course.py` | Course reconstruction, directed passages, VMC and leader gaps |
| `sgp/wind.py` | Wind diagnostics, relative corrections and tacked channels |
| `sgp/polar.py` | Polar targets and restricted-heading grid routes |
| `sgp/race.py` | Consistent leg integration, finish status and distance/speed allocation |
| `sgp/data.py`, `sgp/plots.py` | Loading and plotting helpers |
| `tests/` | Known-answer geometry, timing, routing and invalid-observation tests |
| `VALIDATION.md` | Validation coverage and methodological limits |
| `Data/`, `Boat_Logs_Columns_Description.txt` | Supplied challenge data, unchanged |

## Scope

One afternoon, three fleet races and a three-boat final. Models describe sample associations,
not causal crew actions or demonstrated cross-day performance. The manoeuvre-summary schema
is inferred from numerical relationships, as explained in the notebook. Gate tolerance,
antenna geometry, wind structure and finish extrapolation remain uncertainty sources.

Data comes from SailGP's challenge repository. No additional data licence is asserted here.
