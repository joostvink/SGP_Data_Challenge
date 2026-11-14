# Validation and assumptions

## Implementation checks

Tests cover compass wrapping/concentration, finite timestamp bins, invalid inputs and weights;
gun-straddling crossings, finite start lines, terminal endpoints and directed mark rounding;
gate-progress continuity, whole-gap masking and retained completed boats; leg intervals that
telescope to elapsed race time; observed, estimated and incomplete finishes; equal-time
distance/speed cancellation; distinct valid reference boats; invalid ride-height exclusion;
nonconvex boundary edges, fetches, endpoint margins, finish-line targets and time-aware trailing
tack votes.

The notebook asserts that start and leg contributions reconcile with the same inferred finishes
in all four races. This validates internal consistency, not official results. Logged leg counters
provide a separate comparison, with sample availability and geometric-definition limitations.

## Course and timing

XML-only detection leaves all boats before the start on these supplied files. The analysis
therefore additionally uses measured marks. Intermediate gates permit a 15 m antenna/position
tolerance; start/finish segments are finite. Directed rounding rays estimate passage rather
than adjudicating racing rules. Positions through gaps longer than five seconds are not filled.

Passages are interpolated and transition segments split consistently. Completed boats remain
in ladder progress; unfinished missing boats do not. A short terminal approach may be extrapolated
for at most five seconds using recent closing speed, explicitly marked estimated. This does not
remove antenna/mark uncertainty or establish an official result.

## Wind and manoeuvres

Tack differences can reflect spatial/time sampling as well as calibration. The proposed table
is relative to the fleet; same-day held-out improvement does not establish absolute correctness.
Controlled reciprocal runs and different days/course orientations are needed.

Race tacks are the principal model population; pre-start tacks are a sensitivity population,
and the pre-start flag is a feature only in that pooled model.
Full models contain during-event features and are retrospective. Entry-only validation tests
information available before the turn. Outer evaluation and inner alpha selection group by boat.
Whole-team bootstrap intervals are pointwise and conditional on selected regularisation; ten
team clusters limit precision. Correlated coefficients are associations, not control effects.

## Routing and interpretation

Routing minimises switches on a finite grid of two best-VMG headings. Full edges must satisfy
the margin, but initial tack, legal mark clearance and heading continuity across independently
evaluated legs are not modelled. Gate marks are separate point targets; the finish is a line,
sampled at points at least the margin inside its ends. A grid optimum is not a continuous global optimum. Tactical
use requires grid, wind, sailing-angle and rounding sensitivity checks.

Phase shares use gross positive losses when gains are clipped; this differs from net finish
deficits. The logarithmic-mean distance/speed split is a stated convention with potentially
negative contributions, not unique causal identification. Start correlations pool repeated teams
across races and omit crossing position, traffic and other confounders.
