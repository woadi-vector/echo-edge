# study/ — session analysis (Track A: early-warning behaviour)

Turns the CSVs the browser client exports (`echo_P01_<stamp>.csv`) into a
readable report. It runs on the exported logs and does **not** re-run the model —
the client already logged the classified state and the class vote shares, so the
continuous **readiness** shown here is computed from those.

```bash
# one session
python3 study/analyze.py echo_P01_2026-10-01T14-22.csv

# a whole study folder, plus a tidy per-window CSV for plotting
python3 study/analyze.py /path/to/study_folder --out windows.csv
```

Add to the `Makefile` if you want it under `make`:

```make
.PHONY: report
report:
	python3 study/analyze.py $(DIR) --out $(DIR)/windows.csv
# usage: make report DIR=/path/to/study_folder
```

## What it reports

Per session, and aggregated per participant / across the corpus:

- **Signal quality** — mean and the fraction at/above the floor (`--quality`, default 0.85). Flags sustained <0.9, the protocol's "poor contact" line.
- **Exertion gate** — how many windows sat in `activity=active`, where HRV is the known-unreliable regime.
- **Hysteresis effect** — transitions in the reported `state` vs the `raw_state`, i.e. how much flap the hold absorbed.
- **By condition** (`--by note` default, or `--by activity`) — windows, minutes, mean HR, mean readiness, and the GREEN/AMBER/RED mix.
- **Alert rate** — RED and RED+AMBER windows per hour.
- **Operator marks** — each `mark` with the state and readiness around it.
- **Rest-vs-load AUC** — how well readiness separates load windows from rest, and lead/lag of the first sustained RED relative to load onset.
- **Threshold sweep** — the Rothman Figure-2 tradeoff: sweep the `p_red` alert threshold and read alert-volume vs load-recall vs rest-FPR, to find the alarm-fatigue knee.

## Readiness

`readiness = 100 * (p_green + 0.5 * p_amber)` from the logged vote shares — RED
pulls it to 0, GREEN to 100, the uncertain AMBER band to the middle. This is the
Track-B continuous scalar, computed here from existing logs so the study gets a
trendable value with no firmware change. When Track B lands, the C core emits
the same number natively.

## Honesty

In device-shakedown data there are **no adverse events**, so "lead time" and the
rest-vs-load AUC measure how the score *behaves* around protocol stages, not how
early it predicts anything. Every such figure is labelled as a behaviour
diagnostic. It becomes an early-warning claim only when the approved study
attaches real outcomes.

## Options

| flag | default | meaning |
|---|---|---|
| `--by` | `note` | condition field to group by (`note` or `activity`) |
| `--quality` | `0.85` | signal-quality floor for the accepted-fraction figure |
| `--hold` | `3` | consecutive RED windows before an alert counts as "sustained" |
| `--out` | — | write a tidy per-window CSV (with `readiness`) for plotting |

## Files

- `analyze.py` — the analyzer. numpy + stdlib only, no pandas.
- `make_fixture.py` — generates a schema-exact **synthetic** session so the
  analyzer can be exercised before real data exists. Fabricated; not evidence.
