# Echo Edge

Autonomic readiness inference from beat-to-beat heart data, written once in
portable C and deployed to three Arm targets from the same source: an Arm64
server, a browser running WebAssembly on a phone's Arm cores, and — by
cross-compilation — a Cortex-M microcontroller.

Given RR intervals from a chest strap, Echo Edge extracts heart-rate-variability
features over a rolling window, expresses them as deviation from that
individual's own resting baseline, and classifies the wearer's autonomic state
as **Green**, **Amber**, or **Red**. Inference is local. Nothing is
transmitted.

**Live demo:** https://woadi-vector.github.io/echo-edge/ (Chrome or Edge —
Web Bluetooth is not available in Safari)

**Arm AI Optimization Challenge 2026 — Cloud AI track.**

---

## Why this is interesting

Most on-device inference work takes a large model and shrinks it to fit a small
machine. This project inverts the premise. The model was already small, so the
interesting question was what the *rest of the pipeline* cost, and how much of
the deployment surface could be deleted outright.

What it replaced: Python, scikit-learn, a pickled estimator, and a serial
bridge to a host machine that had to sit next to the sensor.

What it is now: one C translation unit, no allocator, no runtime, no dependency
beyond `sqrtf`, compiling unchanged for

| Target | Toolchain | Role |
|---|---|---|
| Arm64 / Neoverse | `gcc -mcpu=neoverse-n1` | fleet-scale scoring |
| WebAssembly | `emcc -msimd128` | in-browser, on-device |
| Cortex-M | `arm-none-eabi-gcc` | wearable firmware |

The optimization work produced results in both directions, and the failures are
documented here alongside the wins.

---

## Results

All Arm figures measured on an AWS `t4g.small` (Graviton2, Neoverse-N1) against
the WESAD-trained model, parity-verified against scikit-learn on every run.
T-series instances are burstable, so absolute nanosecond values carry more
run-to-run variance than dedicated silicon would; the ratios are stable.

### Arm-targeted compilation

Same model, same source, compile flags only:

| | `-O0` | `-O3 -mcpu=neoverse-n1` | gain |
|---|---|---|---|
| feature extraction | 1351.8 ns/beat | 219.0 ns/beat | **6.2×** |
| classify | 1107.3 ns | 361.4 ns | **3.1×** |
| end-to-end | 2350.0 ns/beat | 482.2 ns/beat | **4.9×** |
| throughput | 425,533/s/core | 2,073,776/s/core | — |

The same source on x86 gains 2.8× end-to-end from equivalent flags. **Arm gains
4.9×** — meaningfully more headroom from identical optimization work.

### Model right-sizing

Optimized build, same instance, subject-wise 5-fold cross-validation:

| trees × depth | classify | footprint | held-out accuracy |
|---|---|---|---|
| 100 × 15 | 965.5 ns | 308.8 KB | 0.903 ± 0.035 |
| 60 × 10 | 493.8 ns | — | — |
| 40 × 8 | 329.7 ns | — | — |
| **25 × 6** | **187.7 ns** | **27.8 KB** | **0.891 ± 0.034** |

**5.1× faster and 11.1× smaller for 1.2 points of held-out accuracy** — a
difference well inside one fold standard deviation. The inherited 100×15
configuration was buying almost nothing. This single change produced a larger
speedup than every layout optimization combined, and it is what makes a
Cortex-M target credible at all.

Worth noting what the large model looks like without a subject-wise split: it
scores **1.000 in-sample** against 0.903 held out. Ten points of pure
memorization, and only grouping the split by subject reveals it.

### Independent profile (Arm Performix)

Arm's own `code_hotspots` recipe on the same instance:

| function | % of samples |
|---|---|
| `echo_classify` | 57.2 |
| `echo_features` | 42.0 |
| `echo_step` | 0.6 |
| `main` | 0.2 |

**99.99% of samples land in four functions, three of them ours.** No allocator,
no runtime, no library churn appears anywhere — the "no dependencies beyond
`sqrtf`" claim is confirmed by Arm's own tooling rather than asserted.

Performix's counter-based `cpu_microarchitecture` recipe could not run:
virtualized EC2 instances expose zero PMU counters to the guest. Only the
sampling-based recipe was available.

### What did not work

**Packed node layout: +4% on Arm, −60% on x86.** Collapsing five parallel
arrays into one 8-byte struct improved cache locality but destroyed
instruction-level parallelism. At `-O3` the five-array version gave the
compiler independent loads it could reorder and prefetch; the packed version
serializes into a genuine pointer chase. Arm's narrower reorder window made
locality the dominant term. x86's did not. Same change, opposite outcomes.

**Tree-major batching: 0.53×.** Inverting the loops to stream a batch past each
resident tree should amortize model loads across operators. It did not — packing
had already shrunk the forest to fit in cache, so there was no streaming cost
left to amortize, and the change added scratch buffers and unpredictable
branches. One optimization had made the next one pointless.

**Correctness cost speed, and that was the right trade.** Matching
scikit-learn's probability averaging instead of majority-voting each tree's
argmax made classification roughly 1.9× slower. The faster version was
computing a different estimator.

---

## How it works

**Input.** RR intervals in milliseconds. The browser client reads them over Web
Bluetooth from the standard Heart Rate Service, which the Polar H10 and most
chest straps expose.

**Artifact rejection.** Two filters. An absolute one rejects intervals outside
250–2500 ms. That alone is insufficient: a dropped beat merges two intervals
into one of roughly double length, and at 73 bpm a doubled 820 ms interval is
1640 ms — well inside the plausible range, yet it inflates SDNN across the
entire window by over 800%. So a relative filter also rejects any interval
differing more than 25% from its predecessor. The accepted fraction is reported
as a signal-quality figure; sustained values below ~0.9 mean poor electrode
contact.

**Features.** Eight values over a rolling 60-second window: mean RR, mean HR,
SDNN, RMSSD, pNN50, RR slope, coefficient of variation, and window coverage.
The layout is fixed in `core/echo.h` and mirrored in `train/features.py`.

**Per-operator baselining.** A global z-score asks "is this heart rate high?"
A per-person baseline asks "is this high *for them*?" Only the second question
is answerable — a resting rate of 52 and one of 78 are both normal for their
owners. Under subject-wise cross-validation on WESAD, personalizing against
roughly three minutes of enrollment moved held-out accuracy from
**0.814 ± 0.067 to 0.891 ± 0.034** against a 0.731 majority-class baseline. The
variance halving matters more than the mean: performance stopped depending on
*which* subjects were held out, which is what generalizing to a new person
actually means.

**Model.** A random forest exported from scikit-learn into flat C arrays.
Traversal is a pointer-free walk over one packed node array; leaves index a
shared class-probability table so the C path reproduces scikit-learn's
`predict()` exactly rather than approximating it.

**Hysteresis.** A classifier sitting near a decision boundary crosses it on
almost every beat, producing a stream of transitions that describe the
threshold rather than the operator. A new state must hold for 10 consecutive
beats before it is reported — about 8 seconds at 75 bpm. On a boundary-hovering
test signal this cut reported transitions from 19 to 2.

---

## Setup

Requires a C compiler and Python 3. No accelerator, no framework, no container.

```bash
git clone https://github.com/woadi-vector/echo-edge
cd echo-edge
pip install -r train/requirements.txt

make model      # trains, emits core/echo_model.h and parity fixtures
make compare    # builds twice, prints both, verifies against scikit-learn
```

`make compare` is the validation step. The `parity: N/N fixtures match sklearn`
line confirms the C traversal reproduces scikit-learn's predictions on held-out
vectors. If it fails, the build exits non-zero and the timings below it are
meaningless by construction.

**Expected output** — two blocks, `=== BASELINE (-O0) ===` and
`=== OPTIMIZED ===`, each opening with the model ID, the architecture, and a
parity line, followed by per-stage timings and throughput.

### On Arm64

```bash
# AWS Graviton (Neoverse N1)
make compare ARM_MCPU=neoverse-n1

# Azure Cobalt 100 (Neoverse N2) — the default
make compare
```

### Reproducing the right-sizing sweep

```bash
for cfg in "100 15" "60 10" "40 8" "25 6"; do set -- $cfg
  python3 train/train.py --trees $1 --depth $2 >/dev/null
  make clean >/dev/null && make optimized ARM_MCPU=neoverse-n1 >/dev/null
  echo -n "trees=$1 depth=$2: "; ./bench-optimized 300000 | grep "^classify:"
done
```

### In the browser (WebAssembly on Arm)

```bash
source /path/to/emsdk/emsdk_env.sh
./wasm/build.sh
cd docs && python3 -m http.server 8000
```

Open in Chrome or Edge. **Run simulated feed** exercises the same WASM core
without hardware. On an Arm-powered phone or tablet that inference executes on
the device's Arm cores and no data leaves the browser tab.

### Training on real data

The bundled model is trained on WESAD. `data/harmonize.py` provides adapters
for WESAD, SWELL-KW, and MMASH; each dataset needs downloading separately.

```bash
python3 train/train.py --dataset wesad --root /path/to/WESAD \
    --trees 25 --depth 6 --baseline 8 --amber-band 0.35 0.65 --drop mean_hr
```

---

## Repository layout

```
core/     echo.h, echo.c        portable inference core
          echo_model.h          GENERATED — packed forest + scaler constants
train/    features.py           reference extractor, twin of echo.c
          train.py              fit, export C, emit parity fixtures
data/     harmonize.py          dataset adapters to one feature contract
          baseline.py           per-operator enrollment
          ecg.py                R-peak detection for ECG-only corpora
bench/    bench.c               latency, throughput, parity
wasm/     echo_wasm.c           flat scalar surface for JS
docs/     index.html, app.js    Web Bluetooth client (GitHub Pages)
PROTOCOL.md                     operator protocol for data collection
```

---

## Verification

The parity harness caught two genuine defects that were invisible in the
running application:

1. The exporter wrote scikit-learn's internal class *index* rather than the
   class *label*. With a two-class corpus these diverge, and the application
   displayed a state the model could not produce. It looked entirely plausible
   on screen.
2. The C path majority-voted each tree's argmax while scikit-learn averages
   leaf distributions. These agree on pure leaves and diverge on impure ones —
   which is to say, they diverge on exactly the borderline cases the system
   exists to catch.

Both were caught in seconds by an equivalence test. Neither would have been
found by looking at the demo.

---

## Honest limitations

**Training data.** WESAD is 15 adults, 12 of them male, under acute laboratory
stress. Subject-wise 5-fold cross-validation gives 0.891 ± 0.034 held out
against a 0.731 majority-class baseline. It has not been validated on athletes,
on women specifically, or on physical exertion.

**A second corpus failed.** SWELL-KW produced held-out accuracy *below* its own
majority baseline under subject-wise splits. Published results on that dataset
commonly report ~99% using random row splits; because consecutive rows are
near-duplicate windows, those figures do not survive honest validation. The
negative result is kept here rather than omitted.

**Heart rate dominates.** Feature importances on the WESAD model give mean RR
and mean HR — the same variable expressed twice — a combined 60.6%, against 36%
for all four genuine HRV features. Withholding heart rate entirely drops
held-out accuracy from 0.891 to 0.780. WESAD's stressor raises heart rate
sharply, so on that corpus heart rate is an excellent proxy; in the field it is
confounded by exertion, caffeine, and heat. Field validation on athletes is
pending.

**Features can go dead under load.** In a live cycling session, pNN50 measured
exactly zero in every window above ~90 bpm — no successive intervals differ by
more than 50 ms at exercise intensities. That feature carries ~0.17 importance
on seated WESAD data. Features validated at rest do not necessarily survive
exertion.

**AMBER is a confidence band, not a learned class.** It reports model
uncertainty. It is not validated against graded physiological load.

**Not a medical device.** It produces a readiness indication for a human to act
on. It diagnoses nothing.

---

## License

MIT. See `LICENSE`.
