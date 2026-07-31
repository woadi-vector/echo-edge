# Echo Edge

Autonomic readiness inference from beat-to-beat heart data, written once in
portable C and deployed to three Arm targets from the same source: an Arm64
server, a browser running WebAssembly on a phone's Arm cores, and (via
cross-compilation) a Cortex-M microcontroller.

Given a stream of RR intervals from a chest strap, Echo Edge extracts
heart-rate-variability features over a rolling window and classifies the
wearer's autonomic state as **Green**, **Amber**, or **Red**. Inference is
local. Nothing is transmitted.

---

## Why this is interesting

Most on-device inference work optimizes a large model down to fit a small
machine. This goes the other way: the model is already small, so the
interesting question is what the *rest* of the pipeline costs, and how much of
the deployment surface can be deleted entirely.

The result is a single translation unit with no allocator, no runtime, and no
dependencies beyond `sqrtf`, which compiles unchanged for:

| Target | Toolchain | Role |
|---|---|---|
| Arm64 / Neoverse | `gcc -mcpu=neoverse-n2` | fleet-scale scoring |
| WebAssembly | `emcc -msimd128` | in-browser, on-device |
| Cortex-M | `arm-none-eabi-gcc` | wearable firmware |

The reference implementation it replaces is Python — scikit-learn, pandas, a
pickled estimator, and a serial bridge to a host machine. That host machine is
the thing this removes.

## Functionality

**Input.** RR intervals in milliseconds, from any source. The browser client
reads them over Web Bluetooth from the standard Heart Rate Service, which the
Polar H10 and most chest straps expose. Intervals outside 250–2500 ms are
rejected as ectopic beats or dropped detections.

**Features.** Eight values over a rolling 60-second window: mean RR, mean HR,
SDNN, RMSSD, pNN50, RR slope (a drift-velocity proxy), HR coefficient of
variation, and window coverage. The layout is fixed in `core/echo.h` and
mirrored in `train/features.py`; the two are checked against each other.

**Model.** A random forest, z-scored on entry, exported from scikit-learn to
flat C arrays. Traversal is a pointer-free loop over parallel arrays, which
keeps the hot path cache-friendly and makes the artifact a header file rather
than a pickle pinned to a library version.

**Output.** A state, a vote-share confidence, and the feature vector that
produced them.

## Setup

```bash
git clone <this repo> && cd echo-edge
pip install -r train/requirements.txt

make model      # trains, writes core/echo_model.h + bench/testvectors.h
make compare    # baseline vs optimized, with a parity check
```

`make compare` is the validation step. It builds the same sources twice and
prints both. The parity line confirms the C traversal reproduces scikit-learn's
predictions on held-out fixtures — if that fails, the numbers below are
meaningless and the build exits non-zero.

### On Arm64

```bash
# Azure Cobalt 100 (Neoverse N2) — the default
make compare

# Ampere Altra (Neoverse N1)
make compare ARM_MCPU=neoverse-n1
```

### In the browser

```bash
source /path/to/emsdk/emsdk_env.sh
./wasm/build.sh
cd web && python3 -m http.server 8000
```

Open `http://localhost:8000` in Chrome or Edge. Web Bluetooth requires a secure
context — `localhost` counts, but a deployed copy needs HTTPS. Safari does not
implement Web Bluetooth, so iPhones and iPads cannot pair. **Run simulated
feed** drives the same WASM core with synthetic beats if no strap is present.

## Repository layout

```
core/     echo.h, echo.c        portable inference core
          echo_model.h          GENERATED — scaler + forest constants
train/    features.py           reference extractor, twin of echo.c
          train.py              fit, export C, emit parity fixtures
bench/    bench.c               latency, throughput, parity
wasm/     echo_wasm.c           flat scalar surface for JS
          build.sh              emscripten build
web/      index.html, app.js    Web Bluetooth client
```

## Honest limitations

- **The bundled model is trained on synthetic data.** `train/synthesize()`
  generates physiologically shaped but fabricated windows so the pipeline runs
  end to end on a fresh clone. Any accuracy figure it prints is a statement
  about the generator, not about people. Real labelled data replaces it before
  any claim is made.
- **RR intervals only.** The Polar H10 also streams raw single-lead ECG at
  130 Hz over Polar's proprietary PMD service. That path would give the signal
  processing stage real work to optimize and is the natural next step, but it
  requires implementing Polar's control-point protocol and is not here yet.
- **Not a medical device.** This produces a readiness indication for a human to
  act on. It diagnoses nothing.

## License

MIT. See `LICENSE`.
