# Verify: OAS vs Angular Steering (text-output comparison)

Side-by-side **text** comparison of the OAS controllers against Angular Steering on
any HF causal LM (default **`google/gemma-2-2b-it`**). The sibling of
`../../PTS/verify/` — same harness, same plane, same actuator caveats — but the
steered conditions are OAS (soft-landing LQR + LQG observer) instead of PTS-MPC.

## ⚠️ The two-actuator distinction (read this first)

Exactly as the PTS verify README warns, **two different things get called "angular
steering"** and they behave ~30× differently:

| | hook point | # layers | what it is |
|---|---|---|---|
| **Canonical Angular Steering** | `model.layers.{L}.input_layernorm` output | **1** | the **published** method (Vu & Nguyen, NeurIPS'25), via `utils.get_angular_steering_output_hook`. **Nearly inert on Gemma** (and weak on Qwen). |
| **Residual-stream reset** | `model.layers.{k}` output | 1 or a band | what OAS / PTS / CLAS actually use (`oas_controller`); a far stronger actuator. |

`compare_oas_vs_angular.py` includes **both**: condition 2 is the canonical method at
its own layernorm hook point (the faithful "Angular Steering"); conditions 3–6 reset
the **residual stream** (the OAS actuator). So OAS is compared against the published
method *and* against a like-for-like, same-actuator fixed-angle control.

## Files

- **`compare_oas_vs_angular.py`** — the 6-way comparison:
  1. `baseline` — no steering
  2. `Canonical-AS(1L,LN)` — published Angular Steering (input_layernorm, 1 layer)
  3. `Reset-1L(resid)` — residual reset at the steer layer (= OAS deadbeat corner)
  4. `Reset-band(resid)` — fixed de-refusal angle across the OAS band (= `PolicyMultiAngle`)
  5. `OAS-SoftLanding(band)` — finite-horizon LQR that lands the angle at `kT` (`PolicySoftLandingLQR`)
  6. `OAS-LQG(band)` — Kalman observer + LQR (`PolicyLQG`); ≈ (5) on a clean readout

  Per-condition first-token refusal margin + greedy generations; transcript saved to
  `outputs/compare_oas_vs_angular_<model>.txt`.
- For the **full canonical angle sweep / all-layer scan**, use the PTS verify scripts —
  `../../PTS/verify/angular_sweep.py` and `angular_layer_scan.py` (canonical AS is
  model-agnostic, so they already cover "does the published method control Gemma").
- **`outputs/`** — saved transcripts + run logs.

## How to run

```bash
cd research_proposals/OAS/verify
PY=../../../.venv/bin/python

$PY compare_oas_vs_angular.py --model google/gemma-2-2b-it          # the Gemma comparison
$PY compare_oas_vs_angular.py --model Qwen/Qwen2.5-3B-Instruct      # contrast
```

Useful flags: `--angle <deg>` (override the de-refusal angle), `--rho` / `--q-term`
(OAS soft-landing effort / terminal weight), `--adaptive-mode {0,1}` (canonical AS
mask), `--n-fit`, `--plane-samples`, `--n-gen`, `--max-new-tokens`.

Requirements: the repo venv (`../../../.venv`) + a GPU; Gemma is gated (accept the
license + `huggingface-cli login`).

## What to look for

1. **Canonical AS on Gemma:** its margin barely moves from baseline and its
   generations stay full refusals — the published single-layer layernorm rotation does
   **not** control Gemma-2-2b. (This is the faithful test of "AS didn't work on Gemma.")
2. **Residual reset vs OAS band:** the single-layer residual reset (cond 3) is the
   strongest de-refuser; the OAS soft-landing/LQG band (cond 5/6), landing the angle at
   the geometric band-end `kT`, only *partially* de-refuses — consistent with the OAS
   finding that behaviour is "written" mid-band, so the band-end `kT` is past the
   behavioural sweet spot (an enforcement-layer sweep, OAS Exp 2, would pick a better `kT`).
3. **LQG vs soft-landing:** ≈ identical text on a clean readout (the observer earns its
   place only under measurement noise — see the prototype's Exp 4).
