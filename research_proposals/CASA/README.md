# CASA — Constrained Additive Subspace Ablation

The **additive salvage of PTS**, now steering on a refusal **concept cone** instead
of the 2D plane. Two moves beyond Angular Steering:

1. **L1 — additive actuator.** Replace AS's norm-preserving rotation with bounded
   *additive* directional ablation across a layer band. On Gemma-2-2b this flips the
   first-token refusal margin **+10.83 → −2.99** with coherent compliance at +0.10
   neutral-NLL tax, where the rotation band only reaches +3.67 (still hedging). *The
   actuator was the bottleneck.*
2. **L2 — refusal concept cone.** Replace the blunt SVD subspace (which gutted
   coherence) with a **refusal-specific concept cone** discovered by gradient descent
   under a **retain loss**, following Wollschläger et al., *The Geometry of Refusal in
   LLMs* (ICML 2025). The retain loss is what makes a k>1 subspace coherent.

See **`CASA_PROPOSAL.md`** for the direction (four levers, method, novelty boundary)
and **`CASA_RESULTS.md`** for the authoritative experimental record (full table,
per-experiment analysis, generations, limitations, reproduction).

## Status (2026-06-12) — full convergence, judged by StrongREJECT

Trained to convergence (160 steps, 128 targets, best-of-32 selection) and scored on
256-token generations by the **StrongREJECT fine-tuned judge** (the paper's judge;
0=refusal/empty/gibberish → 1=full actionable assistance):

- **L1 (additive actuator): validated.** At convergence k=1 ablation is a strong
  jailbreak — StrongREJECT **0.69–0.71** at ~zero coherence tax.
- **L2 (concept cone): rescued — decisively on coherence.** The blunt SVD subspace at
  k≥2 is gibberish (StrongREJECT **≈0**, neutral ΔNLL +1.5→+2.3, genNLL 2.2). The
  retain-loss-trained **concept cone at k=4 is a strong, coherent jailbreak**
  (StrongREJECT **0.68**, 82% of prompts >0.5, neutral ΔNLL **−0.71**, genNLL 1.1).
  *A refusal-specific k>1 subspace works exactly where a blunt one fails.* The bare
  cone ties the best k=1 (0.68 vs 0.69–0.71) — dimensionality alone doesn't beat k=1.
- **k×k plant + bounded-u additive MPC: the headline win.** Plant R²≈0.999; the MPC
  distribution gives the **single best operating point of all** — StrongREJECT
  **0.76** at the lowest coherence tax (−0.41). The PTS apparatus, inert on the
  rotation actuator, is load-bearing on the additive cone actuator (the lever PTS's
  verdict named). The full CASA stack (cone + MPC) is where k>1 pays off.
- **Substring-ASR is uninformative here** (≈1.0 for everything, blind to hedges and
  gibberish); a 64-token eval also understated harm (kept in `outputs/*_64tok.*`).
  Always judge behaviour at realistic length with a real judge.

## Modules & how to run

| File | What it is |
|------|-----------|
| `casa_actuator.py` | Bounded, k-dim, bidirectional **cone actuator** (L1+L5) |
| `casa_cone.py` | **RDO + RCO** concept-cone discovery (Geometry-of-Refusal Alg. 1–2) + DIM/SVD baselines |
| `casa_control.py` | k-dim cone **plant + bounded-u additive MPC** (PTS's 2×2 generalized, no angle collapse) |
| `casa_judge.py` | **StrongREJECT** fine-tuned judge (behavioural ground truth) |
| `casa_experiment.py` | **L2-rescue + Exp-3** driver (`--full`, `--quick`, `--load-subspaces`) |
| `casa_plot.py` | Behaviour/coherence **Pareto** figure |

```bash
cd research_proposals/CASA
../../.venv/bin/python casa_actuator.py    # self-tests (no model)
../../.venv/bin/python casa_cone.py
../../.venv/bin/python casa_control.py
../../.venv/bin/python casa_judge.py       # downloads the StrongREJECT judge (~5GB)
../../.venv/bin/python casa_experiment.py --quick --mpc                          # fast smoke
../../.venv/bin/python casa_experiment.py --model google/gemma-2-2b-it --full --mpc   # full convergence + judge
# re-judge the saved subspaces at any generation length (no retraining):
../../.venv/bin/python casa_experiment.py --model google/gemma-2-2b-it --full --mpc --load-subspaces --max-new-tokens 512
../../.venv/bin/python casa_plot.py outputs/casa_cone_gemma-2-2b-it.json
```

Outputs land in `outputs/casa_cone_<model>.{txt,json,png}`; trained subspaces persist
to `casa_subspaces_<model>.npz` for cheap re-judging. The original L1 prototype lives
at `../PTS/verify/additive_subspace_steer.py`.

## ⚠️ This is de-refusal (jailbreak) research

The harmful generations are the *measured experimental outcome* (the de-refusal we
quantify), not a product. Authorized safety/steering research; treat outputs as data.
The genNLL metric exists precisely so we never mistake gibberish for a successful
jailbreak.
