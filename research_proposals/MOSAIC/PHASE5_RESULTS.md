# MOSAIC Phase 5 — true per-token loop vs chunk-stale feedback (Gemma-2-2b-it)

**Question.** Phase 4 won its hold result with a *chunked* loop (re-measure reading-level, re-set the
push every 32 tokens). Does a **true per-token loop** (re-measure and re-set every single generated
token) hold tighter? Phase-4's controller is *proportional* — `u = m_F·g_F − κ·e_R·g_R`, push **set**
(not accumulated) from the current reading error — so the honest variable a per-token loop changes is
not the gain but the **feedback staleness**. Phase 5 isolates exactly that on one KV-cached decode:
`update_every ∈ {1, 8, 32}`, same κ, same m_F, same prompts. (Correctness gate: zero-push per-token
decode is token-for-token identical to `model.generate`, 24/24.)

## Headline: raw per-token *chatters*; the fix (slew-limiting) is what wins

The naive expectation "per-token feedback beats chunked" is **false as stated** — a raw per-token loop
only holds tighter by going incoherent (chatter). The control-theoretic fix (slew-limited feedback)
recovers coherence and, in the best-conditioned cell, edges out the chunked loop by ~1.7 SE. Net: the
cheap chunked loop is a strong default; a *properly filtered* per-token loop is the best controller.

At **matched κ** (the only fair comparison; loop-rate the sole variable), the raw per-token loop (ue=1)
holds reading tighter than the 32-token-chunked loop (ue=32) in 3 of 4 cells — **but only by dropping
below the coherence floor.** The one cell where per-token stays coherent, its hold advantage vanishes.

| m_F | κ | per-token (ue=1) \|drift\| (d2) | chunk-stale (ue=32) \|drift\| (d2) | paired chunk−fine | verdict |
|----:|--:|--:|--:|--:|---|
| 0.06 | 4 | **1.77** (0.89) | 1.84 (0.94) | +0.06 ± 0.40 (+0.2 SE) | **tie**, ue=1 less coherent |
| 0.06 | 8 | 1.52 (**0.84**✗) | 2.32 (0.92) | +0.80 ± 0.52 (+1.5 SE) | ue=1 tighter **but incoherent** |
| 0.08 | 4 | 2.02 (**0.81**✗) | 2.89 (0.91) | +0.87 ± 0.53 (+1.6 SE) | ue=1 tighter **but incoherent** |
| 0.08 | 8 | 1.40 (**0.73**✗) | 2.25 (0.84) | +0.85 ± 0.56 (+1.5 SE) | ue=1 tighter **but incoherent** |

Averaged over closed-loop cells: **per-token mean distinct-2 = 0.82 (1/4 coherent) vs chunked 0.90
(3/4 coherent)**, at comparable hold. At matched gain **and** coherence, the chunked loop weakly
dominates — equal hold (\|drift\| ≈ 1.8), better coherence, **32× cheaper**.

## Mechanism: actuator chatter (a hypothesis)

The working interpretation: updating the steering vector every token = **actuator chatter** — the noisy
per-token Flesch-Kincaid error jerks the push token-to-token, and abrupt residual-stream perturbations
degrade next-token fluency (the LLM-steering analogue of bang-bang/high-gain control chattering and wearing
the plant). The textbook fix would then be to **slew-rate-limit the control signal** (low-pass the
feedback). **Caveat — not directly measured:** we logged the FK-*error* trace (whose per-token step-size is
actually small, ~0.16) but not the *push* total-variation, so the coherence cost is equally consistent with
the per-token loop applying a larger *effective cumulative* correction over the generation. The slew-limit
result below is consistent with chatter-removal but also with plain lag / magnitude reduction (Qwen, where
the loop never chatters and the EMA only loosens the hold, hints at the latter); the two are not separated
here — logging push total-variation per token is the cheap test. What is robust: the raw per-token loop is
*less coherent* than the chunked one at matched κ, and filtering the feedback recovers it.

(Count: per-token is tighter than chunked in **all 4** Gemma cells by raw |drift|; **3 of 4** clear the
>1 SE paired bar; of those, **2** are *clean* chatter — per-token incoherent, chunked reference coherent —
and 1 (κ=8/m_F=0.08) is ambiguous because its chunked reference is also sub-floor. The JSON verdict reports
the conservative "2/4 clean-chatter" count. Paired SEs are population SE = std/√n, ddof=0; sample SE nudges
the headlines ~2–4 % with no verdict change.)

## Chatter fix: slew-limited per-token (EMA on the feedback signal)

`e_filt ← β·e_filt + (1−β)·e_measured`, controller acts on `e_filt` (β=0.8); same low lag (ue=1),
filtered command. **It works — and in the best-conditioned cell it beats the chunked loop:**

| m_F | κ | raw per-token (ue=1) | **slew-limited (ema=0.8)** | chunk-stale (ue=32) | slew vs chunk (paired) |
|----:|--:|---|---|---|---|
| 0.06 | 4 | 1.77 (0.89) | 1.82 (**0.91**) | 1.84 (0.94) | +0.01 ± 0.55 — tie |
| 0.06 | 8 | 1.52 (0.84✗) | 2.43 (**0.86**↑coh) | 2.32 (0.92) | −0.12 ± 0.84 — tie |
| 0.08 | 4 | 2.02 (0.81✗) | **1.92 (0.89↑coh)** | 2.89 (0.91) | **+0.97 ± 0.57 (+1.7 SE) — slew tighter** |
| 0.08 | 8 | 1.40 (0.73✗) | 1.58 (0.79) | 2.25 (0.84) | +0.68 ± 0.68 — tie (still incoherent) |

Slew-limiting **recovers coherence in the 2 cells where raw per-token had collapsed** (0.81→0.89,
0.84→0.86). In the best-conditioned cell (**m_F=0.08, κ=4**) the slew-limited per-token loop holds
reading **~1.7 SE tighter than the chunked loop at matched coherence** (1.92 @ d2 0.89 vs 2.89 @ d2
0.91) and near-matched formality-gain (+0.213 vs +0.231) — a genuine, if suggestive (one cell, ~1.7
SE, same effect size as the Phase-4 Gemma headline), coherent improvement. In the other cells it ties.

## Takeaways

1. **Naive "faster feedback is better" is false here.** A *raw* per-token loop trades fluency for a
   hold improvement that evaporates once coherence is held fixed (chatter). The cheap 32-token chunked
   loop is a strong default — equal coherent hold, 32× cheaper.
2. **But the right per-token controller — slew-limited (filtered feedback) — is the actual best loop in
   the better-conditioned cells**, recovering coherence and edging out the chunked loop by ~1.7 SE
   (suggestive, one cell). The control-theory framing pays off twice: it *names the failure* (actuator
   chatter) and *prescribes the fix* (slew-rate limiting / low-pass the feedback signal).
3. The reading side-effect **re-accumulates on a timescale longer than ~32 tokens**, so the marginal
   value of finer feedback is small — consistent with the Phase-4 picture (a slow downstream drift).
4. Practical guidance: **default to the chunked loop; add slew-limited per-token only when the extra
   ~1 FK-grade of hold tightness is worth the 32× cost** and you are in a well-conditioned operating cell.

## Cross-family (Qwen2.5-3B-Instruct) — no chatter, so no fix needed

On Qwen the **raw per-token loop stays coherent** at every matched-κ cell (dist2 0.86–0.90, all ≥ the
floor) — it does **not** chatter the way Gemma's does. (Gemma's chatter is scale-specific: its pscale
180.9 vs Qwen's 58.2 means a given relative push is larger in absolute residual-stream terms, so
per-token push changes perturb fluency more.) Consequently:

- Raw per-token **wins one cell coherently** (m_F=0.08, κ=2: 1.36 @ d2 0.89 vs chunk 1.84 @ d2 0.88,
  paired +0.48 ± 0.33 = **+1.5 SE**), ties the other three — the same small, suggestive, one-cell edge
  seen on Gemma's *slew-limited* arm.
- **Slew-limiting is unnecessary and slightly hurts on Qwen** (it loosens the hold: 1.66 vs raw 1.36 at
  m_F=0.08/κ=2), because there is no chatter to suppress and the EMA just adds lag.

**Unified reading across families:** loop *rate* is a minor lever — a per-token loop edges out the
chunked one by ~1.5–1.7 SE *at best*, in a single operating cell, and only when coherent. What governs
whether it's coherent is the **smoothness of the control signal** (chatter), not its update frequency:
on the larger-pscale model you must slew-limit to stay coherent; on the smaller one you needn't. The
cheap chunked loop remains the robust default on both.

Artifacts: `outputs/mosaic_phase5_{gemma-2-2b-it,Qwen2.5-3B-Instruct}.{json,png}`,
`outputs/phase5_{full,qwen}.log`.
