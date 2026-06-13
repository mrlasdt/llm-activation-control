"""CALM Phase 3 — token-domain sustained steering on the ambient coherence signal.

CALM Phase 1+2 (layer domain) found that putting coherence INSIDE the controller does not move
the strength↔coherence frontier — because the cone-space density it optimized does not predict
genNLL (r=0.20). Coherence degradation is an AMBIENT / next-token-distribution phenomenon, and
layer-domain refusal SATURATES (one shared frontier). This module tests the regime the diagnosis
points to:

  * the GENERATION (token) axis — non-saturating, where control accumulates over the body;
  * coherence measured WHERE IT LIVES — the per-token KL between the steered and unsteered
    next-token distribution (the Dynamic-Activation-Composition signal, arXiv:2406.17563);
  * a closed loop that MODULATES per-token additive-ablation strength on that KL signal
    (the In-Distribution-Steering dynamic-intensity idea, arXiv:2510.13285, as feedback control).

Disciplined ordering (three prior nulls say lookahead is inert): build static-hold + a KL
thermostat FIRST; only build a predictive token-MPC if the thermostat shows a real *dynamic* win
over the best fixed bound. Reuses CASA's additive actuator (the authority CLAS's rotation lacked),
CLAS's KV-cached decode loop (adapted: batched + Gemma-2 dense-cache-correct + dual cache for KL),
the saved k=4 cone, and StrongREJECT — so Phase 3 sits on the SAME plant as Phase 1+2.

Run:
  python calm_token.py                 # no-model math self-tests
  python calm_token.py --selfcheck     # MANDATORY correctness gates (verify vs model.generate)
  python calm_token.py --run           # the fail-fast token-domain frontier experiment
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))           # casa_*

from utils import get_input_data, tokenize_instructions_fn, add_hooks       # noqa: E402
from observables import make_margin_fn                                       # noqa: E402
from casa_actuator import ConeActuator, orthonormalize                       # noqa: E402

CASA_OUT = _HERE.parents[1] / "CASA" / "outputs"


# =============================================================================
# coherence signal: per-token KL(steered ‖ unsteered) next-token distribution
# =============================================================================


def kl_to_unsteered(log_s, log_c):
    """KL( P_steered ‖ P_clean ) per row, in nats. log_s/log_c: (B,V) raw logits.
    Penalizes the steered model putting mass where the clean model would not
    (off-distribution drift). Computed in float32."""
    ls = F.log_softmax(log_s.float(), dim=-1)
    lc = F.log_softmax(log_c.float(), dim=-1)
    p = ls.exp()
    return (p * (ls - lc)).sum(-1)                            # (B,)


# =============================================================================
# token-axis controllers — one set_strength(act, kl, margin, step) contract
# =============================================================================


class StaticHold:
    """Rung 0. Fixed u_max for all tokens — reproduces the CASA/CALM static ablation
    INSIDE the token loop. Sanity anchor: must match the casa_experiment frontier numbers."""

    def __init__(self, u_max):
        self.u_max = u_max

    def reset(self):
        pass

    def set_strength(self, act, kl, margin, step):
        act.u_max = self.u_max


class TokenThermostat:
    """Rung 1. Feedback on per-token ambient KL — the IDS-style dynamic-intensity dial.
    Hypothesis: push hard (u_hi) to flip/hold refusal while KL is cheap, ease off (toward u_lo)
    once KL exceeds a budget so the generated body stays fluent — allocating the coherence
    budget over generation-time, which a single FIXED bound cannot do.

      mode="bang"  : u = u_hi if mean-KL < budget else u_lo.
      mode="prop"  : u = u_lo + (u_hi-u_lo)·max(0, 1 − max(0,KL−budget)/budget)  (clamped)."""

    def __init__(self, u_hi, u_lo, kl_budget=0.10, mode="bang"):
        self.u_hi = float(u_hi); self.u_lo = float(u_lo)
        self.kl_budget = float(kl_budget); self.mode = mode

    def reset(self):
        pass

    def set_strength(self, act, kl, margin, step):
        klm = float(np.asarray(kl).mean())
        if self.mode == "bang":
            act.u_max = self.u_hi if klm < self.kl_budget else self.u_lo
        else:                                                 # proportional
            excess = max(0.0, klm - self.kl_budget)
            f = max(0.0, 1.0 - excess / (self.kl_budget + 1e-9))
            act.u_max = self.u_lo + (self.u_hi - self.u_lo) * f


# Rung 2 (TokenMPC) is intentionally NOT built here — gated behind a measured dynamic win
# from TokenThermostat (see CALM_RESULTS Phase 3 / the design spec). Building it first would
# repeat the lookahead-is-inert failure mode of PTS and CALM Phase 1+2.


# =============================================================================
# batched KV-cached dual-stream decode loop (steered + clean, for per-token KL)
# =============================================================================


def decode_dual(model, tokenizer, prompts, actuator, band, module_dict, controller,
                device, max_new_tokens=256):
    """Greedy decode where the STEERED stream drives generation and a parallel CLEAN stream
    (same forced tokens, no hooks) gives the unsteered next-token distribution for per-token KL.

    Returns texts + (B,T) kl/margin arrays + (T,) realized u_max schedule + per-row gen_len.
    Gemma-2 correctness: dense DynamicCache on both streams + explicit incrementing position_ids."""
    from transformers import DynamicCache
    margin_fn, _, _ = make_margin_fn(tokenizer, device)
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    B = ids.shape[0]
    act_hooks = actuator.band_hooks(module_dict, band)
    controller.reset()
    controller.set_strength(actuator, np.zeros(B), np.zeros(B), -1)   # initialize strength

    gen = [[] for _ in range(B)]
    gen_len = np.full(B, max_new_tokens, dtype=int)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    kl_t, margin_t, alpha_t = [], [], []

    with torch.no_grad():
        with add_hooks(module_forward_hooks=act_hooks):
            out_s = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                          past_key_values=DynamicCache(), use_cache=True)
        out_c = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                      past_key_values=DynamicCache(), use_cache=True)
        past_s, past_c = out_s.past_key_values, out_c.past_key_values
        log_s = out_s.logits[:, -1, :]; log_c = out_c.logits[:, -1, :]
        cur_pos = pos[:, -1:]                                  # (B,1) last real position

        for step in range(max_new_tokens):
            kl = kl_to_unsteered(log_s, log_c)                 # (B,) coherence of the upcoming token
            m = margin_fn(log_s)                               # (B,) refusal margin
            controller.set_strength(actuator, kl.cpu().numpy(), m.cpu().numpy(), step)
            kl_t.append(kl.cpu().numpy()); margin_t.append(m.cpu().numpy())
            alpha_t.append(actuator.u_max if actuator.u_max is not None else np.inf)

            nxt = log_s.argmax(-1)                             # steered drives generation
            for i in range(B):
                if not finished[i]:
                    gen[i].append(int(nxt[i]))
            newly = (nxt == tokenizer.eos_token_id) & (~finished)
            for i in torch.nonzero(newly, as_tuple=False).flatten().tolist():
                gen_len[i] = step + 1
            finished |= (nxt == tokenizer.eos_token_id)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break

            cur_pos = cur_pos + 1
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            tok = nxt[:, None]
            with add_hooks(module_forward_hooks=act_hooks):
                out_s = model(input_ids=tok, attention_mask=attn, position_ids=cur_pos,
                              past_key_values=past_s, use_cache=True)
            out_c = model(input_ids=tok, attention_mask=attn, position_ids=cur_pos,
                          past_key_values=past_c, use_cache=True)
            past_s, past_c = out_s.past_key_values, out_c.past_key_values
            log_s = out_s.logits[:, -1, :]; log_c = out_c.logits[:, -1, :]

    texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
    kl_arr = np.stack(kl_t, 1) if kl_t else np.zeros((B, 0))   # (B,T)
    margin_arr = np.stack(margin_t, 1) if margin_t else np.zeros((B, 0))
    return {"texts": texts, "kl": kl_arr, "margin": margin_arr,
            "alpha": np.array(alpha_t, float), "gen_len": gen_len}


def valid_mean_kl(kl, gen_len):
    """Mean KL over pre-eos tokens only (post-eos steps are meaningless)."""
    B, T = kl.shape
    vals = []
    for i in range(B):
        n = min(int(gen_len[i]), T)
        if n > 0:
            vals.append(kl[i, :n])
    return (float(np.concatenate(vals).mean()), float(np.concatenate(vals).max())) if vals \
        else (float("nan"), float("nan"))


# =============================================================================
# clean-model coherence of the steered continuation (cross-experiment-comparable genNLL)
# =============================================================================


def gen_coherence(model, tokenizer, device, prompts, texts, max_new_tokens):
    """Mean per-token NLL of each steered continuation under the CLEAN (un-hooked) model.
    Copied from casa_experiment.gen_coherence so the genNLL number is directly comparable."""
    vals = []
    for p, t in zip(prompts, texts):
        if len(t.strip()) < 4:
            vals.append(float("nan")); continue
        pin = tokenize_instructions_fn([p], tokenizer)
        pids = pin.input_ids[0].tolist()
        tids = tokenizer.encode(t, add_special_tokens=False)[:max_new_tokens]
        full = torch.tensor([pids + tids], device=device)
        with torch.no_grad():
            lg = model(input_ids=full).logits[0, :-1, :].float()
        lab = full[0, 1:]
        tgt = torch.full_like(lab, -100); tgt[len(pids) - 1:] = lab[len(pids) - 1:]
        vals.append(float(F.cross_entropy(lg, tgt, ignore_index=-100)))
    v = np.array(vals, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


# =============================================================================
# shared loaders
# =============================================================================


def load_cone_and_band(model, device):
    """Load the saved k=4 cone basis + band/pscale from the CALM Phase-1+2 artifacts, so
    Phase 3 sits on the IDENTICAL plant (the fairness contract)."""
    name = model.config._name_or_path.split("/")[-1]
    z = np.load(CASA_OUT / f"casa_subspaces_{name}.npz")
    key = "CONE__k=4" if "CONE__k=4" in z.files else next(k for k in z.files if k.startswith("CONE"))
    U = orthonormalize(torch.from_numpy(z[key]).float().to(device))    # (k,d)
    meta = json.load(open(CASA_OUT / f"casa_frontier_{name}.json"))["meta"]
    lo, hi = meta["band"]; band = list(range(lo, hi + 1))
    return U, band, float(meta["pscale"])


def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    return model, tok


# =============================================================================
# MANDATORY correctness gates (model smoke) — the verify-against-generate checks
# =============================================================================


def selfcheck(model_name="google/gemma-2-2b-it"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(model_name, device)
    module_dict = dict(model.named_modules())
    U, band, ps = load_cone_and_band(model, device)
    harmful, harmful_test = get_input_data("harmful", "en")
    prompts = harmful_test[:2]

    # ---- Gate 1: identity-steering (mode=off) reproduces model.generate AND KL≈0 ----
    act = ConeActuator(U=U, mode="off", u_max=None)
    r = decode_dual(model, tok, prompts, act, band, module_dict, StaticHold(None),
                    device, max_new_tokens=16)
    from transformers import DynamicCache  # noqa
    inp = tokenize_instructions_fn(prompts, tok)
    with torch.no_grad():
        # NOTE: transformers 4.48 has no "dynamic" cache_implementation flag — DynamicCache is the
        # default when unspecified. Our seqs (<300 tok) are far below Gemma-2's 4096 sliding window,
        # so the manual DynamicCache loop and generate's default cache are mathematically equivalent
        # (the gate below confirms it token-for-token).
        g = model.generate(inp.input_ids.to(device), attention_mask=inp.attention_mask.to(device),
                           max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    ref = g[:, inp.input_ids.shape[1]:]
    man = [tok.encode(t, add_special_tokens=False) for t in r["texts"]]
    print("\n[gate 1] identity decode vs model.generate (n=2, 16 tok):")
    ok1 = True
    for i in range(len(prompts)):
        ref_i = ref[i].tolist()
        man_i = man[i]
        k = 0
        while k < min(len(man_i), len(ref_i)) and man_i[k] == ref_i[k]:
            k += 1
        first = (len(man_i) > 0 and len(ref_i) > 0 and man_i[0] == ref_i[0])
        print(f"  row {i}: first-token match={first}  agree {k}/{min(len(man_i),len(ref_i))} tokens")
        ok1 &= first and k >= 14
    klmax = float(np.abs(r["kl"]).max()) if r["kl"].size else 0.0
    print(f"  max |KL| under identity steering = {klmax:.2e}  (must be ≈0)")
    ok1 &= klmax < 1e-3
    print(f"  GATE 1 {'PASS' if ok1 else 'FAIL'}")

    # ---- Gate 2: batched == singleton (left-padding / position_ids correctness) ----
    act = ConeActuator(U=U, mode="off", u_max=None)
    rb = decode_dual(model, tok, prompts, act, band, module_dict, StaticHold(None),
                     device, max_new_tokens=32)
    ok2 = True
    print("\n[gate 2] batched vs singleton decode (mode=off, 32 tok):")
    for i, p in enumerate(prompts):
        rs = decode_dual(model, tok, [p], act, band, module_dict, StaticHold(None),
                         device, max_new_tokens=32)
        bi = tok.encode(rb["texts"][i], add_special_tokens=False)
        si = tok.encode(rs["texts"][0], add_special_tokens=False)
        match = bi == si
        print(f"  row {i}: batched==singleton: {match}  (len {len(bi)} vs {len(si)})")
        ok2 &= match
    print(f"  GATE 2 {'PASS' if ok2 else 'FAIL'}")

    # ---- Gate 3: ablation actually steers + KL becomes nonzero (sanity, not a pass/fail bar) ----
    act = ConeActuator(U=U, mode="ablate", u_max=0.5 * ps, rho=1.0)
    ra = decode_dual(model, tok, prompts, act, band, module_dict, StaticHold(0.5 * ps),
                     device, max_new_tokens=24)
    klm, klpk = valid_mean_kl(ra["kl"], ra["gen_len"])
    print(f"\n[gate 3] under bounded ablation (u_max=0.5·ps): mean-KL {klm:.3f} peak {klpk:.3f} "
          f"(should be >0 — the actuator perturbs the distribution)")
    print(f"  steered opening: {ra['texts'][0][:120]!r}")

    print(f"\n[selfcheck] {'ALL GATES PASS' if (ok1 and ok2) else 'GATES FAILED — fix before experiment'}")
    return ok1 and ok2


# =============================================================================
# the fail-fast token-domain frontier experiment
# =============================================================================


def run_experiment(model_name="google/gemma-2-2b-it", n=10, max_new_tokens=256, judge=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, tok = load_model(model_name, device)
    name = model_name.split("/")[-1]
    module_dict = dict(model.named_modules())
    U, band, ps = load_cone_and_band(model, device)
    print(f"model={name} band={band[0]}..{band[-1]} pscale={ps:.2f} n={n} tok={max_new_tokens}")
    harmful, harmful_test = get_input_data("harmful", "en")
    prompts = harmful_test[:n]

    # condition grid (9): 3 static anchors + 3 bang-bang + 3 proportional
    conds = []
    for fr in (1e9, 0.5, 0.25):                               # static anchors (must match frontier)
        um = None if fr > 1e8 else fr * ps
        conds.append((f"static|u={'inf' if um is None else f'{fr:g}ps'}",
                      ConeActuator(U=U, mode="ablate", u_max=um, rho=1.0), StaticHold(um)))
    # KL budgets calibrated to the observed per-token KL scale (mean ~1.85, peak ~12.8 nats under
    # u_max=0.5·ps, from the selfcheck) — set near/below the mean so the thermostat actually MODULATES
    # (pushes u_hi on low-KL tokens, eases to u_lo on high-KL/drift tokens).
    for uhi, ulo in ((1.0, 0.25), (0.5, 0.125), (1.0, 0.0)):  # bang-bang (×ps), budget 1.0 nat
        conds.append((f"bang|hi={uhi:g}lo={ulo:g}",
                      ConeActuator(U=U, mode="ablate", u_max=uhi * ps, rho=1.0),
                      TokenThermostat(uhi * ps, ulo * ps, kl_budget=1.0, mode="bang")))
    for kb in (0.5, 1.0, 2.0):                                # proportional, u_hi=1·ps (KL nats)
        conds.append((f"prop|kb={kb:g}",
                      ConeActuator(U=U, mode="ablate", u_max=ps, rho=1.0),
                      TokenThermostat(ps, 0.0, kl_budget=kb, mode="prop")))

    # baseline (no steering)
    base = decode_dual(model, tok, prompts, ConeActuator(U=U, mode="off"), band, module_dict,
                       StaticHold(None), device, max_new_tokens=max_new_tokens)
    base_gnll = gen_coherence(model, tok, device, prompts, base["texts"], max_new_tokens)

    results = {"baseline": {"texts": base["texts"], "gen_nll": base_gnll,
                            "mean_kl": 0.0, "peak_kl": 0.0}}
    print(f"\n  {'condition':22s} {'meanKL':>7s} {'peakKL':>7s} {'genNLL':>7s} {'alphaμ':>8s} {'alphaσ':>7s}")
    for tag, act, ctrl in conds:
        r = decode_dual(model, tok, prompts, act, band, module_dict, ctrl, device,
                        max_new_tokens=max_new_tokens)
        gnll = gen_coherence(model, tok, device, prompts, r["texts"], max_new_tokens)
        mkl, pkl = valid_mean_kl(r["kl"], r["gen_len"])
        a = r["alpha"]; a = a[np.isfinite(a)]
        amu = float(a.mean()) if a.size else float("nan")
        asd = float(a.std()) if a.size else 0.0
        results[tag] = {"texts": r["texts"], "gen_nll": gnll, "mean_kl": mkl, "peak_kl": pkl,
                        "alpha_mean": amu, "alpha_std": asd,
                        "alpha_t": r["alpha"].tolist(),
                        "kl_t_ex": r["kl"][:3].tolist(), "margin_t_ex": r["margin"][:3].tolist()}
        print(f"  {tag:22s} {mkl:7.3f} {pkl:7.3f} {gnll:7.3f} {amu:8.2f} {asd:7.2f}")

    # ---- StrongREJECT (behavioural ground truth) ----
    sr = {}
    if judge:
        from casa_judge import StrongRejectJudge
        print("\n[judge] StrongREJECT ...")
        J = StrongRejectJudge()
        for tag, d in results.items():
            s = J.score(list(prompts), d["texts"])
            sr[tag] = (float(np.nanmean(s)), float(np.mean(s > 0.5)))
            d["sr_score"], d["sr_asr"] = sr[tag]
        J.free()
        print(f"\n  {'condition':22s} {'srScore':>8s} {'sr>0.5':>7s} {'genNLL':>7s} {'meanKL':>7s} {'α_std':>7s}")
        for tag, d in results.items():
            print(f"  {tag:22s} {d.get('sr_score', float('nan')):8.3f} "
                  f"{d.get('sr_asr', float('nan')):7.2f} {d['gen_nll']:7.3f} "
                  f"{d['mean_kl']:7.3f} {d.get('alpha_std', 0.0):7.2f}")

    meta = {"model": model_name, "band": [band[0], band[-1]], "pscale": ps, "n": n,
            "max_new_tokens": max_new_tokens, "base_gen_nll": base_gnll,
            "base_sr_score": sr.get("baseline", (None,))[0]}
    out = _HERE.parent / "outputs" / f"calm_token_{name}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"meta": meta, "results": results, "prompts": list(prompts)}, open(out, "w"), indent=2)
    print(f"\nsaved {out}  (elapsed {time.time()-t0:.0f}s)")


# =============================================================================
# no-model math self-tests
# =============================================================================


def _selftest_kl():
    torch.manual_seed(0)
    V = 50
    a = torch.randn(4, V)
    assert float(kl_to_unsteered(a, a).abs().max()) < 1e-6, "KL(P‖P) != 0"
    b = torch.randn(4, V)
    kl = kl_to_unsteered(a, b)
    assert (kl >= -1e-6).all(), "KL must be non-negative"
    # match a hand-rolled computation
    p = F.softmax(a, -1); ref = (p * (F.log_softmax(a, -1) - F.log_softmax(b, -1))).sum(-1)
    assert torch.allclose(kl, ref, atol=1e-6)
    # match F.kl_div (which expects input=log Q, target=P, reduction batchmean→ KL(P‖Q))
    fkl = F.kl_div(F.log_softmax(b, -1), F.softmax(a, -1), reduction="none").sum(-1)
    assert torch.allclose(kl, fkl, atol=1e-5), "KL != F.kl_div"
    print("[kl] OK — KL(P‖P)=0, ≥0, matches hand-rolled and F.kl_div")


def _selftest_controllers():
    class FakeAct:
        u_max = None
    a = FakeAct()
    sh = StaticHold(7.0)
    for s in range(50):
        sh.set_strength(a, np.array([1.0]), np.array([0.0]), s)
    assert a.u_max == 7.0, "StaticHold should hold constant"
    # bang-bang: high when KL<budget, low when KL>budget
    th = TokenThermostat(10.0, 2.0, kl_budget=0.1, mode="bang")
    th.set_strength(a, np.array([0.05]), np.array([0.0]), 0); assert a.u_max == 10.0
    th.set_strength(a, np.array([0.5]), np.array([0.0]), 1); assert a.u_max == 2.0
    # proportional: monotone non-increasing in KL, clamped to [u_lo,u_hi]
    tp = TokenThermostat(10.0, 0.0, kl_budget=0.1, mode="prop")
    tp.set_strength(a, np.array([0.0]), None, 0); hi = a.u_max
    tp.set_strength(a, np.array([0.1]), None, 1); mid = a.u_max
    tp.set_strength(a, np.array([10.0]), None, 2); lo = a.u_max
    assert abs(hi - 10.0) < 1e-9 and 0.0 <= lo <= mid <= hi, f"prop not monotone: {hi},{mid},{lo}"
    print("[controllers] OK — StaticHold constant; bang flips at budget; prop monotone & clamped")


def _selftest_actuator_mutation():
    """The whole feedback design rests on the hook reading the LIVE actuator.u_max each forward."""
    torch.manual_seed(0)
    d, k = 16, 2
    U = orthonormalize(torch.randn(k, d))
    act = ConeActuator(U=U, mode="ablate", u_max=None, rho=1.0)
    h = torch.randn(1, 3, d)
    full = (act.apply(h) - h).norm().item()                  # unbounded removal
    act.u_max = 0.1
    bounded = (act.apply(h) - h).norm().item()               # tighter bound ⇒ smaller Δ
    assert bounded < full, f"mutating u_max had no effect: {bounded:.3f} !< {full:.3f}"
    print(f"[actuator-mutation] OK — live u_max changes realized ‖Δh‖ {full:.3f} -> {bounded:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="model correctness gates")
    ap.add_argument("--run", action="store_true", help="the token-domain frontier experiment")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.model)
    elif args.run:
        run_experiment(args.model, n=args.n, max_new_tokens=args.max_new_tokens, judge=not args.no_judge)
    else:
        _selftest_kl()
        _selftest_controllers()
        _selftest_actuator_mutation()
        print("[calm_token self-test] ALL OK — KL signal, controllers, live actuator mutation")
