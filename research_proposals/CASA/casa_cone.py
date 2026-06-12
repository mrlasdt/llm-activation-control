"""CASA concept-cone discovery — the refusal-specific subspace that rescues L2.

CASA's L2 lever ("ablate a k-dim refusal *subspace*, not the 2D plane") was a
NEGATIVE result as first built: the subspace was the top-k right singular vectors
of the per-layer mean-difference directions (`svd_subspace` below). That span is
not refusal-specific — it sweeps in capability-bearing directions, so projecting
all k out at every band layer destroyed coherence (+2.33 neutral-NLL, gibberish)
without de-refusing. See `CASA_PROPOSAL.md` §2 ("Why L2 backfired").

This module builds a *refusal-specific* subspace instead, following Wollschläger
et al., "The Geometry of Refusal in LLMs: Concept Cones and Representational
Independence" (ICML 2025, `papers/The Geometry of Refusal in LLM.pdf`):

  RDO  (Algorithm 1) — Refusal Direction Optimization: a single direction trained
       by gradient descent under three losses,
         L_abl  = CE(f_ablate(r)(p_harm),  t_answer)   — ablating r ⇒ answer harmful
         L_add  = CE(f_add(α r̂, l)(p_safe), t_refusal) — adding r ⇒ refuse harmless
         L_ret  = KL(f_ablate(r)(p_safe) ‖ f(p_safe))  — ablation must NOT move safe
       The RETAIN loss (L_ret) is the mechanism that makes the direction refusal-
       *specific*: it explicitly forbids the intervention from changing behaviour
       on harmless prompts. This is exactly the property the blunt SVD subspace
       lacks, and the reason it gutted coherence.

  RCO  (Algorithm 2) — Refusal Cone Optimization: an orthonormal basis B=[b_1..b_k]
       whose non-negative span (the *concept cone* {Σ λ_i b_i : λ_i ≥ 0}) is
       entirely refusal-mediating. Trained by applying RDO's ComputeLoss to (a)
       Monte-Carlo unit samples drawn from the cone (λ_i ≥ 0) and (b) the basis
       vectors themselves (the cone boundary, which degrades first), with a
       Gram–Schmidt re-projection each step.

Deployment is separate (`casa_actuator.py`): the trained cone basis B is ablated
as a *subspace* across the band (the compositional ablation of the paper's
Figure 8 — "ablating top-k directions increases ASR monotonically with k"). The
hypothesis CASA tests: because every direction in the cone is refusal-specific
(retain loss), the subspace span(B) is too — so a k>1 cone ablation de-refuses
*more completely* than k=1 at a *low* coherence tax, unlike the blunt SVD subspace.

All interventions act on the **residual-stream output of `model.layers.{j}`** (the
hook point that actually flips behaviour; ~30× the layernorm point — PTS verdict),
across a layer `band`. Orientation: directions point toward refusal (positive
coordinate = more refusal), fixed by the addition loss / DIM sign.

Self-test (synthetic, no big model):  python casa_cone.py
Real discovery is driven by `casa_experiment.py`.
"""

from __future__ import annotations

import gc
import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib

from utils import add_hooks, tokenize_instructions_fn          # noqa: E402


# =============================================================================
# Difference-in-means (DIM) in residual-stream space  +  blunt SVD subspace
# =============================================================================


def residual_means(model, tokenizer, prompts, device, bs=8):
    """Per-layer mean of the last-token residual-stream output of model.layers.{k}.

    Returns (L, d) numpy. This is the space the CASA actuator acts in (NOT the
    layernorm 'mid' point used for the AS plane)."""
    module_dict = dict(model.named_modules())
    L = model.config.num_hidden_layers
    cache = {k: [] for k in range(L)}

    def mk(k):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[k].append(h[:, -1, :].float().cpu())
        return hook

    hooks = [(module_dict[f"model.layers.{k}"], mk(k)) for k in range(L)]
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                inp = tokenize_instructions_fn(prompts[i:i + bs], tokenizer)
                model(input_ids=inp.input_ids.to(device),
                      attention_mask=inp.attention_mask.to(device))
    return np.stack([torch.cat(cache[k], 0).mean(0).numpy() for k in range(L)], 0)


def dim_directions(hmean, lmean):
    """Per-layer DIM directions (L,d), unit-normalized, oriented harmful−harmless
    (positive projection = refusal side)."""
    diff = hmean - lmean
    return diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9)


def svd_subspace(dim_per_layer, band, k):
    """The FAILED L2 baseline: top-k right singular vectors of the per-layer DIM
    directions over the band. Not refusal-specific. Returns (k,d) torch.float32."""
    diff = dim_per_layer[band]                              # (nb, d), unit rows
    _, _, Vh = np.linalg.svd(diff, full_matrices=False)
    return torch.from_numpy(Vh[:k].copy()).float()


# =============================================================================
# Differentiable interventions (depend on a direction/basis with grad)
# =============================================================================


def _ablate_hook(U):
    """Rank-k directional ablation of span(U): h' = h − UᵀU h. Differentiable in U.
    U:(k,d) need NOT be orthonormal for rank-1 (k=1); for k>1 pass orthonormal U."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        Ud = U.to(h.dtype)
        steered = h - (h @ Ud.t()) @ Ud
        return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
    return hook


def _add_hook(r_unit, alpha):
    """Activation addition of a unit direction: h' = h + α r̂. Differentiable."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        steered = h + alpha * r_unit.to(h.dtype)
        return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
    return hook


# =============================================================================
# (prompt, target) packing for teacher-forced CE / KL on target tokens
# =============================================================================


def _pack(prompts, targets, tokenizer, device, max_target_tok=32):
    """Left-padded (input_ids, attention_mask, position_ids, labels) for a batch of
    (prompt, target) pairs. `labels` are the target token ids with −100 elsewhere,
    aligned for the standard causal shift CE(logits[:, :-1], labels[:, 1:]).
    Also returns `tgt_mask`:(B,S) bool marking target positions (for KL)."""
    rows = []
    for p, t in zip(prompts, targets):
        pin = tokenize_instructions_fn([p], tokenizer)        # chat-templated prompt
        pids = pin.input_ids[0].tolist()
        tids = tokenizer.encode(t, add_special_tokens=False)[:max_target_tok]
        if len(tids) == 0:
            tids = [tokenizer.eos_token_id]
        rows.append((pids, tids))
    full = [p + t for p, t in rows]
    Lm = max(len(f) for f in full)
    pad = tokenizer.pad_token_id
    ids, attn, labels = [], [], []
    for (p, t), f in zip(rows, full):
        npad = Lm - len(f)
        ids.append([pad] * npad + f)
        attn.append([0] * npad + [1] * len(f))
        lab = [-100] * (npad + len(p)) + list(t)              # only target tokens scored
        labels.append(lab)
    ids = torch.tensor(ids, device=device)
    attn = torch.tensor(attn, device=device)
    labels = torch.tensor(labels, device=device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    tgt_mask = labels != -100
    return ids, attn, pos, labels, tgt_mask


def _ce_on_targets(logits, labels):
    """Mean CE over positions where labels != −100 (standard causal shift)."""
    lg = logits[:, :-1, :].float()
    lab = labels[:, 1:]
    loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1),
                           ignore_index=-100, reduction="mean")
    return loss


def _kl_on_targets(logits, ref_logits, tgt_mask):
    """KL(ref ‖ p) averaged over target positions — matches the paper's retain loss
    KL(f(safe) ‖ f_ablate(safe)) over the target span (forward KL from the clean
    reference distribution to the ablated one)."""
    m = tgt_mask[:, 1:]                                       # align with shift
    if not m.any():
        return logits.new_zeros(())
    # index target positions FIRST, then take the vocab-sized softmax only there —
    # avoids a (B,S,vocab) intermediate (the OOM at scale; gemma vocab=256k).
    lg = logits[:, :-1, :][m].float()                         # (Ntok, V)
    rg = ref_logits[:, :-1, :][m].float()                     # (Ntok, V)
    lp = F.log_softmax(lg, dim=-1)
    lq = F.log_softmax(rg, dim=-1)
    return (lq.exp() * (lq - lp)).sum(-1).mean()              # KL(ref‖p)


# =============================================================================
# Targets (paper recipe): t_answer via DIM-ablation, t_retain clean, t_refusal via DIM-add
# =============================================================================


@dataclass
class Targets:
    harm_prompts: list
    t_answer: list
    safe_prompts: list
    t_retain: list
    t_refusal: list


def generate_targets(model, tokenizer, dim_U, l_add, alpha, band, device,
                     harm_prompts, safe_prompts, n_tok=24):
    """Build RDO/RCO targets with the DIM direction (the paper's bootstrap):
      t_answer  = greedy continuation of harm_prompts under DIM ablation over band
      t_retain  = greedy continuation of safe_prompts with no intervention
      t_refusal = greedy continuation of safe_prompts under DIM addition at l_add
    `dim_U`:(1,d) unit DIM direction (torch, on device)."""
    module_dict = dict(model.named_modules())

    def gen(prompts, hooks):
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                g = model.generate(ids, attention_mask=attn, max_new_tokens=n_tok,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)

    abl = _ablate_hook(dim_U)
    abl_hooks = [(module_dict[f"model.layers.{j}"], abl) for j in band]
    add = _add_hook(dim_U[0], alpha)
    add_hooks_ = [(module_dict[f"model.layers.{l_add}"], add)]

    t_answer = gen(harm_prompts, abl_hooks)
    t_retain = gen(safe_prompts, [])
    t_refusal = gen(safe_prompts, add_hooks_)
    return Targets(harm_prompts, t_answer, safe_prompts, t_retain, t_refusal)


# =============================================================================
# ComputeLoss (shared by RDO and RCO) for a single unit direction
# =============================================================================


@dataclass
class TrainConfig:
    band: list = field(default_factory=list)
    l_add: int = 0
    alpha: float = 1.0
    lam_abl: float = 1.0
    lam_add: float = 0.2
    lam_ret: float = 1.0
    lr: float = 0.02
    steps: int = 40
    batch: int = 4
    n_mc: int = 4                # RCO Monte-Carlo cone samples per step
    max_target_tok: int = 24
    lr_decay: bool = True        # cosine-anneal the LR over `steps`
    snapshot_last: int = 0       # keep the last-K bases for best-of-K selection (0=off)
    seed: int = 0


class _Loss:
    """Holds packed batches + cached clean reference logits; computes the three
    losses for an arbitrary unit direction r̂ (or basis U for subspace variants)."""

    def __init__(self, model, tokenizer, tgt: Targets, cfg: TrainConfig, device):
        self.model = model; self.tok = tokenizer; self.cfg = cfg; self.device = device
        self.module_dict = dict(model.named_modules())
        # pre-pack the three datasets once
        self.abl = _pack(tgt.harm_prompts, tgt.t_answer, tokenizer, device, cfg.max_target_tok)
        self.ret = _pack(tgt.safe_prompts, tgt.t_retain, tokenizer, device, cfg.max_target_tok)
        self.add = _pack(tgt.safe_prompts, tgt.t_refusal, tokenizer, device, cfg.max_target_tok)
        # NB: the clean retain reference is recomputed per sampled batch in compute()
        # (no grad), NOT precomputed for the whole set — storing full-vocab logits for
        # all n_target retain prompts is ~GBs at gemma's 256k vocab and OOMs at scale.

    def _sample(self, pack, n):
        idx = torch.randperm(pack[0].shape[0])[:n]
        return tuple(x[idx] for x in pack)

    def compute(self, r_unit):
        """r_unit:(d,) unit direction with grad. Returns scalar loss (+ parts dict)."""
        cfg = self.cfg
        U = r_unit.unsqueeze(0)                              # (1,d)
        abl = _ablate_hook(U)
        add = _add_hook(r_unit, cfg.alpha)
        band_abl = [(self.module_dict[f"model.layers.{j}"], abl) for j in cfg.band]
        add_at = [(self.module_dict[f"model.layers.{cfg.l_add}"], add)]

        # L_abl: ablate over band on harmful, CE vs t_answer
        ids, attn, pos, lab, _ = self._sample(self.abl, cfg.batch)
        with add_hooks(module_forward_hooks=band_abl):
            lg = self.model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
        L_abl = _ce_on_targets(lg, lab)

        # L_add: add at l_add on safe, CE vs t_refusal
        ids, attn, pos, lab, _ = self._sample(self.add, cfg.batch)
        with add_hooks(module_forward_hooks=add_at):
            lg = self.model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
        L_add = _ce_on_targets(lg, lab)

        # L_ret: ablate over band on safe, KL vs clean reference (recomputed per batch)
        n = min(cfg.batch, self.ret[0].shape[0])
        idx = torch.randperm(self.ret[0].shape[0])[:n]
        ids, attn, pos, lab, tmask = (x[idx] for x in self.ret)
        with torch.no_grad():                                # clean ref: no hooks, no grad
            ref_lg = self.model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
        with add_hooks(module_forward_hooks=band_abl):
            lg = self.model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
        L_ret = _kl_on_targets(lg, ref_lg, tmask)

        L = cfg.lam_abl * L_abl + cfg.lam_add * L_add + cfg.lam_ret * L_ret
        return L, {"abl": float(L_abl.detach()), "add": float(L_add.detach()),
                   "ret": float(L_ret.detach())}


# =============================================================================
# RDO — single refusal direction (Algorithm 1)
# =============================================================================


def rdo(model, tokenizer, tgt: Targets, cfg: TrainConfig, device, r_init=None,
        log=print):
    """Optimize a single unit refusal direction. Returns (final (1,d) cpu, snapshots)
    where snapshots is a list of (1,d) cpu bases from the last `cfg.snapshot_last`
    steps (empty if snapshot_last=0) for best-of-K selection."""
    torch.manual_seed(cfg.seed)
    d = model.config.hidden_size
    r = (torch.randn(d, device=device) if r_init is None
         else torch.as_tensor(r_init, device=device).float().clone())
    r = (r / r.norm()).requires_grad_(True)
    opt = torch.optim.AdamW([r], lr=cfg.lr, weight_decay=0.0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
             if cfg.lr_decay else None)
    loss_fn = _Loss(model, tokenizer, tgt, cfg, device)
    snaps = []
    for step in range(cfg.steps):
        opt.zero_grad()
        L, parts = loss_fn.compute(r / r.norm())             # normalize for the loss
        L.backward()
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            r.div_(r.norm())                                 # project to unit sphere
        if cfg.snapshot_last and step >= cfg.steps - cfg.snapshot_last:
            snaps.append(r.detach().unsqueeze(0).cpu().clone())
        if step % max(1, cfg.steps // 8) == 0 or step == cfg.steps - 1:
            log(f"  RDO step {step:3d}  L={float(L.detach()):.4f}  "
                f"abl={parts['abl']:.3f} add={parts['add']:.3f} ret={parts['ret']:.4f}")
    return r.detach().unsqueeze(0).cpu(), snaps


# =============================================================================
# RCO — refusal concept cone (Algorithm 2)
# =============================================================================


def _gram_schmidt(B):
    """Orthonormalize rows of B:(k,d) in place-safe (returns new tensor)."""
    Q, _ = torch.linalg.qr(B.t())                            # (d,k)
    return Q.t()                                             # (k,d) orthonormal rows


def _sample_cone(B, n, generator=None):
    """n unit directions in the cone {Σ λ_i b_i : λ_i ≥ 0}: s≥0, ‖s‖=1, r = sᵀB.
    Returns (n,d). B:(k,d) orthonormal so ‖r‖=‖s‖=1."""
    k = B.shape[0]
    s = torch.randn(n, k, device=B.device, generator=generator).abs()  # λ_i ≥ 0
    s = s / (s.norm(dim=1, keepdim=True) + 1e-9)
    return s @ B                                             # (n,d)


def rco(model, tokenizer, tgt: Targets, cfg: TrainConfig, device, dim=2,
        B_init=None, log=print):
    """Optimize a `dim`-dimensional refusal concept cone basis. Returns (dim,d)
    torch.float32 (CPU), orthonormal rows, oriented toward refusal."""
    torch.manual_seed(cfg.seed)
    d = model.config.hidden_size
    if B_init is None:
        B = torch.randn(dim, d, device=device)
    else:
        B = torch.as_tensor(B_init, device=device).float().clone()
        if B.shape[0] < dim:                                 # pad with random rows
            B = torch.cat([B, torch.randn(dim - B.shape[0], d, device=device)], 0)
    B = _gram_schmidt(B).requires_grad_(True)
    opt = torch.optim.AdamW([B], lr=cfg.lr, weight_decay=0.0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
             if cfg.lr_decay else None)
    loss_fn = _Loss(model, tokenizer, tgt, cfg, device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    snaps = []

    for step in range(cfg.steps):
        opt.zero_grad()
        # Pre-draw DETACHED cone coefficients (λ_i ≥ 0, unit). We recompute the
        # sample r_j = s_j·B INSIDE the loop so each sample's full forward graph frees
        # right after its backward — bounding memory to one direction at a time. (The
        # earlier `samples=s·B` once + retain_graph=True kept all n_mc model graphs
        # alive simultaneously and OOM'd at scale.) Grad still flows to B via s_j·B.
        s_coeffs = torch.randn(cfg.n_mc, dim, device=device, generator=gen).abs()
        s_coeffs = s_coeffs / (s_coeffs.norm(dim=1, keepdim=True) + 1e-9)
        parts_acc = {"abl": 0.0, "add": 0.0, "ret": 0.0}
        total = 0.0
        for j in range(cfg.n_mc):
            r_j = s_coeffs[j] @ B                            # (d,), grad→B, fresh small graph
            Lj, pj = loss_fn.compute(r_j)
            (Lj / cfg.n_mc).backward()                       # frees this sample's graph
            total += float(Lj.detach()) / cfg.n_mc
            for kk in parts_acc: parts_acc[kk] += pj[kk] / cfg.n_mc
        # L_basis: ComputeLoss on each basis vector (the cone boundary)
        for i in range(dim):
            bi = B[i] / B[i].norm()
            Li, _ = loss_fn.compute(bi)
            (Li / dim).backward()
            total += float(Li.detach()) / dim
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            B.data = _gram_schmidt(B.data)                   # project back to Stiefel
        if cfg.snapshot_last and step >= cfg.steps - cfg.snapshot_last:
            snaps.append(B.detach().cpu().clone())
        if step % max(1, cfg.steps // 8) == 0 or step == cfg.steps - 1:
            log(f"  RCO(k={dim}) step {step:3d}  Ltot≈{total:.4f}  "
                f"abl={parts_acc['abl']:.3f} add={parts_acc['add']:.3f} "
                f"ret={parts_acc['ret']:.4f}")
    return B.detach().cpu(), snaps


def orient_to_refusal(B, dim_U):
    """Flip each basis row so it has non-negative inner product with the DIM
    refusal direction — fixes the cone orientation (positive coord = refusal).
    B:(k,d), dim_U:(1,d) both torch."""
    s = torch.sign(B @ dim_U[0])
    s = torch.where(s == 0, torch.ones_like(s), s)
    return B * s.unsqueeze(1)


# =============================================================================
# Self-test (synthetic — no big model)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    d, k = 32, 3

    # 1) cone sampling: λ_i ≥ 0, unit norm, lies in span(B)
    B = _gram_schmidt(torch.randn(k, d))
    assert torch.allclose(B @ B.t(), torch.eye(k), atol=1e-5), "GS not orthonormal"
    S = _sample_cone(B, 64)
    assert torch.allclose(S.norm(dim=1), torch.ones(64), atol=1e-5), "samples not unit"
    coeffs = S @ B.t()                                       # (64,k) = λ
    assert (coeffs >= -1e-5).all(), "cone sample has negative coefficient"
    resid = S - coeffs @ B
    assert resid.abs().max() < 1e-5, "sample not in span(B)"

    # 2) differentiable ablation: gradient flows to the direction
    r = torch.randn(d, requires_grad=True)
    h = torch.randn(2, 4, d)
    hook = _ablate_hook((r / r.norm()).unsqueeze(0))
    hh = hook(None, None, (h.clone(), "kv"))[0]
    loss = (hh ** 2).sum()
    loss.backward()
    assert r.grad is not None and r.grad.abs().sum() > 0, "no grad to direction"

    # 3) _pack + CE: target tokens scored, prompt/pad masked
    class _Tok:
        pad_token_id = 0; eos_token_id = 2
        def encode(self, s, add_special_tokens=False): return [3, 4, 5][:max(1, len(s) % 3 + 1)]
    # use a real-ish tokenizer stub via tokenize_instructions_fn is overkill; check shift logic
    labels = torch.tensor([[-100, -100, 7, 8]])
    logits = torch.zeros(1, 4, 10); logits[0, 1, 7] = 5.0; logits[0, 2, 8] = 5.0
    ce = _ce_on_targets(logits, labels)
    assert ce < 0.2, f"CE shift wrong ({ce})"               # positions 1,2 predict 7,8

    # 4) KL on targets is zero when distributions match
    lg = torch.randn(1, 4, 10)
    tmask = torch.tensor([[False, False, True, True]])
    assert _kl_on_targets(lg, lg.clone(), tmask).abs() < 1e-5, "KL(p‖p) != 0"

    # 5) orientation flips rows toward DIM
    dimU = torch.randn(1, d)
    Bo = orient_to_refusal(B.clone(), dimU)
    assert (Bo @ dimU[0] >= -1e-5).all(), "orientation failed"

    print("[casa_cone self-test] OK — cone-sampling(λ≥0,unit,in-span), "
          "diff-ablation grad, CE-shift, KL(p‖p)=0, orientation")
