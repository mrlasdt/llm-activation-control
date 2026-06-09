"""Closed-Loop Angular Steering (CLAS) — prototype controller.

Implements the core pieces the CLAS proposal (research_proposal_feedback_control.md)
identifies as not-yet-existing in the repo:

  1. A *continuous* behavioral observable read off the model's own logits
     (refusal/compliance log-prob margin) — has dynamic range where the binary
     refusal label is saturated.
  2. A mutable absolute-angle steering hook whose commanded angle is read fresh
     from a shared state object on every forward pass (so a per-token loop can
     change it). Same SO(2) reset semantics as utils.get_angular_steering_output_hook.
  3. An output-feedback PI "thermostat" over tokens with anti-windup, a band
     clamp, and an online slope sign-guard (freezes the integrator near the
     non-monotone peak the repo's own sweep exhibits).

These are deliberately model-agnostic and tuple-aware (HF decoder layers return
a tuple of (hidden_states, ...)).
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Shared steering command + mutable absolute-angle hook
# =============================================================================


class SteerState:
    """Mutable per-forward steering command shared between the decode loop and
    the forward hook. `theta` is radians: a 0-dim tensor (same angle for all
    rows) or a (B,) tensor (per-sequence angle)."""

    def __init__(self):
        self.theta = None
        self.enabled = True


def make_clas_hook(b1, b2, state: SteerState):
    """Absolute-angle reset hook that reads `state.theta` each forward pass.

        h_out = h - proj_plane(h) + ||proj_plane(h)|| * (cos theta * b1 + sin theta * b2)

    This is the verified SO(2) actuator: it discards the incoming in-plane angle
    and forces the in-plane component to the absolute angle `theta`, preserving
    its magnitude (norm-preserving on the plane). Tuple-aware.
    """

    def hook(module, inp, out):
        if not state.enabled or state.theta is None:
            return out
        hidden = out[0] if isinstance(out, tuple) else out
        _b1 = b1.to(hidden.device, hidden.dtype)
        _b2 = b2.to(hidden.device, hidden.dtype)

        c1 = hidden @ _b1                      # (B, S)
        c2 = hidden @ _b2
        proj = c1.unsqueeze(-1) * _b1 + c2.unsqueeze(-1) * _b2
        scale = torch.sqrt(c1 ** 2 + c2 ** 2).unsqueeze(-1)

        theta = state.theta
        if theta.dim() == 0:
            steer = (torch.cos(theta) * _b1 + torch.sin(theta) * _b2).view(1, 1, -1)
        else:
            ct = torch.cos(theta).to(hidden.dtype)   # (B,)
            st = torch.sin(theta).to(hidden.dtype)
            steer = (ct[:, None] * _b1[None, :] + st[:, None] * _b2[None, :]).unsqueeze(1)

        steered = hidden - proj + scale * steer
        if isinstance(out, tuple):
            return (steered,) + tuple(out[1:])
        return steered

    return hook


# =============================================================================
# Continuous behavioral observable: refusal/compliance log-prob margin
# =============================================================================


def make_margin_fn(tokenizer, device):
    """Return (margin_fn, R_ids, C_ids).

    margin(logits) = log P(first token in R) - log P(first token in C),
    a continuous, bounded readout of the model's *output* distribution. High =
    leaning refusal, low/negative = leaning compliance.
    """
    refusal_phrases = ["I", "I'm", "I am", "Sorry", "As", "Unfortunately",
                       "Cannot", "It", "Apolog", "While"]
    comply_phrases = ["Sure", "Here", "Certainly", "To", "The", "Step",
                      "First", "1", "Yes", "Of", "Below", "Let"]

    def first_ids(phrases):
        ids = set()
        for p in phrases:
            for variant in (p, " " + p):
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
        return sorted(ids)

    R = torch.tensor(first_ids(refusal_phrases), device=device)
    C = torch.tensor(first_ids(comply_phrases), device=device)

    def margin(logits):                         # logits: (B, V) -> (B,)
        lp = F.log_softmax(logits.float(), dim=-1)
        return torch.logsumexp(lp[:, R], dim=-1) - torch.logsumexp(lp[:, C], dim=-1)

    return margin, R, C


# =============================================================================
# Output-feedback PI thermostat over tokens
# =============================================================================


class OuterThermostat:
    """Discrete PI controller on a measured behavioral output `y`, commanding
    the absolute steering angle (radians). Anti-windup on the integrator, a band
    clamp keeping the angle inside the measured monotone region, and an online
    slope sign-guard that freezes the integrator (and relaxes toward nominal)
    when the local response slope is wrong-signed or too flat — the honest fix
    for the unimodal G(theta) map.
    """

    def __init__(self, theta_nom, y_star, Kp, Ki, band, slope_sign=1.0,
                 sig_max=8.0, slope_floor=1e-3):
        self.theta_nom = float(theta_nom)
        self.y_star = float(y_star)
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.lo, self.hi = float(band[0]), float(band[1])
        self.slope_sign = float(slope_sign)
        self.sig_max = float(sig_max)
        self.slope_floor = float(slope_floor)
        self.sigma = None
        self.prev_y = None
        self.prev_theta = None
        self.theta = None

    def reset(self, B, device):
        self.sigma = torch.zeros(B, device=device)
        self.theta = torch.full((B,), self.theta_nom, device=device)
        self.prev_y = None
        self.prev_theta = None
        return self.theta

    def update(self, y):                         # y: (B,) -> theta: (B,)
        eps = self.y_star - y

        # Sign-guard: only on a SUSTAINED, clearly wrong-signed local response (the
        # controller is near the non-monotone peak) do we stop integrating. We trust
        # the slope estimate only when the angle actually moved enough to measure it.
        freeze = torch.zeros_like(y, dtype=torch.bool)
        if self.prev_y is not None:
            dth = self.theta - self.prev_theta
            dy = y - self.prev_y
            moved = dth.abs() > 3e-2
            ghat = torch.where(moved, dy / torch.where(moved, dth, torch.ones_like(dth)),
                               torch.zeros_like(dy))
            freeze = moved & ((ghat * self.slope_sign) < self.slope_floor)

        # Conditional integration (anti-windup): freeze the integrator when the guard
        # trips or the command is saturated at a band edge, but keep the proportional
        # term regulating. We never yank theta back to nominal (that caused a limit
        # cycle); the band clamp alone keeps us inside the monotone region.
        at_edge = ((self.theta <= self.lo + 1e-6) & (eps * self.slope_sign < 0)) | \
                  ((self.theta >= self.hi - 1e-6) & (eps * self.slope_sign > 0))
        hold = freeze | at_edge
        self.sigma = torch.where(
            hold, self.sigma,
            torch.clamp(self.sigma + eps, -self.sig_max, self.sig_max),
        )
        raw = self.theta_nom + self.slope_sign * (self.Kp * eps + self.Ki * self.sigma)
        theta_new = torch.clamp(raw, self.lo, self.hi)

        self.prev_y = y
        self.prev_theta = self.theta
        self.theta = theta_new
        return theta_new
