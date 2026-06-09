"""Shared behavioral observables read off the model's output distribution.

A behavioral observable maps next-token logits to a continuous, bounded scalar
that reflects the model's *output* behavior (not an internal coordinate). Used by
multiple research proposals (PTS, and originally CLAS). Kept in the shared library
so proposals depend on `pytorch_pure/`, not on each other.
"""

import torch
import torch.nn.functional as F


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
