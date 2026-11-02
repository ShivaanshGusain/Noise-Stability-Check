"""Noise-stability regularizer (paper Definition 6.1): penalty = -gamma * E[<M(X), M(Y)>], where Y replaces each
token of X by a uniformly random token with probability (1 - rho) / 2.

"original" repeats the released code (noise_stability/measure_noise_stability.py) exactly: it recomputes the clean
forward pass (two extra passes per step) and, for language-model outputs [batch, length, vocab], takes the softmax
over dim=1, i.e. over SEQUENCE POSITIONS, and sums the product over positions.
"fixed" follows Definition 6.1 per position: next-token distributions (softmax over the vocabulary), reusing the
logits of the training forward pass, so it costs one extra forward pass.
"""

import torch
import torch.nn.functional as F


def corrupt(ids, rho, vocab_size, generator=None):
    """Replace each token with probability (1 - rho) / 2 by a uniformly random token (as in the released code)."""
    replace = torch.rand(ids.shape, device=ids.device, generator=generator) > (1 + rho) / 2
    random = torch.randint(0, vocab_size, ids.shape, device=ids.device, generator=generator)
    return torch.where(replace, random, ids)


def stability(model, ids, logits, rho, vocab_size, mode, generator=None):
    """Differentiable noise-stability estimate for one batch (higher = more stable)."""
    noisy = model(corrupt(ids, rho, vocab_size, generator))
    if mode == "original":
        clean = model(ids)  # recomputed, as released
        return (F.softmax(clean, dim=1) * F.softmax(noisy, dim=1)).sum(dim=1).mean()
    if mode == "fixed":
        return (F.softmax(logits, dim=-1) * F.softmax(noisy, dim=-1)).sum(dim=-1).mean()
    raise ValueError(f"Unknown regularizer mode {mode!r}")
