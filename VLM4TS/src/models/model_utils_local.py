"""Local Temporal Reference utilities for ViT4TS.

Replaces the global-median reference in cross-patch comparison with a
temporally-local median built from the k nearest windows on each side.

All existing model_utils functions are untouched.
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pass 1: collect all window embeddings in temporal order
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_ordered_embeddings(model, dataloader, patch_size, device):
    """Encode every window and return embeddings sorted by window_id.

    Parameters
    ----------
    model : nn.Module  Vision encoder with encode_image() interface.
    dataloader : DataLoader  shuffle=False, yields {img, cls_name, window_id}.
    patch_size : int
    device : torch.device

    Returns
    -------
    large_embeds : Tensor [L, N_L, D]
    mid_embeds   : Tensor [L, N_M, D]
    patch_embeds : Tensor [L, N,   D]   (CLS dim squeezed)
    large_mask   : mask tensor (same for all windows)
    mid_mask     : mask tensor
    window_ids   : list[int] sorted temporal order
    """
    store = {"large": [], "mid": [], "patch": [], "wid": []}
    large_mask = mid_mask = None

    for batch in dataloader:
        imgs = batch["img"].to(device)
        wids = batch["window_id"]
        B = imgs.shape[0]

        for i in range(B):
            img = imgs[i].unsqueeze(0)
            L_tok, M_tok, P_tok, _, lmask, mmask = model.encode_image(img, patch_size)

            if large_mask is None:
                large_mask = lmask
                mid_mask   = mmask

            store["large"].append(L_tok.cpu())          # [1, N_L, D]
            store["mid"].append(M_tok.cpu())             # [1, N_M, D]
            store["patch"].append(P_tok.squeeze(2).cpu()) # [1, N, D]
            store["wid"].append(wids[i].item())

    # Sort by window_id
    order = sorted(range(len(store["wid"])), key=lambda i: store["wid"][i])

    large_embeds = torch.cat([store["large"][i] for i in order], dim=0)
    mid_embeds   = torch.cat([store["mid"][i]   for i in order], dim=0)
    patch_embeds = torch.cat([store["patch"][i] for i in order], dim=0)
    window_ids   = [store["wid"][i] for i in order]

    return large_embeds, mid_embeds, patch_embeds, large_mask, mid_mask, window_ids


# ---------------------------------------------------------------------------
# Local reference extraction
# ---------------------------------------------------------------------------

def get_local_reference(embeds: torch.Tensor, i: int, k: int, min_ref: int = 5):
    """Return the median embedding of the k-nearest temporal neighbors of window i.

    Parameters
    ----------
    embeds  : [L, N, D]
    i       : current window index (excluded from reference)
    k       : half-window size — uses windows in [i-k, i+k] \ {i}
    min_ref : if fewer than min_ref neighbors exist, fall back to global median

    Returns
    -------
    ref      : [N, D] median reference
    is_local : bool — True if local reference was used
    """
    L = embeds.shape[0]
    idx = list(range(max(0, i - k), i)) + list(range(i + 1, min(L, i + k + 1)))

    if len(idx) < min_ref:
        # global fallback (exclude self)
        idx = list(range(0, i)) + list(range(i + 1, L))
        is_local = False
    else:
        is_local = True

    local = embeds[idx]                          # [K, N, D]
    return torch.median(local, dim=0).values, is_local   # [N, D]


# ---------------------------------------------------------------------------
# Dissimilarity against a pre-computed reference
# ---------------------------------------------------------------------------

def compute_dissimilarity_with_ref(token: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cosine dissimilarity between one window's token and a reference.

    Parameters
    ----------
    token : [1, N, D]  current window embedding
    ref   : [N, D]     pre-computed reference (e.g. local median)

    Returns
    -------
    M : [1, N]  0.5 * min-dissimilarity per patch position
    """
    if token.ndim == 4 and token.shape[2] == 1:
        token = token.squeeze(2)    # [1, N, D]

    token_norm = F.normalize(token, dim=-1)          # [1, N, D]
    ref_norm   = F.normalize(ref.unsqueeze(0), dim=-1)  # [1, N, D]

    sim    = torch.bmm(token_norm, ref_norm.permute(0, 2, 1))  # [1, N, N]
    dissim = 1.0 - sim
    M = 0.5 * torch.min(dissim, dim=2).values        # [1, N]
    return M
