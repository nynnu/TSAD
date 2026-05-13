"""Local Temporal Reference utilities — v2 (batch-aware).

Identical interface to model_utils_local.py but build_ordered_embeddings
processes the full DataLoader batch in one GPU forward pass instead of
iterating image-by-image.

Speedup on GPU: ~batch_size× (default batch_size=20 → up to 20× faster
for Pass 1 encoding).

No existing files modified.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pass 1: batch-aware encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_ordered_embeddings(model, dataloader, patch_size, device):
    """Encode every window in batches and return embeddings sorted by window_id.

    v2 change: calls model.encode_image(imgs_batch, patch_size) once per
    DataLoader batch instead of once per image — fully utilises GPU parallelism.

    Parameters / Returns: identical to model_utils_local.build_ordered_embeddings
    """
    store = {"large": [], "mid": [], "patch": [], "wid": []}
    large_mask = mid_mask = None

    for batch in dataloader:
        imgs = batch["img"].to(device)          # [B, C, H, W]
        wids = batch["window_id"]
        B    = imgs.shape[0]

        # --- single forward pass for the whole batch ---
        L_tok, M_tok, P_tok, _, lmask, mmask = model.encode_image(imgs, patch_size)
        # L_tok : [B, N_L, D]
        # M_tok : [B, N_M, D]
        # P_tok : [B, N, 1, D]  (patch_tokens.unsqueeze(2) in encode_image)

        if large_mask is None:
            large_mask = lmask
            mid_mask   = mmask

        for i in range(B):
            store["large"].append(L_tok[i:i+1].cpu())             # [1, N_L, D]
            store["mid"].append(M_tok[i:i+1].cpu())               # [1, N_M, D]
            store["patch"].append(P_tok[i:i+1].squeeze(2).cpu())  # [1, N,   D]
            store["wid"].append(wids[i].item())

    order = sorted(range(len(store["wid"])), key=lambda i: store["wid"][i])

    large_embeds = torch.cat([store["large"][i] for i in order], dim=0)
    mid_embeds   = torch.cat([store["mid"][i]   for i in order], dim=0)
    patch_embeds = torch.cat([store["patch"][i] for i in order], dim=0)
    window_ids   = [store["wid"][i] for i in order]

    return large_embeds, mid_embeds, patch_embeds, large_mask, mid_mask, window_ids


# ---------------------------------------------------------------------------
# get_local_reference / compute_dissimilarity_with_ref — unchanged from v1
# ---------------------------------------------------------------------------

def get_local_reference(embeds: torch.Tensor, i: int, k: int, min_ref: int = 5):
    L   = embeds.shape[0]
    idx = list(range(max(0, i - k), i)) + list(range(i + 1, min(L, i + k + 1)))

    if len(idx) < min_ref:
        idx      = list(range(0, i)) + list(range(i + 1, L))
        is_local = False
    else:
        is_local = True

    local = embeds[idx]
    return torch.median(local, dim=0).values, is_local


def compute_dissimilarity_with_ref(token: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if token.ndim == 4 and token.shape[2] == 1:
        token = token.squeeze(2)

    token_norm = F.normalize(token, dim=-1)
    ref_norm   = F.normalize(ref.unsqueeze(0), dim=-1)

    sim    = torch.bmm(token_norm, ref_norm.permute(0, 2, 1))
    dissim = 1.0 - sim
    return 0.5 * torch.min(dissim, dim=2).values
