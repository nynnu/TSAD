"""DINOv2-based vision encoder for anomaly detection.

Drop-in replacement for CLIP_AD, sharing the same encode_image interface.
Uses DINOv2 ViT-B-14 (patch_size=14, embed_dim=768) loaded via torch.hub.
"""

import torch
import torch.nn as nn
from models.model_utils import patch_scale
from models.clip_vision import multi_scale_aggregation


class DINO_AD(nn.Module):
    """DINOv2 vision encoder with multi-scale patch aggregation.

    Parameters
    ----------
    model_name : str
        torch.hub model name, e.g. 'dinov2_vitb14'
    device : torch.device, optional
        Defaults to CUDA if available.
    image_size : tuple
        (H, W) of input images; must be divisible by patch_size (14).
    """

    def __init__(self, model_name: str = 'dinov2_vitb14', device=None, image_size=(224, 224)):
        super(DINO_AD, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.image_size = image_size

        print(f"Loading DINOv2 model: {model_name} ...")
        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mask_helper = patch_scale(image_size=image_size)

    def encode_image(self, image: torch.Tensor, patch_size: int, use_mask: bool = True):
        """Extract multi-scale patch tokens from DINOv2.

        Parameters
        ----------
        image : torch.Tensor
            Shape [B, C, H, W].
        patch_size : int
            ViT patch size (14 for ViT-B-14).
        use_mask : bool
            Whether to compute multi-scale pooled tokens.

        Returns
        -------
        large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens,
        large_mask, mid_mask  —  same structure as CLIP_AD.encode_image.
        """
        image = image.to(self.device, non_blocking=True)

        with torch.no_grad():
            out = self.model.forward_features(image)

        # forward_features returns dict:
        #   x_norm_clstoken   : [B, D]
        #   x_norm_patchtokens: [B, N, D]  where N = (H/patch_size)^2
        patch_tokens = out['x_norm_patchtokens']  # [B, N, D]

        if use_mask:
            # kernel_size=48 → 48//14=3 patches per group (3×3)
            # kernel_size=32 → 32//14=2 patches per group (2×2)
            # stride_size=patch_size=14 → stride=1 in patch space
            # Same integer-division logic as CLIP_AD (patch_size=16 → 48//16=3, 32//16=2)
            large_scale_tokens, mid_scale_tokens, large_mask, mid_mask = multi_scale_aggregation(
                patch_tokens, patch_size, self.mask_helper
            )
        else:
            large_scale_tokens = mid_scale_tokens = large_mask = mid_mask = None

        class_tokens = patch_tokens.mean(dim=1, keepdim=True)   # [B, 1, D]
        patch_tokens = patch_tokens.unsqueeze(2)                 # [B, N, 1, D]

        return large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_mask, mid_mask
