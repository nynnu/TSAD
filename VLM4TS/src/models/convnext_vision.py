"""ConvNeXt V2 vision encoder for anomaly detection.

Drop-in replacement for CLIP_AD / DINO_AD.

ConvNeXt V2 is a CNN (no explicit patch tokens), so we extract the stage-2
feature map [B, 512, 14, 14] and treat each spatial cell as a "patch token",
giving 14×14 = 196 tokens — same count as CLIP ViT-B-16.

Virtual patch_size = 224 // 14 = 16, so the multi-scale aggregation kernel
sizes (48, 32) produce the same 3×3 and 2×2 patch groups as for CLIP.

Requirements:
    pip install timm
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e

from models.model_utils import patch_scale
from models.clip_vision import multi_scale_aggregation


class ConvNeXtV2_AD(nn.Module):
    """ConvNeXt V2 Base encoder with multi-scale spatial aggregation.

    Parameters
    ----------
    model_name : str
        timm model name. Default uses FCMAE pretrained + IN-22k/1k fine-tuning.
    device : torch.device, optional
    image_size : tuple
        (H, W); must be 224×224 (stage-2 output = 14×14 for this input size).
    """

    # Stage-2 output: [B, 512, 14, 14]  → 196 tokens, virtual patch_size = 16
    VIRTUAL_PATCH_SIZE = 16

    def __init__(
        self,
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        device=None,
        image_size=(224, 224),
    ):
        super(ConvNeXtV2_AD, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.image_size = image_size

        print(f"Loading ConvNeXt V2 model: {model_name} ...")
        # features_only=True → returns list of feature maps per stage
        # out_indices=[2]    → only stage 2 (14×14 with 224px input)
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=[2],
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mask_helper = patch_scale(image_size=image_size)

    def encode_image(self, image: torch.Tensor, patch_size: int, use_mask: bool = True):
        """Extract multi-scale spatial tokens from ConvNeXt V2 stage-2 feature map.

        Parameters
        ----------
        image : torch.Tensor  [B, C, H, W]
        patch_size : int
            Virtual patch size (should be 16 = 224 // 14 for ConvNeXt V2 Base).
        use_mask : bool

        Returns
        -------
        Same 6-tuple as CLIP_AD.encode_image:
            large_scale_tokens, mid_scale_tokens, patch_tokens,
            class_tokens, large_mask, mid_mask
        """
        image = image.to(self.device, non_blocking=True)

        with torch.no_grad():
            feats = self.model(image)   # list with one element for out_indices=[2]

        feat_map = feats[0]             # [B, 512, 14, 14]
        B, C, H, W = feat_map.shape

        # Flatten spatial dims → treat each cell as a patch token
        # [B, C, H, W] → [B, H*W, C]
        patch_tokens = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if use_mask:
            # kernel_size=48 → 48//16=3 → 3×3 cell groups
            # kernel_size=32 → 32//16=2 → 2×2 cell groups
            large_scale_tokens, mid_scale_tokens, large_mask, mid_mask = multi_scale_aggregation(
                patch_tokens, patch_size, self.mask_helper
            )
        else:
            large_scale_tokens = mid_scale_tokens = large_mask = mid_mask = None

        class_tokens = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
        patch_tokens = patch_tokens.unsqueeze(2)               # [B, N, 1, C]

        return large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_mask, mid_mask
