"""MAE (Masked Autoencoder) vision encoder for anomaly detection.

Drop-in replacement for CLIP_AD / DINO_AD / ConvNeXtV2_AD.

Uses timm's ViT-B/16 with MAE pretrained weights as a pure feature extractor
(no reconstruction). forward_features() returns [B, 197, 768]; we strip the
CLS token to get 196 patch tokens of dim 768 — identical to CLIP ViT-B/16.

Because patch_size=16 and embed_dim=768 match CLIP exactly:
  - Same 14×14 patch grid  → no grid structure changes
  - Same multi-scale kernel sizes (48→3×3, 32→2×2)
  - No projection layer needed
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e

from models.model_utils import patch_scale
from models.clip_vision import multi_scale_aggregation


class MAE_AD(nn.Module):
    """MAE ViT-B/16 encoder used as a frozen feature extractor.

    Parameters
    ----------
    model_name : str
        timm model name. Default: 'vit_base_patch16_224.mae'
    device : torch.device, optional
    image_size : tuple  (H, W); must be 224×224.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.mae",
        device=None,
        image_size=(224, 224),
    ):
        super(MAE_AD, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.image_size = image_size

        print(f"Loading MAE model: {model_name} ...")
        self.model = timm.create_model(model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # ViT-B/16 has a CLS token prepended; patch tokens start at index 1
        self._has_cls = self.model.cls_token is not None

        self.mask_helper = patch_scale(image_size=image_size)

    def encode_image(self, image: torch.Tensor, patch_size: int, use_mask: bool = True):
        """Extract multi-scale patch tokens from MAE encoder.

        MAE forward_features → [B, 197, 768]  (CLS + 196 patch tokens)
        After stripping CLS → [B, 196, 768]  — same as CLIP ViT-B/16.

        Parameters
        ----------
        image : torch.Tensor  [B, C, H, W]
        patch_size : int  (16 for ViT-B/16)
        use_mask : bool

        Returns
        -------
        Same 6-tuple as CLIP_AD.encode_image.
        """
        image = image.to(self.device, non_blocking=True)

        with torch.no_grad():
            out = self.model.forward_features(image)  # [B, 197, 768]

        # Strip CLS token → patch tokens [B, 196, 768]
        patch_tokens = out[:, 1:, :] if self._has_cls else out

        if use_mask:
            # Identical kernel sizes as CLIP: 48//16=3 (3×3), 32//16=2 (2×2)
            large_scale_tokens, mid_scale_tokens, large_mask, mid_mask = multi_scale_aggregation(
                patch_tokens, patch_size, self.mask_helper
            )
        else:
            large_scale_tokens = mid_scale_tokens = large_mask = mid_mask = None

        class_tokens = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, 768]
        patch_tokens = patch_tokens.unsqueeze(2)               # [B, 196, 1, 768]

        return large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_mask, mid_mask
