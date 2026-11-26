"""CamoXpert: one class, switches for every component (replaces CamoXpertV11 / V12 / V12_Progressive).

    CamoXpert-Base (legacy v11) = CamoXpert(asw=False, brm=False, progressive_iters=0)
    CamoXpert      (legacy v12) = CamoXpert(asw=True,  brm=True,  progressive_iters=0)
    CamoXpert-PR   (legacy v12-Progressive) = CamoXpert(asw=True, brm=True, progressive_iters=2)
The loss is chosen in the config (camoxpert.losses), not inside the model.

Forward pass (input: one RGB batch `image` in [0, 1], size S x S, S = 448 by default):
    1. Input pyramid: l = image (S), m = resize(image, 0.75 S), s = resize(image, 0.5 S).
    2. Shared PVTv2-B5 encoder on each of l, m, s -> 4 stages per scale with strides 4, 8, 16, 32
       and 64, 128, 320, 512 channels. (Same weights for the three scales.)
    3. Stages 5, 4, 3 (strides 32, 16, 8): 1x1 conv to 64 channels, [ASW], ScanMoEFusion
       (merges l, m, s at m's resolution), DilatedContextBlock; deeper result is upsampled and
       added before the block (top-down path).
    4. Stage 2 (stride 4, large scale only): LowLevelGuidance.
    5. [BRM] -> head: 3x3 conv to 1 channel (or ProgressiveRefinement heads).
    The logits come out at stride 4 of the large input (112 x 112 for S = 448); the loss and the
    evaluation upsample them bilinearly.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b5
from .camoxpert_layers import (
    AdaptiveScaleWeighting,
    BoundaryRefinementModule,
    DilatedContextBlock,
    LowLevelGuidance,
    ProgressiveRefinement,
    ScanMoEFusion,
)
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("camoxpert")


def build_pvt_v2_b5(pretrained=True, pretrained_path=None, use_checkpoint=False):
    """PVTv2-B5 with ImageNet-1k weights.

    pretrained_path: local copy of pvt_v2_b5.pth (use this on a machine without internet).
    Otherwise, pretrained=True downloads it from the official PVT GitHub release via torch.hub.
    """
    if pretrained_path:
        model = pvt_v2_eff_b5(pretrained=False, use_checkpoint=use_checkpoint)
        state = torch.load(pretrained_path, map_location="cpu")
        state.pop("head.weight", None)
        state.pop("head.bias", None)
        model.load_state_dict(state, strict=True)
        LOGGER.info(f"Loaded PVTv2-B5 weights from {pretrained_path}")
        return model
    return pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


def make_pyramid(image, scales):
    """Resize the input to each relative scale (bilinear, align_corners=False). Same code in
    training and evaluation (the legacy code used align_corners=True in training only)."""
    out = []
    for s in scales:
        if s == 1.0:
            out.append(image)
        else:
            out.append(F.interpolate(image, scale_factor=s, mode="bilinear", align_corners=False))
    return out


class CamoXpert(nn.Module):
    backbone_prefix = "backbone."  # parameters under this prefix get the backbone learning rate

    def __init__(
        self,
        pretrained=True,
        pretrained_path=None,
        asw=True,
        brm=True,
        progressive_iters=0,
        use_scan=True,
        use_moe=True,
        num_experts=2,
        scales=(1.0, 0.75, 0.5),
        mid_dim=64,
        use_checkpoint=False,
    ):
        super().__init__()
        assert len(scales) == 3 and scales[0] >= scales[1] >= scales[2], "scales = (large, medium, small)"
        self.scales = tuple(scales)
        self.use_asw = asw
        self.use_brm = brm
        self.progressive_iters = progressive_iters

        self.backbone = build_pvt_v2_b5(pretrained, pretrained_path, use_checkpoint)
        self.normalize_encoder = PixelNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        embed_dims = [64, 128, 320, 512]

        self.tra_5 = ConvBNReLU(embed_dims[3], mid_dim, 1)
        self.tra_4 = ConvBNReLU(embed_dims[2], mid_dim, 1)
        self.tra_3 = ConvBNReLU(embed_dims[1], mid_dim, 1)
        self.tra_2 = ConvBNReLU(embed_dims[0], mid_dim, 1)

        if asw:
            self.asw_5 = AdaptiveScaleWeighting(mid_dim)
            self.asw_4 = AdaptiveScaleWeighting(mid_dim)
            self.asw_3 = AdaptiveScaleWeighting(mid_dim)

        fusion_kw = dict(num_experts=num_experts, use_scan=use_scan, use_moe=use_moe)
        self.siu_5 = ScanMoEFusion(mid_dim, **fusion_kw)
        self.siu_4 = ScanMoEFusion(mid_dim, **fusion_kw)
        self.siu_3 = ScanMoEFusion(mid_dim, **fusion_kw)

        self.hmu_5 = DilatedContextBlock(mid_dim, mid_dim)
        self.hmu_4 = DilatedContextBlock(mid_dim, mid_dim)
        self.hmu_3 = DilatedContextBlock(mid_dim, mid_dim)

        self.rgpu = LowLevelGuidance(mid_dim, mid_dim)

        if brm:
            self.boundary_refine = BoundaryRefinementModule(mid_dim)
        if progressive_iters > 0:
            self.progressive_refine = ProgressiveRefinement(mid_dim, progressive_iters)
        else:
            self.proj = nn.Conv2d(mid_dim, 1, 3, 1, 1)

    def encode(self, x):
        """Backbone features as a list [c2, c3, c4, c5] (strides 4, 8, 16, 32)."""
        feats = self.backbone(self.normalize_encoder(x))
        return [feats["reduction_2"], feats["reduction_3"], feats["reduction_4"], feats["reduction_5"]]

    def _stage(self, idx, feats_l, feats_m, feats_s, prev):
        tra = getattr(self, f"tra_{idx}")
        k = idx - 2  # stage 5 -> c5 = list index 3
        l, m, s = tra(feats_l[k]), tra(feats_m[k]), tra(feats_s[k])
        if self.use_asw:
            l, m, s = getattr(self, f"asw_{idx}")(l, m, s)
        lms, aux = getattr(self, f"siu_{idx}")(l, m, s)
        if prev is not None:
            lms = lms + resize_to(prev, lms.shape[-2:])
        return getattr(self, f"hmu_{idx}")(lms), aux

    def forward(self, image):
        """image: [B, 3, S, S] in [0, 1]. Returns a dict of logits (no sigmoid) and the aux loss:
        logits      [B, 1, S/4, S/4]  final prediction
        side_logits list of all progressive predictions (empty without progressive refinement)
        edge_logits [B, 1, S/4, S/4] or None
        aux_loss    0-dim tensor, sum of the three MoE load-balancing losses (0 in eval)
        """
        img_l, img_m, img_s = make_pyramid(image, self.scales)
        f_l, f_m, f_s = self.encode(img_l), self.encode(img_m), self.encode(img_s)

        x, aux5 = self._stage(5, f_l, f_m, f_s, None)
        x, aux4 = self._stage(4, f_l, f_m, f_s, x)
        x, aux3 = self._stage(3, f_l, f_m, f_s, x)
        x = self.rgpu(self.tra_2(f_l[0]), x)

        edge_logits = None
        if self.use_brm:
            x, edge_logits = self.boundary_refine(x)

        if self.progressive_iters > 0:
            side = self.progressive_refine(x)
            logits = side[-1]
        else:
            side = []
            logits = self.proj(x)
        return dict(logits=logits, side_logits=side, edge_logits=edge_logits, aux_loss=aux5 + aux4 + aux3)
