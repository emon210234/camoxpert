"""ZoomNeXt with a PVTv2-B5 encoder: the baseline for run R0.

The network (`_UpstreamPvtV2ZoomNeXt`) is copied from ZoomNeXt
(https://github.com/lartpang/ZoomNeXt, commit 614af43, methods/zoomnext/zoomnext.py, classes
PvtV2B2_ZoomNeXt / PvtV2B5_ZoomNeXt). Only two things differ, both outside the network:
  * `ZoomNeXtB5.forward` takes one image and builds the input pyramid itself, like CamoXpert,
    and returns the same output dict as CamoXpert, so train/eval scripts treat both the same.
  * The loss (BCE + UAL) lives in camoxpert/losses.py; upstream computed it inside forward().

The legacy repo's `PvtV2B5_ZoomNeXt` was NOT ZoomNeXt: it had no body() (it returned None) and
used simplified MHSIU/RGPU/SimpleASPP blocks. See CHANGES.md, finding C8.

Upstream input pyramid: s, m, l = 0.5x, 1.0x, 1.5x of a 384 x 384 base, i.e. 192 / 384 / 576.
With our convention (the input is the LARGE scale) that is image = 576, scales (1, 2/3, 1/3).
"""

import torch.nn as nn

from .camoxpert import build_pvt_v2_b5, make_pyramid
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from .zoomnext_layers import MHSIU, RGPU, SimpleASPP


class _UpstreamPvtV2ZoomNeXt(nn.Module):
    """Upstream PvtV2B2_ZoomNeXt (verbatim apart from the encoder factory call)."""

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        pretrained_path=None,
    ):
        super().__init__()
        self.encoder = build_pvt_v2_b5(pretrained, pretrained_path, use_checkpoint)

        self.embed_dims = self.encoder.embed_dims
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = self.tra_5(l_trans_feats[3]), self.tra_5(m_trans_feats[3]), self.tra_5(s_trans_feats[3])
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2]), self.tra_4(s_trans_feats[2])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1]), self.tra_3(s_trans_feats[1])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0]), self.tra_2(s_trans_feats[0])
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        x = self.tra_1(x)
        return self.predictor(x)


class ZoomNeXtB5(_UpstreamPvtV2ZoomNeXt):
    backbone_prefix = "encoder."

    def __init__(self, pretrained=True, pretrained_path=None, scales=(1.0, 2.0 / 3.0, 1.0 / 3.0), **kwargs):
        super().__init__(pretrained=pretrained, pretrained_path=pretrained_path, **kwargs)
        self.scales = tuple(scales)

    def forward(self, image):
        img_l, img_m, img_s = make_pyramid(image, self.scales)
        logits = self.body(dict(image_l=img_l, image_m=img_m, image_s=img_s))
        return dict(logits=logits, side_logits=[], edge_logits=None, aux_loss=logits.new_zeros(()))



class ZoomNeXtB5Plus(ZoomNeXtB5):
    """R8: ZoomNeXt-B5 + CamoXpert's AdaptiveScaleWeighting (ASW) and BoundaryRefinementModule (BRM).

    Everything of ZoomNeXtB5 is kept (encoder, tra_*, MHSIU siu_*, RGPU hmu_*, tra_1, predictor).
    Two optional insertions, the same modules (same classes) CamoXpert uses:
      * asw=True : before EVERY multi-scale fusion (siu_5, siu_4, siu_3, siu_2), the three scale
                   features l, m, s are re-weighted per image: (l, m, s) <- (w_l l, w_m m, w_s s),
                   w = softmax(MLP(GAP)). (CamoXpert fuses at 3 stages, ZoomNeXt at 4, so ASW is at 4.)
      * brm=True : after hmu_2, before tra_1 and the predictor (64 channels; the same place in
                   the decoder as in CamoXpert), x <- BRM(x); BRM also returns edge logits that
                   the edge loss supervises. ZoomNeXt merges scales at the MEDIUM scale's
                   resolution, so here BRM runs at stride 4 of the medium input (84 x 84 for
                   S = 448), where CamoXpert's BRM runs at 112 x 112. The final logits are at
                   0.75 S (336 x 336), exactly as in R0; loss and evaluation resize them.
    With asw=False and brm=False no extra module is created, so the network (and its random
    initialisation for a given seed) is exactly ZoomNeXtB5 (checked in tests/test_units.py).
    """

    def __init__(self, asw=True, brm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_asw = asw
        self.use_brm = brm
        if asw:
            from .camoxpert_layers import AdaptiveScaleWeighting

            for k in (5, 4, 3, 2):
                setattr(self, f"asw_{k}", AdaptiveScaleWeighting(64))
        if brm:
            from .camoxpert_layers import BoundaryRefinementModule

            self.boundary_refine = BoundaryRefinementModule(64)

    def _fuse(self, k, feats_l, feats_m, feats_s, prev):
        tra = getattr(self, f"tra_{k}")
        i = k - 2
        l, m, s = tra(feats_l[i]), tra(feats_m[i]), tra(feats_s[i])
        if self.use_asw:
            l, m, s = getattr(self, f"asw_{k}")(l, m, s)
        lms = getattr(self, f"siu_{k}")(l=l, m=m, s=s)
        if prev is not None:
            lms = lms + resize_to(prev, tgt_hw=lms.shape[-2:])
        return getattr(self, f"hmu_{k}")(lms)

    def forward(self, image):
        img_l, img_m, img_s = make_pyramid(image, self.scales)
        f_l, f_m, f_s = self.normalize_encoder(img_l), self.normalize_encoder(img_m), self.normalize_encoder(img_s)
        x = None
        for k in (5, 4, 3, 2):
            x = self._fuse(k, f_l, f_m, f_s, x)
        edge_logits = None
        if self.use_brm:
            x, edge_logits = self.boundary_refine(x)
        logits = self.predictor(self.tra_1(x))
        return dict(logits=logits, side_logits=[], edge_logits=edge_logits, aux_loss=logits.new_zeros(()))
