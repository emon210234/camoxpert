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

