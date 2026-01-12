import abc
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b5
from .layers import MHSIU, RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from einops import rearrange

LOGGER = logging.getLogger("main")

# =========================================================
# NOVELTY 1: CROSS-SCALE SELECTIVE SCAN (MAMBA-LIKE)
# Replaces MHSIU with Dynamic Global Context Fusion
# =========================================================
class CrossScaleSelectiveScan(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_in = ConvBNReLU(dim * 3, dim, 1)
        
        # We use GRU to approximate Mamba's "Selective Scan" logic
        # This captures long-range dependencies across the image
        self.scan_h = nn.GRUCell(dim, dim)
        self.scan_w = nn.GRUCell(dim, dim)
        
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.proj_out = ConvBNReLU(dim, dim, 1)

    def forward(self, l, m, s):
        # 1. Align & Merge
        tgt_size = m.shape[2:]
        l = F.interpolate(l, size=tgt_size, mode='bilinear')
        s = F.interpolate(s, size=tgt_size, mode='bilinear')
        x = torch.cat([l, m, s], dim=1)
        x = self.proj_in(x) 
        
        b, c, h, w = x.shape
        
        # 2. SELECTIVE SCAN (Height & Width)
        # Scan H
        x_perm = x.permute(0, 3, 2, 1).reshape(-1, h, c) # [B*W, H, C]
        h_state = torch.zeros(b*w, c, device=x.device)
        out_h = []
        for t in range(h):
            h_state = self.scan_h(x_perm[:, t, :], h_state)
            out_h.append(h_state)
        out_h = torch.stack(out_h, dim=1).reshape(b, w, h, c).permute(0, 3, 2, 1)
        
        # Scan W
        x_perm = x.permute(0, 2, 3, 1).reshape(-1, w, c) # [B*H, W, C]
        w_state = torch.zeros(b*h, c, device=x.device)
        out_w = []
        for t in range(w):
            w_state = self.scan_w(x_perm[:, t, :], w_state)
            out_w.append(w_state)
        out_w = torch.stack(out_w, dim=1).reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 3. Fuse & Gate
        scanned = out_h + out_w
        gated = scanned * self.gate(scanned)
        
        return self.proj_out(gated) + x # Residual

# =========================================================
# NOVELTY 2: SPECTRAL MoE REFINEMENT
# =========================================================
class SpectralRouter(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(dim * 2, num_experts)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer('laplacian', torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32).view(1,1,3,3))

    def forward(self, x):
        b, c, h, w = x.shape
        with torch.no_grad():
            weight = self.laplacian.expand(c, 1, 3, 3).to(x.device)
            high_freq = F.conv2d(x, weight, padding=1, groups=c)
        global_ctx = self.pool(x).flatten(1)
        freq_ctx = self.pool(high_freq.abs()).flatten(1)
        combined = torch.cat([global_ctx, freq_ctx], dim=1)
        return self.fc(combined).view(b, -1, 1, 1)

class ConvExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = ConvBNReLU(dim, dim, 3, 1, 1)
    def forward(self, x): return self.conv(x)

class MambaMoE(nn.Module):
    def __init__(self, in_dim, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.aux_loss = 0.0
        
        # STEP 1: Mamba Fusion (Replaces MHSIU)
        self.context_aggregator = CrossScaleSelectiveScan(in_dim)
        
        # STEP 2: MoE Refinement
        self.experts = nn.ModuleList([ConvExpert(in_dim) for _ in range(num_experts)])
        self.router = SpectralRouter(in_dim, num_experts)
        self.fuse = ConvBNReLU(in_dim, in_dim, 1)

    def forward(self, l, m, s):
        # 1. Global Context via Mamba
        x = self.context_aggregator(l, m, s)
        
        # 2. Local Detail via Spectral MoE
        router_logits = self.router(x)
        if self.training:
            probs = F.softmax(router_logits.flatten(2), dim=1)
            density = probs.mean(dim=0)
            target = torch.ones_like(density) / self.num_experts
            self.aux_loss = F.mse_loss(density.mean(1), target) * 10.0
        else:
            self.aux_loss = 0.0
            
        weights = F.softmax(router_logits, dim=1)
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        output = (expert_out * weights.unsqueeze(2)).sum(dim=1)
        
        return self.fuse(output)

# =========================================================
# STANDARD CLASSES
# =========================================================
class _ZoomNeXt_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)): return 1.0 
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__()
        self.num_frames = num_frames; self.pretrained = pretrained; self.use_checkpoint = use_checkpoint
    @abc.abstractmethod
    def encoder(self, image): pass
    @abc.abstractmethod
    def body(self, data): pass
    def forward(self, data):
        if self.training:
            logits = self.body(data); mask = data["mask"]
            if logits.shape[-2:] != mask.shape[-2:]: logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            return dict(loss=F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean"))
        else: return dict(pred=self.body(data).sigmoid())

class ZoomNeXt(_ZoomNeXt_Base):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint)
        self.normalize_encoder = PixelNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def body(self, data): pass 
    def encoder(self, x): pass

# Baseline Wrappers
class PvtV2_ZoomNeXt(ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False, variant='b2'):
        super().__init__(num_frames, pretrained, use_checkpoint)
        if variant == 'b2': self.backbone = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)
        elif variant == 'b5': self.backbone = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
        embed_dims = [64, 128, 320, 512]; out_dim = 64 
        self.tra_5 = ConvBNReLU(embed_dims[3], out_dim, 1); self.tra_4 = ConvBNReLU(embed_dims[2], out_dim, 1)
        self.tra_3 = ConvBNReLU(embed_dims[1], out_dim, 1); self.tra_2 = ConvBNReLU(embed_dims[0], out_dim, 1) 
        self.siu_5 = MHSIU(out_dim, num_groups=4); self.siu_4 = MHSIU(out_dim, num_groups=4); self.siu_3 = MHSIU(out_dim, num_groups=4)
        self.hmu_5 = SimpleASPP(out_dim, out_dim); self.hmu_4 = SimpleASPP(out_dim, out_dim); self.hmu_3 = SimpleASPP(out_dim, out_dim)
        self.rgpu = RGPU(in_dim=out_dim, out_dim=out_dim); self.proj = nn.Conv2d(out_dim, 1, kernel_size=3, padding=1)
    def encoder(self, x):
        feats = self.backbone(x)
        if isinstance(feats, dict): feats = list(feats.values())
        return [None] + feats 

class PvtV2B2_ZoomNeXt(PvtV2_ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False): super().__init__(num_frames, pretrained, use_checkpoint, variant='b2')
class PvtV2B5_ZoomNeXt(PvtV2_ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False): super().__init__(num_frames, pretrained, use_checkpoint, variant='b5')
class PvtV2B3_ZoomNeXt(PvtV2_ZoomNeXt): pass 
class PvtV2B4_ZoomNeXt(PvtV2_ZoomNeXt): pass 

# Placeholders
class CamoXpertV2(ZoomNeXt): pass 
class CamoXpertV3(ZoomNeXt): pass
class CamoXpertV4(ZoomNeXt): pass
class CamoXpertV5(ZoomNeXt): pass
class CamoXpertV6(ZoomNeXt): pass
class CamoXpertV7(ZoomNeXt): pass
class CamoXpertV8(ZoomNeXt): pass
class CamoXpertV9(ZoomNeXt): pass
class CamoXpertV10(ZoomNeXt): pass
class RN50_ZoomNeXt(ZoomNeXt): pass
class EffB1_ZoomNeXt(ZoomNeXt): pass
class EffB4_ZoomNeXt(ZoomNeXt): pass
class videoPvtV2B5_ZoomNeXt(PvtV2B5_ZoomNeXt): pass

# =========================================================
# CAMOXPERT V11 (THE FINAL ANSWER)
# B5 Backbone + Mamba-MoE Decoder
# =========================================================
class CamoXpertV11(ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint)
        
        # 1. B5 BACKBONE (The Ferrari)
        self.backbone = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
        embed_dims = [64, 128, 320, 512]; out_dim = 64
        
        self.tra_5 = ConvBNReLU(embed_dims[3], out_dim, 1); self.tra_4 = ConvBNReLU(embed_dims[2], out_dim, 1)
        self.tra_3 = ConvBNReLU(embed_dims[1], out_dim, 1); self.tra_2 = ConvBNReLU(embed_dims[0], out_dim, 1)
        
        # 2. MAMBA + MoE DECODER
        self.siu_5 = MambaMoE(out_dim); self.siu_4 = MambaMoE(out_dim); self.siu_3 = MambaMoE(out_dim)
        
        self.hmu_5 = SimpleASPP(out_dim, out_dim); self.hmu_4 = SimpleASPP(out_dim, out_dim); self.hmu_3 = SimpleASPP(out_dim, out_dim)
        self.rgpu = RGPU(out_dim, out_dim); self.proj = nn.Conv2d(out_dim, 1, 3, 1, 1)

    def encoder(self, x):
        feats = self.backbone(x)
        if isinstance(feats, dict): feats = list(feats.values())
        return [None] + feats 

    def body(self, data):
        l_f = self.encoder(self.normalize_encoder(data["image_l"]))
        m_f = self.encoder(self.normalize_encoder(data["image_m"]))
        s_f = self.encoder(self.normalize_encoder(data["image_s"]))
        
        l, m, s = self.tra_5(l_f[4]), self.tra_5(m_f[4]), self.tra_5(s_f[4])
        lms = self.siu_5(l, m, s); x = self.hmu_5(lms)
        
        l, m, s = self.tra_4(l_f[3]), self.tra_4(m_f[3]), self.tra_4(s_f[3])
        lms = self.siu_4(l, m, s); x = self.hmu_4(lms + resize_to(x, lms.shape[-2:]))
        
        l, m, s = self.tra_3(l_f[2]), self.tra_3(m_f[2]), self.tra_3(s_f[2])
        lms = self.siu_3(l, m, s); x = self.hmu_3(lms + resize_to(x, lms.shape[-2:]))
        
        l = self.tra_2(l_f[1])
        x = self.rgpu(l, x)
        return self.proj(x)

    def forward(self, data):
        if self.training:
            logits = self.body(data); mask = data["mask"]
            if logits.shape[-2:] != mask.shape[-2:]: logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear')
            loss = F.binary_cross_entropy_with_logits(logits, mask)
            aux = self.siu_5.aux_loss + self.siu_4.aux_loss + self.siu_3.aux_loss
            return dict(loss=loss + 0.1 * aux)
        else:
            return dict(pred=self.body(data).sigmoid())