"""Building blocks of CamoXpert.

Every block here reproduces, operation for operation, the block of the same role in the
pre-P2 code (methods/zoomnext/zoomnext.py and methods/zoomnext/improved_modules.py, archived in
_archive/camoxpert_pre_P2/). tools/verify_against_legacy.py checks that the outputs match
the legacy classes to float precision when both get the same weights. See CHANGES.md for the
few deliberate differences (dead parameters removed, edge head returns logits).

Notation used in the comments:
    B batch, C channels (64 in the decoder), H x W spatial size of the feature map.
    l, m, s = features of the large / medium / small input scale (same stage of the backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import ConvBNReLU, resize_to


# ---------------------------------------------------------------------------------------------
# Decoder blocks inherited from the pre-P2 code. They resemble, but are NOT identical to,
# ZoomNeXt's SimpleASPP / RGPU (see CHANGES.md, section "Naming"), so they get their own names.
# ---------------------------------------------------------------------------------------------
class DilatedContextBlock(nn.Module):
    """Legacy `SimpleASPP` of CamoXpert (the `hmu_*` modules).

    1x1 conv to 2C, split into y1 and y5; y1 goes through three chained 3x3 convs with
    dilation 3 (receptive field grows 7 -> 13 -> 19 pixels); concat [y5, y1, y2, y3, y4] (5C)
    -> 1x1 -> 3x3. Unlike ZoomNeXt's SimpleASPP there is no global-average-pool branch: y5 is
    used as is. (The legacy class also built an unused `conv1x1_2`; it is dropped here.)
    """

    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y1, y5 = self.conv1x1_1(x).chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        return self.fuse(torch.cat([y5, y1, y2, y3, y4], dim=1))


class LowLevelGuidance(nn.Module):
    """Legacy `RGPU` of CamoXpert (the `rgpu` module): fuse stride-4 features with the decoder.

    curr = conv(low-level features), prev = upsample(conv(decoder features))
    out  = conv([curr * prev, curr])     -- the decoder output gates the low-level detail.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_curr = ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        self.conv_prev = ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        self.conv_out = ConvBNReLU(2 * out_dim, out_dim, 3, 1, 1)

    def forward(self, curr_x, prev_x):
        curr = self.conv_curr(curr_x)
        prev = resize_to(self.conv_prev(prev_x), tgt_hw=curr.shape[2:])
        return self.conv_out(torch.cat([curr * prev, curr], dim=1))


# ---------------------------------------------------------------------------------------------
# Cross-scale fusion: directional recurrent scan + image-level mixture of two conv experts.
# (Legacy names: CrossScaleSelectiveScan, SpectralRouter, ConvExpert, MambaMoE.)
# ---------------------------------------------------------------------------------------------
class RecurrentScanFusion(nn.Module):
    """Merge l/m/s at the m resolution, then add context from two GRU scans.

    1) l and s are bilinearly resized to m's size; x = 1x1conv(concat[l, m, s])  (3C -> C).
    2) Column scan: every column of x is a sequence of H vectors (length C); a GRUCell walks it
       top -> bottom. Row scan: every row is a sequence of W vectors; a second GRUCell walks it
       left -> right. Only these two directions exist (no reverse scans).
       GRU update for one step (z: update gate, r: reset gate, sigma: sigmoid):
           z = sigma(W_z x_t + U_z h + b_z)          r = sigma(W_r x_t + U_r h + b_r)
           n = tanh(W_n x_t + r * (U_n h + b_n))      h_new = (1 - z) * n + z * h
       So the hidden state at pixel (i, j) summarises everything above it (column scan) or to
       its left (row scan). Cost is linear in the number of pixels, but the steps run one after
       another in a Python loop, so wall-clock speed must be measured (tools/profile_model.py).
    3) y = h_col + h_row; y = y * sigmoid(1x1conv(y)); out = 1x1conv(y) + x  (residual).

    use_scan=False (ablation) skips steps 2-3 and returns x.
    """

    def __init__(self, dim, use_scan=True):
        super().__init__()
        self.dim = dim
        self.use_scan = use_scan
        self.proj_in = ConvBNReLU(dim * 3, dim, 1)
        if use_scan:
            self.scan_h = nn.GRUCell(dim, dim)  # runs along H (down each column)
            self.scan_w = nn.GRUCell(dim, dim)  # runs along W (along each row)
            self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
            self.proj_out = ConvBNReLU(dim, dim, 1)

    @staticmethod
    def _scan(cell, seq):
        """seq: [N, T, C] -> hidden states [N, T, C], starting from h = 0.

        Runs in float32 even under bf16/fp16 autocast: the hidden state is updated up to 42 times
        in a row (stage 3 at 336 px), and low-precision rounding would accumulate along the scan.
        The loop is limited by kernel-launch overhead, so float32 costs little time here."""
        n, t_len, c = seq.shape
        with torch.autocast(device_type=seq.device.type, enabled=False):
            seq = seq.float()
            h = seq.new_zeros(n, c)
            outs = []
            for t in range(t_len):
                h = cell(seq[:, t, :], h)
                outs.append(h)
            return torch.stack(outs, dim=1)

    def forward(self, l, m, s):
        tgt = m.shape[2:]
        l = F.interpolate(l, size=tgt, mode="bilinear")
        s = F.interpolate(s, size=tgt, mode="bilinear")
        x = self.proj_in(torch.cat([l, m, s], dim=1))
        if not self.use_scan:
            return x

        b, c, h, w = x.shape
        # columns: [B, C, H, W] -> [B, W, H, C] -> [B*W, H, C]
        col = self._scan(self.scan_h, x.permute(0, 3, 2, 1).reshape(-1, h, c))
        col = col.reshape(b, w, h, c).permute(0, 3, 2, 1)
        # rows:    [B, C, H, W] -> [B, H, W, C] -> [B*H, W, C]
        row = self._scan(self.scan_w, x.permute(0, 2, 3, 1).reshape(-1, w, c))
        row = row.reshape(b, h, w, c).permute(0, 3, 1, 2)

        y = col + row
        y = y * self.gate(y)
        return self.proj_out(y) + x


class SpectralRouter(nn.Module):
    """Image-level router: one weight vector per image (not per pixel).

    hf = |Laplacian(x)| applied per channel with the fixed kernel [[0,-1,0],[-1,4,-1],[0,-1,0]]
    (a high-pass filter: large where the feature map changes quickly, i.e. texture/edges).
    logits = Linear(concat[GAP(x), GAP(hf)])  -> [B, E, 1, 1].  GAP = global average pool.
    """

    def __init__(self, dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(dim * 2, num_experts)
        self.pool = nn.AdaptiveAvgPool2d(1)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", lap)

    def forward(self, x):
        b, c = x.shape[:2]
        with torch.no_grad():  # the filter is fixed; no gradient flows through hf (as in legacy)
            hf = F.conv2d(x, self.laplacian.expand(c, 1, 3, 3).to(x.dtype), padding=1, groups=c)
        ctx = torch.cat([self.pool(x).flatten(1), self.pool(hf.abs()).flatten(1)], dim=1)
        return self.fc(ctx).view(b, -1, 1, 1)


class ConvExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = ConvBNReLU(dim, dim, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


class ScanMoEFusion(nn.Module):
    """Legacy `MambaMoE` (despite the old name, there is no Mamba/SSM inside; see CHANGES.md).

    x   = RecurrentScanFusion(l, m, s)
    p   = softmax(router(x)) over E=2 experts        (dense/soft mixture, every expert runs)
    out = 1x1conv( sum_e p_e * expert_e(x) )

    Load-balancing auxiliary loss (training only), d = batch-mean of p:
        aux = 10 * mean_e (d_e - 1/E)^2      -- pushes the average routing towards uniform.
    The training loss adds 0.1 * aux for each of the three fusion blocks.

    use_moe=False (ablation): a single ConvExpert, no router, aux = 0.
    """

    def __init__(self, in_dim, num_experts=2, use_scan=True, use_moe=True):
        super().__init__()
        self.use_moe = use_moe
        self.num_experts = num_experts if use_moe else 1
        self.context_aggregator = RecurrentScanFusion(in_dim, use_scan=use_scan)
        self.experts = nn.ModuleList([ConvExpert(in_dim) for _ in range(self.num_experts)])
        if use_moe:
            self.router = SpectralRouter(in_dim, self.num_experts)
        self.fuse = ConvBNReLU(in_dim, in_dim, 1)

    def forward(self, l, m, s):
        """Returns (features, aux_loss). aux_loss is a 0-dim tensor (0 in eval or without MoE)."""
        x = self.context_aggregator(l, m, s)
        if not self.use_moe:
            return self.fuse(self.experts[0](x)), x.new_zeros(())

        logits = self.router(x)  # [B, E, 1, 1]
        if self.training:
            density = F.softmax(logits.flatten(2).float(), dim=1).mean(dim=0)  # [E, 1]
            target = torch.full_like(density.mean(1), 1.0 / self.num_experts)
            aux = F.mse_loss(density.mean(1), target) * 10.0
        else:
            aux = x.new_zeros(())
        weights = F.softmax(logits, dim=1)  # [B, E, 1, 1]
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, C, H, W]
        out = (expert_out * weights.unsqueeze(2)).sum(dim=1)
        return self.fuse(out), aux


# ---------------------------------------------------------------------------------------------
# CamoXpert additions (legacy improved_modules.py)
# ---------------------------------------------------------------------------------------------
class AdaptiveScaleWeighting(nn.Module):
    """Per-image softmax weights for the three scales: w = softmax(MLP(concat[GAP(l),GAP(m),GAP(s)])).

    Each scale's whole feature map is multiplied by its scalar weight (w_l + w_m + w_s = 1).
    MLP: 3C -> C -> ReLU -> Dropout(0.1) -> C/2 -> ReLU -> 3.
    """

    def __init__(self, dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, l, m, s):
        ctx = torch.cat([self.global_pool(t).flatten(1) for t in (l, m, s)], dim=1)
        w = self.fc(ctx)  # [B, 3]
        w_l, w_m, w_s = (w[:, i].view(-1, 1, 1, 1) for i in range(3))
        return l * w_l, m * w_m, s * w_s


class BoundaryRefinementModule(nn.Module):
    """Predict an edge map from the features and feed it back.

    e_logit = conv(conv(conv(x)))                          [B, 1, H, W]  (edge head)
    out     = 3 x ConvBNReLU(concat[x, sigmoid(e_logit)])   [B, C, H, W]
    The edge head is supervised by edges extracted from the GT mask (losses.edge_targets).
    Returns (out, e_logit). Legacy returned sigmoid(e_logit); the loss applies the sigmoid
    itself (in float32, outside autocast), giving the same value (see CHANGES.md).
    """

    def __init__(self, in_dim):
        super().__init__()
        self.edge_conv = nn.Sequential(
            ConvBNReLU(in_dim, in_dim // 2, 3, 1, 1),
            ConvBNReLU(in_dim // 2, in_dim // 4, 3, 1, 1),
            nn.Conv2d(in_dim // 4, 1, 1),
        )
        self.refine = nn.Sequential(
            ConvBNReLU(in_dim + 1, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )

    def forward(self, x):
        edge_logits = self.edge_conv(x)
        refined = self.refine(torch.cat([x, edge_logits.sigmoid()], dim=1))
        return refined, edge_logits


class ProgressiveRefinement(nn.Module):
    """T prediction heads applied one after another (CamoXpert-PR).

    p_0 = 0;   for t = 1..T:  f_t = 3 x ConvBNReLU(concat[x, sigmoid(p_{t-1}) (no grad)]),
                              p_t = 1x1conv(f_t)
    Note: every step starts again from the same decoder features x (not from f_{t-1}); only the
    previous prediction is carried over, and it is detached. All p_t are supervised; the last
    one is the output.
    """

    def __init__(self, in_dim, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations
        self.refinement_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(in_dim + 1, in_dim, 3, 1, 1),
                    ConvBNReLU(in_dim, in_dim, 3, 1, 1),
                    ConvBNReLU(in_dim, in_dim, 3, 1, 1),
                )
                for _ in range(num_iterations)
            ]
        )
        self.pred_heads = nn.ModuleList([nn.Conv2d(in_dim, 1, 1) for _ in range(num_iterations)])

    def forward(self, x):
        preds = []
        prev = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        for i in range(self.num_iterations):
            feat = self.refinement_blocks[i](torch.cat([x, prev], dim=1))
            p = self.pred_heads[i](feat)
            preds.append(p)
            prev = p.sigmoid().detach()
        return preds
