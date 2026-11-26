"""Training losses. All terms are computed in float32, outside autocast.

Symbols: z = logits upsampled to the mask size (S x S), p = sigmoid(z) in (0, 1),
g = ground-truth mask in [0, 1], averages run over pixels and images.

  BCE   = mean[ -g log p - (1-g) log(1-p) ]                     (pixel-wise)
  IoU   = 1 - mean_images[ sum(p*g) / (sum(p) + sum(g) - sum(p*g) + 1e-8) ]   (soft IoU)
  SSIM  = 1 - mean SSIM(p, g), 11x11 Gaussian window (sigma 1.5), C1 = 0.01^2, C2 = 0.03^2
  Edge  = BCE(up(sigmoid(e)), E(g)),  E(g) = 1[ |Sobel(g)| > 0.1 ]  (e: edge-head logits at
          stride 4, up: bilinear upsampling to S x S; E(g): edge map from the mask)
  UAL   = c(t) * mean[ 1 - |2p - 1|^2 ],  c(t) = (1 - cos(pi t)) / 2, t = training progress
          (ZoomNeXt's uncertainty-aware loss: penalises p near 0.5, weight ramps 0 -> 1)
  Aux   = MoE load-balancing term returned by the model.

Per prediction:  L_pred = w_bce*BCE + w_iou*IoU + w_ssim*SSIM (+ w_ual*UAL)
Total:           L = sum_k a_k * L_pred_k  +  w_edge * a_last * Edge  +  w_aux * Aux
where k runs over the supervised predictions: just the final one, or, for progressive
refinement, every step with a_k = 1 + 0.2 k (k = 0, 1, ...). This reproduces the legacy v11 /
v12 / v12-Progressive losses exactly (see CHANGES.md).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(prob, mask):
    inter = (prob * mask).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - inter
    return 1 - (inter / (union + 1e-8)).mean()


def gaussian_window(window_size=11, sigma=1.5, dtype=torch.float32, device=None):
    g = torch.arange(window_size, dtype=dtype, device=device)
    g = torch.exp(-((g - window_size // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)  # outer product -> [1, 1, k, k]


def ssim(pred, target, window_size=11, sigma=1.5):
    """Mean SSIM with zero padding (as in the legacy code).

    SSIM = ((2 mu_x mu_y + C1)(2 sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)),
    local means / variances / covariance taken with the Gaussian window.
    """
    c = pred.size(1)
    k = gaussian_window(window_size, sigma, pred.dtype, pred.device).expand(c, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(pred, k, padding=pad, groups=c)
    mu2 = F.conv2d(target, k, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(pred * pred, k, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(target * target, k, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(pred * target, k, padding=pad, groups=c) - mu12
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean()


def edge_targets(mask, thr=0.1):
    """Binary edge map of the mask: Sobel gradient magnitude > thr (about 2 px wide)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    return (torch.sqrt(gx**2 + gy**2 + 1e-8) > thr).float()


def ual_coef(progress):
    """ZoomNeXt's cosine ramp from 0 (start of training) to 1 (end); progress in [0, 1]."""
    progress = min(max(progress, 0.0), 1.0)
    return (1 - math.cos(progress * math.pi)) / 2


def ual_loss(prob):
    return (1 - (2 * prob - 1).abs().pow(2)).mean()


class SegLoss(nn.Module):
    def __init__(self, bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=0.0, aux=0.0, side_step=0.2):
        super().__init__()
        self.w = dict(bce=bce, iou=iou, ssim=ssim, edge=edge, ual=ual, aux=aux)
        self.side_step = side_step

    def _upsample(self, z, size):
        z = z.float()
        if z.shape[-2:] != size:
            z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
        return z

    def _pred_terms(self, z, mask, progress):
        terms = {"bce": F.binary_cross_entropy_with_logits(z, mask)}
        p = z.sigmoid()
        if self.w["iou"]:
            terms["iou"] = soft_iou_loss(p, mask)
        if self.w["ssim"]:
            terms["ssim"] = 1 - ssim(p, mask)
        if self.w["ual"]:
            terms["ual"] = ual_coef(progress) * ual_loss(p)
        total = sum(self.w[k] * v for k, v in terms.items())
        return total, terms

    def forward(self, out, mask, progress=1.0):
        """out: model output dict; mask: [B, 1, S, S] float in [0, 1]; progress: iteration / total.
        Returns (total_loss, dict of floats for logging)."""
        mask = mask.float()
        size = mask.shape[-2:]
        preds = out["side_logits"] if out["side_logits"] else [out["logits"]]
        total = 0.0
        log = {}
        for k, z in enumerate(preds):
            a_k = 1.0 + self.side_step * k
            l_k, terms = self._pred_terms(self._upsample(z, size), mask, progress)
            total = total + a_k * l_k
            for name, v in terms.items():
                log[f"{name}{k}" if len(preds) > 1 else name] = v.item()
        a_last = 1.0 + self.side_step * (len(preds) - 1)

        if self.w["edge"]:
            if out["edge_logits"] is None:
                raise ValueError("edge loss weight > 0 but the model has no edge head (brm=False)")
            # Legacy order: sigmoid at the edge-head resolution, THEN bilinear upsampling of the
            # probabilities, then BCE. (Upsampling logits first would give a different value,
            # because sigmoid is nonlinear.) Safe here: we are outside autocast, in float32.
            e = out["edge_logits"].float().sigmoid()
            if e.shape[-2:] != size:
                e = F.interpolate(e, size=size, mode="bilinear", align_corners=False)
            edge = F.binary_cross_entropy(e, edge_targets(mask))
            total = total + a_last * self.w["edge"] * edge
            log["edge"] = edge.item()
        if self.w["aux"]:
            aux = out["aux_loss"].float()
            total = total + self.w["aux"] * aux
            log["aux"] = aux.item()
        log["total"] = total.item()
        return total, log
