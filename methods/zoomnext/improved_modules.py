"""
Improved modules for CamoXpert architecture enhancement.
Contains boundary refinement, adaptive scale weighting, and enhanced losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import ConvBNReLU


class BoundaryRefinementModule(nn.Module):
    """
    Refines object boundaries using edge-aware attention.
    
    This module explicitly detects edges and uses them to guide
    feature refinement, which is critical for camouflaged object detection
    where boundaries are ambiguous.
    """
    def __init__(self, in_dim):
        super().__init__()
        # Edge detection branch
        self.edge_conv = nn.Sequential(
            ConvBNReLU(in_dim, in_dim//2, 3, 1, 1),
            ConvBNReLU(in_dim//2, in_dim//4, 3, 1, 1),
            nn.Conv2d(in_dim//4, 1, 1)
        )
        
        # Feature refinement guided by edges
        self.refine = nn.Sequential(
            ConvBNReLU(in_dim + 1, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            refined: Refined features [B, C, H, W]
            edge_map: Predicted edge map [B, 1, H, W] (for supervision)
        """
        # Detect edges
        edge_map = self.edge_conv(x).sigmoid()
        
        # Concatenate edge information with features
        x_with_edge = torch.cat([x, edge_map], dim=1)
        
        # Refine features using edge guidance
        refined = self.refine(x_with_edge)
        
        return refined, edge_map


class AdaptiveScaleWeighting(nn.Module):
    """
    Learns optimal scale weights per image based on content characteristics.
    
    Different camouflaged objects require different scale emphasis:
    - Large objects benefit from large scales
    - Small details benefit from small scales
    - Complex textures benefit from medium scales
    """
    def __init__(self, dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Network to predict scale importance
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, l, m, s):
        """
        Args:
            l: Large scale features [B, C, H_l, W_l]
            m: Medium scale features [B, C, H_m, W_m]
            s: Small scale features [B, C, H_s, W_s]
        Returns:
            l_weighted, m_weighted, s_weighted: Weighted features
        """
        # Extract global context from each scale
        l_ctx = self.global_pool(l).flatten(1)  # [B, C]
        m_ctx = self.global_pool(m).flatten(1)  # [B, C]
        s_ctx = self.global_pool(s).flatten(1)  # [B, C]
        
        # Learn scale importance weights
        combined = torch.cat([l_ctx, m_ctx, s_ctx], dim=1)  # [B, 3C]
        weights = self.fc(combined)  # [B, 3]
        
        w_l, w_m, w_s = weights.chunk(3, dim=1)  # Each [B, 1]
        
        # Apply learned weights to features
        l_weighted = l * w_l.view(-1, 1, 1, 1)
        m_weighted = m * w_m.view(-1, 1, 1, 1)
        s_weighted = s * w_s.view(-1, 1, 1, 1)
        
        return l_weighted, m_weighted, s_weighted


class CamouflageDetectionLoss(nn.Module):
    """
    Multi-component loss function optimized for camouflaged object detection.
    
    Combines:
    1. BCE Loss - pixel-wise classification
    2. IoU Loss - region-based optimization
    3. SSIM Loss - structural similarity preservation
    4. Edge Loss - boundary awareness
    """
    def __init__(self, bce_weight=1.0, iou_weight=0.5, ssim_weight=0.3, edge_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
    def forward(self, pred, mask, edge_pred=None):
        """
        Args:
            pred: Predicted logits [B, 1, H, W]
            mask: Ground truth mask [B, 1, H, W]
            edge_pred: Optional predicted edge map [B, 1, H, W]
        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary with individual loss values
        """
        # 1. Binary Cross-Entropy Loss (pixel-wise)
        bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        
        # 2. IoU Loss (region-based)
        pred_sigmoid = pred.sigmoid()
        intersection = (pred_sigmoid * mask).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - intersection
        iou_loss = 1 - (intersection / (union + 1e-8)).mean()
        
        # 3. SSIM Loss (structural similarity)
        ssim_loss = 1 - self.ssim(pred_sigmoid, mask)
        
        # 4. Edge Loss (boundary awareness)
        if edge_pred is not None:
            edge_gt = self.extract_edges(mask)
            # Resize edge_pred to match edge_gt size
            if edge_pred.shape[-2:] != edge_gt.shape[-2:]:
                edge_pred = F.interpolate(edge_pred, size=edge_gt.shape[-2:], 
                                         mode='bilinear', align_corners=False)
            edge_loss = F.binary_cross_entropy(edge_pred, edge_gt, reduction='mean')
        else:
            edge_loss = torch.tensor(0.0, device=pred.device)
        
        # Weighted combination
        total_loss = (
            self.bce_weight * bce_loss + 
            self.iou_weight * iou_loss + 
            self.ssim_weight * ssim_loss + 
            self.edge_weight * edge_loss
        )
        
        # Return individual losses for monitoring
        loss_dict = {
            'bce': bce_loss.item(),
            'iou': iou_loss.item(),
            'ssim': ssim_loss.item(),
            'edge': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else 0.0,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def ssim(self, pred, target, window_size=11, eps=1e-8):
        """
        Structural Similarity Index Measure (SSIM).
        
        Measures structural similarity between prediction and ground truth,
        which is important for preserving object shape in camouflage detection.
        """
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
        gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # Create 2D Gaussian kernel
        kernel = gauss.view(1, 1, -1, 1) * gauss.view(1, 1, 1, -1)
        kernel = kernel.expand(pred.size(1), 1, window_size, window_size).contiguous()
        
        # Compute statistics
        mu1 = F.conv2d(pred, kernel, padding=window_size//2, groups=pred.size(1))
        mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=target.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size//2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=window_size//2, groups=pred.size(1)) - mu1_mu2
        
        # SSIM formula
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def extract_edges(self, mask):
        """
        Extract edges from ground truth mask using Sobel operator.
        """
        # Sobel kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        # Convolve with Sobel kernels
        edges_x = F.conv2d(mask, kernel_x, padding=1)
        edges_y = F.conv2d(mask, kernel_y, padding=1)
        
        # Compute edge magnitude
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        
        # Threshold to binary edge map
        edges = (edges > 0.1).float()
        
        return edges


class ProgressiveRefinement(nn.Module):
    """
    Iteratively refines predictions using residual updates.
    
    Multiple refinement stages allow the model to:
    1. First pass: Rough segmentation
    2. Subsequent passes: Detail refinement
    """
    def __init__(self, in_dim, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Refinement blocks
        self.refinement_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(in_dim + 1, in_dim, 3, 1, 1),  # +1 for previous prediction
                ConvBNReLU(in_dim, in_dim, 3, 1, 1),
                ConvBNReLU(in_dim, in_dim, 3, 1, 1)
            ) for _ in range(num_iterations)
        ])
        
        # Prediction heads for each iteration
        self.pred_heads = nn.ModuleList([
            nn.Conv2d(in_dim, 1, 1) for _ in range(num_iterations)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            predictions: List of predictions at each refinement stage
        """
        predictions = []
        curr_pred = None
        
        for i in range(self.num_iterations):
            if curr_pred is not None:
                curr_feat = torch.cat([x, curr_pred.sigmoid().detach()], dim=1)
            else:
                # First iteration: create dummy prediction (zeros) to match expected input channels
                dummy_pred = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                                        device=x.device, dtype=x.dtype)
                curr_feat = torch.cat([x, dummy_pred], dim=1)
            
            # Refine features
            curr_feat = self.refinement_blocks[i](curr_feat)
            
            # Generate prediction
            curr_pred = self.pred_heads[i](curr_feat)
            predictions.append(curr_pred)
        
        return predictions  # Return all for deep supervision


class SparseSpectralRouter(nn.Module):
    """
    Top-K sparse routing for efficient Mixture of Experts.
    
    Only activates the K most relevant experts per input region,
    significantly reducing computational cost while maintaining accuracy.
    """
    def __init__(self, dim, num_experts, k=2):
        super().__init__()
        self.k = min(k, num_experts)
        self.num_experts = num_experts
        
        # Context extraction
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Frequency analysis (fixed Laplacian filter)
        self.register_buffer('laplacian', 
            torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
            dtype=torch.float32).view(1, 1, 3, 3))
        
        # Router network
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_experts)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            routing_weights: Sparse routing weights [B, num_experts, 1, 1]
            top_k_indices: Indices of top-K experts [B, K, 1, 1]
        """
        b, c, h, w = x.shape
        
        # Extract frequency information (high-frequency = edges/texture)
        with torch.no_grad():
            weight = self.laplacian.expand(c, 1, 3, 3).to(x.device)
            high_freq = F.conv2d(x, weight, padding=1, groups=c)
        
        # Global context
        global_ctx = self.pool(x).flatten(1)
        freq_ctx = self.pool(high_freq.abs()).flatten(1)
        
        # Combine contexts
        combined = torch.cat([global_ctx, freq_ctx], dim=1)
        
        # Router logits
        logits = self.fc(combined).view(b, self.num_experts, 1, 1)
        
        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=1)
        
        # Sparse routing weights (only top-K are non-zero)
        routing_weights = torch.zeros_like(logits)
        routing_weights.scatter_(1, top_k_indices, 
                                F.softmax(top_k_logits, dim=1))
        
        return routing_weights, top_k_indices
