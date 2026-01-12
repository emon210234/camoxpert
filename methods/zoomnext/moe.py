import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import ConvBNReLU, resize_to

class Expert(nn.Module):
    """
    A Specialist Expert Module.
    In CamoXpert V2, each expert is a Residual Block designed to process 
    specific features (texture vs structure) preserved from the backbone.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBNReLU(dim, dim, 3, 1, 1)
        self.conv2 = ConvBNReLU(dim, dim, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class DynamicScaleMoE(nn.Module):
    """
    Dynamic Scale Mixture of Experts (DS-MoE).
    Replaces the static MHSIU fusion. 
    Uses a Gating Network (Router) to dynamically weight experts per pixel.
    """
    def __init__(self, in_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. Scale Pre-processing (Same as ZoomNeXt for fair comparison)
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        # 2. The MoE Core
        # Input channels = 3 * in_dim (Large + Medium + Small concatenated)
        self.in_channels = in_dim * 3
        
        # The Experts (4 parallel networks)
        self.experts = nn.ModuleList([Expert(self.in_channels) for _ in range(num_experts)])
        
        # The Router (Gating Network)
        # Decides which expert handles which pixel
        # Output: [B, num_experts, H, W]
        self.router = nn.Sequential(
            nn.Conv2d(self.in_channels, num_experts, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # --- THE FIX IS HERE ---
        # 3. Final Fusion: Project back to 'in_dim' (e.g., 64)
        # Old (Broken): self.fuse = ConvBNReLU(self.in_channels, self.in_channels, 1)
        # New (Fixed):  self.fuse = ConvBNReLU(self.in_channels, in_dim, 1)
        self.fuse = ConvBNReLU(self.in_channels, in_dim, 1)

    def forward(self, l, m, s):
        # Align sizes to Medium scale (Anchor)
        tgt_size = m.shape[2:]
        
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        
        l, m, s = self.conv_l(l), self.conv_m(m), self.conv_s(s)
        
        # Concatenate scales: [B, 3*C, H, W]
        x = torch.cat([l, m, s], dim=1) 
        
        # --- MoE Routing Logic ---
        
        # A. Calculate Expert Weights (Soft Routing)
        # Shape: [B, num_experts, H, W]
        routing_weights = self.router(x)
        
        # B. Execute Experts
        # Shape: [B, num_experts, C, H, W]
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        
        # C. Weighted Sum
        # Unsqueeze weights to [B, num_experts, 1, H, W] to broadcast over Channels
        weights = routing_weights.unsqueeze(2) 
        
        # Sum(Weight * ExpertOutput)
        output = (expert_outputs * weights).sum(dim=1)
        
        return self.fuse(output)