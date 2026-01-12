import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import ConvBNReLU, resize_to

class ConvExpert(nn.Module):
    """
    Local Expert: Good for texture and edges (legs, antennae).
    Uses standard Convolutions.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBNReLU(dim, dim, 3, 1, 1)
        self.conv2 = ConvBNReLU(dim, dim, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class AttentionExpert(nn.Module):
    """
    Global Expert: Good for shape and context (body, shadow).
    Uses Efficient Global Attention (simulating BiFormer/Transformer behavior).
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Efficient Attention: reducing spatial dim for global context
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        
        # 1. Global Context Interaction (The "Transformer" part)
        # Flatten [B, C, H, W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2) 
        x_norm = self.norm(x_flat)
        
        # Global Pooling for Key/Value to save memory (Efficient Attention)
        # We query the global context using the local pixels
        context = self.pool(x).flatten(2).transpose(1, 2) # [B, 1, C]
        
        q = self.proj_q(x_norm)       # [B, HW, C]
        k = self.proj_k(context)      # [B, 1, C]
        v = self.proj_v(context)      # [B, 1, C]
        
        # Attention score: Pixel vs Global Context
        attn = (q @ k.transpose(-2, -1)) * (c ** -0.5) # [B, HW, 1]
        attn = attn.softmax(dim=-1)
        
        # Aggregation
        out = (attn @ v) # [B, HW, C]
        out = self.proj_out(out)
        
        # Reshape back
        out = out.transpose(1, 2).view(b, c, h, w)
        
        return identity + out

class HybridMoE(nn.Module):
    """
    CamoXpert V3: Hybrid Mixture of Experts.
    Router chooses between Local (Conv) and Global (Attention) experts.
    """
    def __init__(self, in_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        
        # Scale Processing
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        self.in_channels = in_dim * 3
        
        # --- THE INNOVATION: HYBRID EXPERTS ---
        self.experts = nn.ModuleList()
        
        # Expert 1 & 2: Local Conv (For Legs)
        self.experts.append(ConvExpert(self.in_channels))
        self.experts.append(ConvExpert(self.in_channels))
        
        # Expert 3 & 4: Global Attention (For Body/Context)
        self.experts.append(AttentionExpert(self.in_channels))
        self.experts.append(AttentionExpert(self.in_channels))
        
        # Context-Aware Router (Upgraded from 3x3 to Global+Local)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.router_fc = nn.Linear(self.in_channels, num_experts)
        self.router_conv = nn.Conv2d(self.in_channels, num_experts, 3, 1, 1)
        
        # Final Fuse
        self.fuse = ConvBNReLU(self.in_channels, in_dim, 1)

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        l, m, s = self.conv_l(l), self.conv_m(m), self.conv_s(s)
        x = torch.cat([l, m, s], dim=1) 
        
        # --- Enhanced Routing Logic ---
        # Combine Global Context decision + Local Pixel decision
        b, c, h, w = x.shape
        
        # 1. Global perspective
        global_ctx = self.global_pool(x).flatten(1)
        global_logits = self.router_fc(global_ctx).view(b, self.num_experts, 1, 1)
        
        # 2. Local perspective
        local_logits = self.router_conv(x)
        
        # 3. Fuse & Noise
        router_logits = global_logits + local_logits
        
        if self.training:
            noise = torch.randn_like(router_logits) * 0.5 # Reduced noise slightly
            router_logits = router_logits + noise
            
        routing_weights = F.softmax(router_logits, dim=1)
        
        # Execution
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        weights = routing_weights.unsqueeze(2) 
        output = (expert_outputs * weights).sum(dim=1)
        
        return self.fuse(output)