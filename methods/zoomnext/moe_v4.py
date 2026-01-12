import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import ConvBNReLU, resize_to

class ConvExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBNReLU(dim, dim, 3, 1, 1)
        self.conv2 = ConvBNReLU(dim, dim, 3, 1, 1)
    def forward(self, x): return x + self.conv2(self.conv1(x))

class AttentionExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        x_flat = x.flatten(2).transpose(1, 2) 
        x_norm = self.norm(x_flat)
        context = self.pool(x).flatten(2).transpose(1, 2)
        q = self.proj_q(x_norm)
        k = self.proj_k(context)
        v = self.proj_v(context)
        attn = (q @ k.transpose(-2, -1)) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        out = self.proj_out(attn @ v)
        out = out.transpose(1, 2).view(b, c, h, w)
        return identity + out

class BalancedHybridMoE(nn.Module):
    """
    CamoXpert V4: Hybrid MoE with Load Balancing Loss.
    """
    def __init__(self, in_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        # We store the auxiliary loss here to retrieve it later
        self.aux_loss = 0.0 
        
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.in_channels = in_dim * 3
        
        self.experts = nn.ModuleList()
        self.experts.append(ConvExpert(self.in_channels))
        self.experts.append(ConvExpert(self.in_channels))
        self.experts.append(AttentionExpert(self.in_channels))
        self.experts.append(AttentionExpert(self.in_channels))
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.router_fc = nn.Linear(self.in_channels, num_experts)
        self.router_conv = nn.Conv2d(self.in_channels, num_experts, 3, 1, 1)
        self.fuse = ConvBNReLU(self.in_channels, in_dim, 1)

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        l, m, s = self.conv_l(l), self.conv_m(m), self.conv_s(s)
        x = torch.cat([l, m, s], dim=1) 
        
        b, c, h, w = x.shape
        
        # Router
        global_ctx = self.global_pool(x).flatten(1)
        global_logits = self.router_fc(global_ctx).view(b, self.num_experts, 1, 1)
        local_logits = self.router_conv(x)
        router_logits = global_logits + local_logits
        
        # --- LOAD BALANCING LOSS CALCULATION ---
        if self.training:
            # Flatten spatial dims: [B, Experts, H, W] -> [B*H*W, Experts]
            logits_flat = router_logits.permute(0, 2, 3, 1).flatten(0, 2)
            probs = F.softmax(logits_flat, dim=1)
            
            # 1. Density: Average probability per expert across the batch
            density = probs.mean(dim=0) # [Experts]
            
            # 2. Assignment: Which expert was actually picked?
            target_density = torch.ones_like(density) / self.num_experts
            
            # Aux Loss
            self.aux_loss = F.mse_loss(density, target_density) * 10.0
            
            # --- COOL-DOWN MODE: NOISE DISABLED ---
            # noise = torch.randn_like(router_logits) * 0.5 
            # router_logits = router_logits + noise
            # --------------------------------------
        else:
            self.aux_loss = 0.0
        # ---------------------------------------

        routing_weights = F.softmax(router_logits, dim=1)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        weights = routing_weights.unsqueeze(2) 
        output = (expert_outputs * weights).sum(dim=1)
        
        return self.fuse(output)