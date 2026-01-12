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

class SupervisedHybridMoE(nn.Module):
    """
    CamoXpert V5: Exposes routing weights for supervision.
    """
    def __init__(self, in_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        # Store weights for loss calculation
        self.routing_weights = None 
        
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.in_channels = in_dim * 3
        
        # Experts 0,1: CONV (Edge) | Experts 2,3: ATTN (Body)
        self.experts = nn.ModuleList()
        self.experts.append(ConvExpert(self.in_channels))     # Expert 0
        self.experts.append(ConvExpert(self.in_channels))     # Expert 1
        self.experts.append(AttentionExpert(self.in_channels)) # Expert 2
        self.experts.append(AttentionExpert(self.in_channels)) # Expert 3
        
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
        
        # Save weights for supervision
        self.routing_weights = F.softmax(router_logits, dim=1)
        
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        weights = self.routing_weights.unsqueeze(2) 
        output = (expert_outputs * weights).sum(dim=1)
        
        return self.fuse(output)