import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import ConvBNReLU, resize_to

class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1), 
            ConvBNReLU(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        out = torch.cat([y5, y1, y2, y3, y4], dim=1)
        return self.fuse(out)

class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1)
        )
        self.num_groups = num_groups

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        
        l, m, s = self.conv_l(l), self.conv_m(m), self.conv_s(s)
        l_groups = torch.chunk(l, self.num_groups, dim=1)
        m_groups = torch.chunk(m, self.num_groups, dim=1)
        s_groups = torch.chunk(s, self.num_groups, dim=1)
        
        outs = []
        for lg, mg, sg in zip(l_groups, m_groups, s_groups):
            joint = torch.cat([lg, mg, sg], dim=1)
            attn = self.trans(joint)
            al, am, as_ = torch.chunk(attn, 3, dim=1)
            outs.append(lg * al + mg * am + sg * as_)
        return torch.cat(outs, dim=1)

class RGPU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_curr = ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        self.conv_prev = ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        self.conv_out = ConvBNReLU(2 * out_dim, out_dim, 3, 1, 1)

    def forward(self, curr_x, prev_x):
        curr = self.conv_curr(curr_x)
        prev = self.conv_prev(prev_x)
        prev = resize_to(prev, tgt_hw=curr.shape[2:])
        guided = curr * prev 
        return self.conv_out(torch.cat([guided, curr], dim=1))
