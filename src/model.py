import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class PatchifyStem(nn.Module):
    """Patchify stem with 4x4 non-overlapping convolution + LayerNorm"""
    def __init__(self, in_channels=3, out_channels=48):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        # Rearrange for LayerNorm: (B, C, H, W) -> (B, H, W, C) -> norm -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvEncoder(nn.Module):
    """Convolution encoder block with depth-wise separable convolution"""
    def __init__(self, dim, kernel_size=3, expansion_ratio=4, drop_path=0.):
        super().__init__()
        expanded_dim = dim * expansion_ratio
        
        # Depth-wise convolution
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim)
        
        # Two pointwise convolutions with GELU and LayerNorm
        self.pw_conv1 = nn.Conv2d(dim, expanded_dim, 1)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(expanded_dim)
        
        self.pw_conv2 = nn.Conv2d(expanded_dim, dim, 1)
        self.act2 = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Depth-wise convolution
        x = self.dw_conv(x)
        
        # First pointwise convolution
        x = self.pw_conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act1(x)
        
        # Second pointwise convolution
        x = self.pw_conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act2(x)
        
        # Skip connection
        x = identity + self.drop_path(x)
        return x

class SDTAEncoder(nn.Module):
    """Spatial-Depthwise Transposed Attention Encoder"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Depth-wise convolution
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Final 1x1 convolutions
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.act2 = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Depth-wise convolution
        x = self.dw_conv(x)
        
        # Prepare for attention: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Q, K, V projections
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # L2 normalize Q and K
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Transposed attention (across channels)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to V
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        
        # Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # First 1x1 convolution
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act1(x)
        
        # Second 1x1 convolution
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act2(x)
        
        # Skip connection
        x = identity + self.drop_path(x)
        return x

class EdgeNeXtStage(nn.Module):
    """A single stage of EdgeNeXt"""
    def __init__(self, in_dim, out_dim, depth, kernel_size=3, num_heads=8, expansion_ratio=4, 
                 drop_path_rates=[], has_sdta=False):
        super().__init__()
        self.has_sdta = has_sdta
        
        # Downsample layer if needed
        if in_dim != out_dim:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()
        
        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            drop_path = drop_path_rates[i] if i < len(drop_path_rates) else 0.
            
            if i == depth - 1 and has_sdta:
                # SDTA block as the last block
                self.blocks.append(SDTAEncoder(out_dim, num_heads, drop_path=drop_path))
            else:
                # Regular convolution encoder block
                self.blocks.append(ConvEncoder(out_dim, kernel_size, expansion_ratio, drop_path))
    
    def forward(self, x):
        x = self.downsample(x)
        
        for block in self.blocks:
            x = block(x)
        
        return x

class EdgeNeXtBackbone(nn.Module):
    """Complete EdgeNeXt backbone"""
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[48, 96, 160, 256],
                 kernel_sizes=[3, 5, 7, 9], num_heads=8, expansion_ratio=4, drop_path_rate=0.):
        super().__init__()
        
        # Patchify stem
        self.patch_embed = PatchifyStem(in_chans, dims[0])
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(4):
            # Determine input and output dimensions
            in_dim = dims[i] if i == 0 else dims[i-1]
            out_dim = dims[i]
            
            # Determine if this stage has SDTA (last three stages)
            has_sdta = (i > 0)  # Stage 2, 3, 4 have SDTA
            
            # Get drop path rates for this stage
            stage_dpr = dpr[sum(depths[:i]):sum(depths[:i+1])]
            
            # Create stage
            stage = EdgeNeXtStage(
                in_dim=in_dim,
                out_dim=out_dim,
                depth=depths[i],
                kernel_size=kernel_sizes[i],
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                drop_path_rates=stage_dpr,
                has_sdta=has_sdta
            )
            self.stages.append(stage)
        
        # Classification head (optional)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], 1000)  # Standard ImageNet classes
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Forward through stages
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)  # Collect features from all stages
        
        # Classification (optional)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return features, x

def build_edgenext(model_type='small'):
    """Helper function to build EdgeNeXt models"""
    if model_type == 'small':
        return EdgeNeXtBackbone(
            depths=[3, 3, 9, 3],  # Stage 1: 3 blocks, Stage 2: 3 blocks, etc.
            dims=[48, 96, 160, 256],
            kernel_sizes=[3, 5, 7, 9]
        )
    elif model_type == 'base':
        return EdgeNeXtBackbone(
            depths=[3, 3, 9, 3],
            dims=[64, 128, 256, 512],
            kernel_sizes=[3, 5, 7, 9]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
