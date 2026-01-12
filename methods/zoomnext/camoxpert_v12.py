"""
CamoXpertV12: Enhanced architecture with boundary refinement and improved loss.
This version builds on V11 with the following improvements:
1. Boundary Refinement Module for better edge detection
2. Adaptive Scale Weighting for intelligent multi-scale fusion
3. Enhanced Loss Function with IoU, SSIM, and edge components
4. Progressive Refinement for iterative detail enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.pvt_v2_eff import pvt_v2_eff_b5
from .layers import SimpleASPP, RGPU
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from .zoomnext import _ZoomNeXt_Base, MambaMoE
from .improved_modules import (
    BoundaryRefinementModule, 
    AdaptiveScaleWeighting,
    CamouflageDetectionLoss,
    ProgressiveRefinement
)


class CamoXpertV12(_ZoomNeXt_Base):
    """
    Enhanced CamoXpert with Priority 1 improvements:
    - Boundary Refinement Module (BRM)
    - Adaptive Scale Weighting (ASW)
    - Multi-component Loss
    - Progressive Refinement (optional)
    """
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False, 
                 use_progressive_refinement=False, num_refinement_iters=2):
        super().__init__(num_frames, pretrained, use_checkpoint)
        
        # Backbone: PVTv2-B5
        self.backbone = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
        self.normalize_encoder = PixelNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        embed_dims = [64, 128, 320, 512]
        out_dim = 64
        
        # Feature projection layers
        self.tra_5 = ConvBNReLU(embed_dims[3], out_dim, 1)
        self.tra_4 = ConvBNReLU(embed_dims[2], out_dim, 1)
        self.tra_3 = ConvBNReLU(embed_dims[1], out_dim, 1)
        self.tra_2 = ConvBNReLU(embed_dims[0], out_dim, 1)
        
        # NEW: Adaptive Scale Weighting
        self.asw_5 = AdaptiveScaleWeighting(out_dim)
        self.asw_4 = AdaptiveScaleWeighting(out_dim)
        self.asw_3 = AdaptiveScaleWeighting(out_dim)
        
        # Multi-scale fusion with Mamba + MoE
        self.siu_5 = MambaMoE(out_dim)
        self.siu_4 = MambaMoE(out_dim)
        self.siu_3 = MambaMoE(out_dim)
        
        # Hierarchical feature refinement
        self.hmu_5 = SimpleASPP(out_dim, out_dim)
        self.hmu_4 = SimpleASPP(out_dim, out_dim)
        self.hmu_3 = SimpleASPP(out_dim, out_dim)
        
        # Final upsampling and fusion
        self.rgpu = RGPU(out_dim, out_dim)
        
        # NEW: Boundary Refinement Module
        self.boundary_refine = BoundaryRefinementModule(out_dim)
        
        # NEW: Progressive Refinement (optional)
        self.use_progressive_refinement = use_progressive_refinement
        if use_progressive_refinement:
            self.progressive_refine = ProgressiveRefinement(out_dim, num_refinement_iters)
        
        # Final prediction head
        self.proj = nn.Conv2d(out_dim, 1, 3, 1, 1)
        
        # NEW: Enhanced Loss Function
        self.loss_fn = CamouflageDetectionLoss(
            bce_weight=1.0,
            iou_weight=0.5,
            ssim_weight=0.3,
            edge_weight=0.2
        )
    
    def encoder(self, x):
        """Extract multi-scale features from backbone"""
        feats = self.backbone(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        return [None] + feats
    
    def body(self, data):
        """
        Forward pass through the decoder.
        
        Args:
            data: Dictionary with keys:
                - image_l: Large scale input
                - image_m: Medium scale input
                - image_s: Small scale input
        
        Returns:
            If use_progressive_refinement:
                List of predictions (for deep supervision)
            Else:
                Single prediction
        """
        # Extract features at multiple scales
        l_f = self.encoder(self.normalize_encoder(data["image_l"]))
        m_f = self.encoder(self.normalize_encoder(data["image_m"]))
        s_f = self.encoder(self.normalize_encoder(data["image_s"]))
        
        # ========== Stage 5 (Deepest) ==========
        l5, m5, s5 = self.tra_5(l_f[4]), self.tra_5(m_f[4]), self.tra_5(s_f[4])
        
        # Apply adaptive scale weighting
        l5_w, m5_w, s5_w = self.asw_5(l5, m5, s5)
        
        # Fuse with Mamba + MoE
        lms5 = self.siu_5(l5_w, m5_w, s5_w)
        x = self.hmu_5(lms5)
        
        # ========== Stage 4 ==========
        l4, m4, s4 = self.tra_4(l_f[3]), self.tra_4(m_f[3]), self.tra_4(s_f[3])
        
        # Apply adaptive scale weighting
        l4_w, m4_w, s4_w = self.asw_4(l4, m4, s4)
        
        # Fuse with Mamba + MoE and add upsampled previous stage
        lms4 = self.siu_4(l4_w, m4_w, s4_w)
        x = self.hmu_4(lms4 + resize_to(x, lms4.shape[-2:]))
        
        # ========== Stage 3 ==========
        l3, m3, s3 = self.tra_3(l_f[2]), self.tra_3(m_f[2]), self.tra_3(s_f[2])
        
        # Apply adaptive scale weighting
        l3_w, m3_w, s3_w = self.asw_3(l3, m3, s3)
        
        # Fuse with Mamba + MoE and add upsampled previous stage
        lms3 = self.siu_3(l3_w, m3_w, s3_w)
        x = self.hmu_3(lms3 + resize_to(x, lms3.shape[-2:]))
        
        # ========== Stage 2 (Final) ==========
        l2 = self.tra_2(l_f[1])
        x = self.rgpu(l2, x)
        
        # NEW: Boundary refinement
        x, edge_map = self.boundary_refine(x)
        
        # Progressive refinement or single prediction
        if self.use_progressive_refinement:
            predictions = self.progressive_refine(x)
            # Store edge map for loss calculation
            self.edge_map = edge_map
            return predictions
        else:
            pred = self.proj(x)
            # Store edge map for loss calculation
            self.edge_map = edge_map
            return pred
    
    def forward(self, data):
        """
        Complete forward pass with loss calculation.
        
        Args:
            data: Dictionary with keys:
                - image_l, image_m, image_s: Multi-scale inputs
                - mask: Ground truth (only during training)
        
        Returns:
            Dictionary with 'loss' (training) or 'pred' (inference)
        """
        if self.training:
            mask = data["mask"]
            
            # Forward pass
            output = self.body(data)
            
            if self.use_progressive_refinement:
                # Deep supervision: compute loss for each refinement stage
                total_loss = 0
                loss_dicts = []
                
                for i, pred in enumerate(output):
                    # Resize if needed
                    if pred.shape[-2:] != mask.shape[-2:]:
                        pred = F.interpolate(pred, size=mask.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                    
                    # Compute loss (only use edge loss for final stage)
                    edge_pred = self.edge_map if i == len(output) - 1 else None
                    loss, loss_dict = self.loss_fn(pred, mask, edge_pred)
                    
                    # Weight: later stages are more important
                    weight = 1.0 + 0.2 * i
                    total_loss += weight * loss
                    loss_dicts.append(loss_dict)
                
                # Add MoE auxiliary losses
                aux_loss = self.siu_5.aux_loss + self.siu_4.aux_loss + self.siu_3.aux_loss
                total_loss += 0.1 * aux_loss
                
                return dict(loss=total_loss, loss_details=loss_dicts)
            
            else:
                # Single prediction
                logits = output
                
                # Resize if needed
                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = F.interpolate(logits, size=mask.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                
                # Compute enhanced loss
                loss, loss_dict = self.loss_fn(logits, mask, self.edge_map)
                
                # Add MoE auxiliary losses
                aux_loss = self.siu_5.aux_loss + self.siu_4.aux_loss + self.siu_3.aux_loss
                total_loss = loss + 0.1 * aux_loss
                
                return dict(loss=total_loss, loss_details=loss_dict)
        
        else:
            # Inference mode
            output = self.body(data)
            
            if self.use_progressive_refinement:
                # Use final refinement stage
                pred = output[-1].sigmoid()
            else:
                pred = output.sigmoid()
            
            return dict(pred=pred)


# Convenience aliases
class CamoXpertV12_Base(CamoXpertV12):
    """CamoXpertV12 without progressive refinement (faster)"""
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint, 
                        use_progressive_refinement=False)


class CamoXpertV12_Progressive(CamoXpertV12):
    """CamoXpertV12 with progressive refinement (higher accuracy)"""
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint, 
                        use_progressive_refinement=True, num_refinement_iters=2)
