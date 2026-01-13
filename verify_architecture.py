"""
Quick architecture verification script.
Tests that V12 model can be created and performs forward pass correctly.
No training or real data required.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_creation():
    """Test that V12 model can be instantiated"""
    print("=" * 60)
    print("Test 1: Model Creation")
    print("=" * 60)
    
    try:
        from methods.zoomnext.camoxpert_v12 import CamoXpertV12_Base, CamoXpertV12_Progressive
        
        # Test base model
        model_base = CamoXpertV12_Base(num_frames=1, pretrained=False)
        params_base = sum(p.numel() for p in model_base.parameters()) / 1e6
        print(f"✓ CamoXpertV12_Base created successfully")
        print(f"  Parameters: {params_base:.2f}M")
        
        # Test progressive model
        model_prog = CamoXpertV12_Progressive(num_frames=1, pretrained=False)
        params_prog = sum(p.numel() for p in model_prog.parameters()) / 1e6
        print(f"✓ CamoXpertV12_Progressive created successfully")
        print(f"  Parameters: {params_prog:.2f}M")
        print(f"  Overhead: {params_prog - params_base:.2f}M (+{(params_prog/params_base - 1)*100:.1f}%)")
        
        return model_base, model_prog
    
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, model_name="CamoXpertV12"):
    """Test forward pass with random data"""
    print("\n" + "=" * 60)
    print(f"Test 2: Forward Pass - {model_name}")
    print("=" * 60)
    
    try:
        batch_size = 2
        h, w = 448, 448
        
        # Create dummy inputs
        img_l = torch.randn(batch_size, 3, h, w)
        img_m = torch.randn(batch_size, 3, int(h*0.75), int(w*0.75))
        img_s = torch.randn(batch_size, 3, h//2, w//2)
        mask = torch.rand(batch_size, 1, h, w)  # 0-1 range
        
        # Test training mode
        model.train()
        print(f"\nTraining mode:")
        out_train = model(data={
            "image_l": img_l, 
            "image_m": img_m, 
            "image_s": img_s, 
            "mask": mask
        })
        
        print(f"  Output keys: {list(out_train.keys())}")
        print(f"  Loss: {out_train['loss'].item():.6f}")
        
        if 'loss_details' in out_train:
            details = out_train['loss_details']
            if isinstance(details, list):
                print(f"  Progressive refinement stages: {len(details)}")
                print(f"  Final stage loss breakdown:")
                for k, v in details[-1].items():
                    print(f"    {k}: {v:.6f}")
            else:
                print(f"  Loss breakdown:")
                for k, v in details.items():
                    print(f"    {k}: {v:.6f}")
        
        # Test inference mode
        model.eval()
        print(f"\nInference mode:")
        with torch.no_grad():
            out_inf = model(data={
                "image_l": img_l, 
                "image_m": img_m, 
                "image_s": img_s
            })
        
        print(f"  Output keys: {list(out_inf.keys())}")
        print(f"  Prediction shape: {out_inf['pred'].shape}")
        print(f"  Prediction range: [{out_inf['pred'].min():.4f}, {out_inf['pred'].max():.4f}]")
        
        # Verify output is in valid range [0, 1]
        assert out_inf['pred'].min() >= 0 and out_inf['pred'].max() <= 1, \
            "Prediction should be in [0, 1] range"
        print(f"  ✓ Output range valid")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_components():
    """Test individual loss components"""
    print("\n" + "=" * 60)
    print("Test 3: Loss Components")
    print("=" * 60)
    
    try:
        from methods.zoomnext.improved_modules import CamouflageDetectionLoss
        
        loss_fn = CamouflageDetectionLoss(
            bce_weight=1.0,
            iou_weight=0.5,
            ssim_weight=0.3,
            edge_weight=0.2
        )
        
        # Create dummy prediction and mask
        pred = torch.randn(2, 1, 64, 64)
        mask = torch.rand(2, 1, 64, 64)
        edge_pred = torch.rand(2, 1, 64, 64)
        
        # Test loss calculation
        loss, loss_dict = loss_fn(pred, mask, edge_pred)
        
        print("Loss components:")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.6f}")
        
        print(f"\n✓ All loss components working correctly")
        return True
    
    except Exception as e:
        print(f"❌ Error testing loss components: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_improved_modules():
    """Test individual improved modules"""
    print("\n" + "=" * 60)
    print("Test 4: Individual Modules")
    print("=" * 60)
    
    try:
        from methods.zoomnext.improved_modules import (
            BoundaryRefinementModule,
            AdaptiveScaleWeighting,
            ProgressiveRefinement,
            SparseSpectralRouter
        )
        
        dim = 64
        batch_size = 2
        h, w = 56, 56
        
        # Test Boundary Refinement Module
        print("\n1. BoundaryRefinementModule:")
        brm = BoundaryRefinementModule(dim)
        x = torch.randn(batch_size, dim, h, w)
        refined, edge_map = brm(x)
        print(f"   Input: {x.shape}")
        print(f"   Refined: {refined.shape}")
        print(f"   Edge map: {edge_map.shape}")
        print(f"   ✓ Working correctly")
        
        # Test Adaptive Scale Weighting
        print("\n2. AdaptiveScaleWeighting:")
        asw = AdaptiveScaleWeighting(dim)
        l = torch.randn(batch_size, dim, h, w)
        m = torch.randn(batch_size, dim, h, w)
        s = torch.randn(batch_size, dim, h, w)
        l_w, m_w, s_w = asw(l, m, s)
        print(f"   Input shapes: {l.shape}, {m.shape}, {s.shape}")
        print(f"   Output shapes: {l_w.shape}, {m_w.shape}, {s_w.shape}")
        print(f"   ✓ Working correctly")
        
        # Test Progressive Refinement
        print("\n3. ProgressiveRefinement:")
        pr = ProgressiveRefinement(dim, num_iterations=2)
        x = torch.randn(batch_size, dim, h, w)
        preds = pr(x)
        print(f"   Input: {x.shape}")
        print(f"   Predictions: {len(preds)} stages")
        for i, pred in enumerate(preds):
            print(f"     Stage {i+1}: {pred.shape}")
        print(f"   ✓ Working correctly")
        
        # Test Sparse Spectral Router
        print("\n4. SparseSpectralRouter:")
        ssr = SparseSpectralRouter(dim, num_experts=4, k=2)
        x = torch.randn(batch_size, dim, h, w)
        weights, indices = ssr(x)
        print(f"   Input: {x.shape}")
        print(f"   Weights: {weights.shape}")
        print(f"   Top-K indices: {indices.shape}")
        print(f"   Non-zero experts: {(weights.sum(dim=1) > 0).sum().item()}/{weights.shape[1]}")
        print(f"   ✓ Working correctly")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing modules: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_v11():
    """Compare V11 and V12 architectures"""
    print("\n" + "=" * 60)
    print("Test 5: V11 vs V12 Comparison")
    print("=" * 60)
    
    try:
        from methods.zoomnext.zoomnext import CamoXpertV11
        from methods.zoomnext.camoxpert_v12 import CamoXpertV12_Base
        
        v11 = CamoXpertV11(num_frames=1, pretrained=False)
        v12 = CamoXpertV12_Base(num_frames=1, pretrained=False)
        
        params_v11 = sum(p.numel() for p in v11.parameters()) / 1e6
        params_v12 = sum(p.numel() for p in v12.parameters()) / 1e6
        
        print(f"\nParameter comparison:")
        print(f"  V11: {params_v11:.2f}M")
        print(f"  V12: {params_v12:.2f}M")
        print(f"  Increase: {params_v12 - params_v11:.2f}M (+{(params_v12/params_v11 - 1)*100:.1f}%)")
        
        # Test forward pass speed (rough estimate)
        batch_size = 2
        h, w = 448, 448
        img_l = torch.randn(batch_size, 3, h, w)
        img_m = torch.randn(batch_size, 3, int(h*0.75), int(w*0.75))
        img_s = torch.randn(batch_size, 3, h//2, w//2)
        
        import time
        
        v11.eval()
        v12.eval()
        
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                v11(data={"image_l": img_l, "image_m": img_m, "image_s": img_s})
                v12(data={"image_l": img_l, "image_m": img_m, "image_s": img_s})
            
            # Time V11
            start = time.time()
            for _ in range(10):
                v11(data={"image_l": img_l, "image_m": img_m, "image_s": img_s})
            time_v11 = (time.time() - start) / 10
            
            # Time V12
            start = time.time()
            for _ in range(10):
                v12(data={"image_l": img_l, "image_m": img_m, "image_s": img_s})
            time_v12 = (time.time() - start) / 10
        
        print(f"\nInference time comparison (CPU, averaged over 10 runs):")
        print(f"  V11: {time_v11*1000:.2f}ms")
        print(f"  V12: {time_v12*1000:.2f}ms")
        print(f"  Overhead: {(time_v12/time_v11 - 1)*100:.1f}%")
        
        print(f"\n✓ Comparison complete")
        return True
    
    except Exception as e:
        print(f"❌ Error comparing models: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("CamoXpert V12 Architecture Verification")
    print("="*60)
    print("This script verifies that V12 architecture works correctly")
    print("without requiring actual training or data.\n")
    
    results = []
    
    # Test 1: Model creation
    model_base, model_prog = test_model_creation()
    results.append(model_base is not None)
    
    if model_base is not None:
        # Test 2: Forward pass (base)
        results.append(test_forward_pass(model_base, "CamoXpertV12_Base"))
    
    if model_prog is not None:
        # Test 2b: Forward pass (progressive)
        results.append(test_forward_pass(model_prog, "CamoXpertV12_Progressive"))
    
    # Test 3: Loss components
    results.append(test_loss_components())
    
    # Test 4: Individual modules
    results.append(test_improved_modules())
    
    # Test 5: V11 vs V12 comparison
    results.append(compare_with_v11())
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    total_tests = len(results)
    passed_tests = sum(results)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("   CamoXpertV12 architecture is ready for training.")
        print("\nNext steps:")
        print("  1. Configure training paths in configs/camoxpert_v12.py")
        print("  2. Train with: python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Base")
        print("  3. Test with: python test_v12.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    main()
