import argparse
from src.config import load_config, update_config
from src.dataset import COD10KDataset
from src.model import build_edgenext
from torch.utils.data import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--root_dir', type=str, help='Path to the COD10K-v3 dataset root')
    parser.add_argument('--test_model', action='store_true', help='Test the model architecture')
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = update_config(args, config)
    
    # Override root_dir if provided via command line
    if args.root_dir:
        config['data']['root_dir'] = args.root_dir
    
    if args.test_model:
        # Test the model architecture
        model = build_edgenext('small')
        print("EdgeNeXt-S model created successfully!")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Test with a dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        features, output = model(dummy_input)
        
        print("Feature shapes:")
        for i, feat in enumerate(features):
            print(f"Stage {i+1}: {feat.shape}")
        
        print(f"Classification output shape: {output.shape}")
        return
    
    # Create datasets
    train_dataset = COD10KDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        img_size=config['data']['img_size'],
        augment=True
    )
    
    val_dataset = COD10KDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        img_size=config['data']['img_size'],
        augment=False
    )
    
    test_dataset = COD10KDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        img_size=config['data']['img_size'],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test data loading
    for images, masks in train_loader:
        print(f"Batch - Images: {images.shape}, Masks: {masks.shape}")
        break

if __name__ == '__main__':
    main()
