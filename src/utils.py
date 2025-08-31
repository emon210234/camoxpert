import torch
import matplotlib.pyplot as plt

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def visualize_sample(image, mask, pred=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Image')
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')
    if pred is not None:
        plt.subplot(133)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')
    plt.show()
