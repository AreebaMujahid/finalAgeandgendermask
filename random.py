import os
import torch


def load_checkpoint(filepath):
    if os.path.exists(filepath):
        print("Loading Checkpoint")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint Loaded Successfully")
    else:
        print(f"No checkpoint found at {filepath}")

# Example usage
checkpoint_path = r'C:\Users\Laiba Ahmad\Downloads\tridentNN_epoch20.pth'
load_checkpoint(checkpoint_path)
