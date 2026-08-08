"""
Lightweight embedding network for Stage 2 re-ranking.

Kept deliberately small -- this plays the same architectural role as the
"compact, ultra-low-latency AI Adjudicator" in the original design: it does
not replace the physics-like coarse search, it supervises/disambiguates its
output. Trained self-supervised (no manual labels beyond what the dataset
generator already recorded as ground truth) with a triplet-margin loss:
  anchor   = reference patch (downsampled to PATCH px)
  positive = the true search-image crop at gt_x, gt_y
  negative = a same-periodicity-phase distractor crop elsewhere in the array
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PATCH = 64
EMBED_DIM = 32


class EmbedNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),   # 64->32
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16->8
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        z = self.conv(x).flatten(1)
        z = self.fc(z)
        return F.normalize(z, dim=1)


def patch_to_tensor(patch_uint8):
    """patch_uint8: 2D numpy array (any size, grayscale) -> (1,1,PATCH,PATCH) tensor in [-1, 1]."""
    pil = Image.fromarray(np.clip(patch_uint8, 0, 255).astype(np.uint8))
    pil = pil.resize((PATCH, PATCH), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def batch_to_tensor(patches):
    """List of 2D numpy arrays -> (N,1,PATCH,PATCH) tensor."""
    return torch.cat([patch_to_tensor(p) for p in patches], dim=0)


def embed(model, patches, device='cpu'):
    with torch.no_grad():
        x = batch_to_tensor(patches).to(device)
        return model(x).cpu().numpy()
