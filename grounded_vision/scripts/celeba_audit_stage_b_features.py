"""Stage B: DINOv2 ViT-S/14 CLS features for exported jpgs (MPS/CPU).

Usage: python stage_b_features.py <split>
Writes: feats_<split>.npy (n, 384) float32
"""
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).parent


def main(split: str):
    out = RUN / f"feats_{split}.npy"
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    files = sorted((RUN / f"imgs_{split}").glob("img_*.jpg"))
    print(f"{split}: {len(files)} images on {device}")
    from PIL import Image
    feats = np.zeros((len(files), 384), dtype=np.float32)
    B = 128
    with torch.no_grad():
        for i in range(0, len(files), B):
            batch = [tfm(Image.open(f).convert("RGB")) for f in files[i:i + B]]
            x = torch.stack(batch).to(device)
            feats[i:i + len(batch)] = model(x).cpu().numpy()
            if (i // B) % 20 == 0:
                print(f"  {i + len(batch)}/{len(files)}")
    np.save(out, feats)
    print(f"[done] {out.name} {feats.shape}")


if __name__ == "__main__":
    main(sys.argv[1])
