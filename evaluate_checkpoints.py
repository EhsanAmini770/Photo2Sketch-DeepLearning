#!/usr/bin/env python3
# evaluate_checkpoints.py

import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from networks import ResnetGenerator

def load_generator(ckpt_path, device):
    netG = ResnetGenerator(in_channels=3, out_channels=3, ngf=64, n_blocks=9).to(device)
    state = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(state["netG_state_dict"])
    netG.eval()
    return netG

def compute_avg_l1_on_test(netG, device, img_size=256):
    """
    Returns average L1 between netG outputs and ground-truth sketches
    over the entire dataset/CUFS/test_photos directory.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    test_photos_dir  = "dataset/CUFS/test_photos"
    test_sketches_dir= "dataset/CUFS/test_sketches"
    filenames = sorted(os.listdir(test_photos_dir))
    total_l1 = 0.0
    count = 0

    with torch.no_grad():
        for fn in filenames:
            if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            # Load and preprocess
            img_photo  = Image.open(os.path.join(test_photos_dir, fn)).convert("RGB")
            img_sketch = Image.open(os.path.join(test_sketches_dir, fn)).convert("RGB")
            t_photo  = transform(img_photo).unsqueeze(0).to(device)
            t_sketch = transform(img_sketch).unsqueeze(0).to(device)

            # Generate and compute L1
            pred_sketch = netG(t_photo)            # in [-1,1]
            pred = (pred_sketch + 1) * 0.5         # scale→[0,1]
            gt   = (t_sketch      + 1) * 0.5
            l1 = torch.nn.functional.l1_loss(pred, gt, reduction="mean").item()

            total_l1 += l1
            count += 1

    return total_l1 / count


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Change this list to whatever epochs you want to evaluate:
    epochs_to_eval = list(range(90, 101))  # 90, 91, …, 100

    results = {}
    for ep in epochs_to_eval:
        ckpt_path = f"checkpoints/pix2pix_cufs/checkpoint_epoch_{ep}.pth"
        if not os.path.exists(ckpt_path):
            continue

        print(f"▶ Evaluating Epoch {ep} …")
        netG = load_generator(ckpt_path, device)
        avg_l1 = compute_avg_l1_on_test(netG, device, img_size=256)
        print(f"    → Epoch {ep}: Avg L1 on test set = {avg_l1:.4f}")
        results[ep] = avg_l1

    # Find which epoch had the minimum avg L1
    best_epoch = min(results, key=lambda e: results[e])
    print("\n✔ Best checkpoint on test set:")
    print(f"    Epoch {best_epoch} with Avg L1 = {results[best_epoch]:.4f}")
