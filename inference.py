#!/usr/bin/env python3
# inference.py

import os
import torch
from PIL import Image
from torchvision import transforms
import argparse

from networks import ResnetGenerator

def load_generator(checkpoint_path: str, device: torch.device):
    """
    Instantiate ResnetGenerator, load state dict from checkpoint.
    Returns: generator in eval() mode.
    """
    # Create model
    netG = ResnetGenerator(in_channels=3, out_channels=3, ngf=64, n_blocks=9).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(ckpt["netG_state_dict"])
    netG.eval()
    return netG


def preprocess_image(img_path: str, img_size: int, device: torch.device):
    """
    Load an RGB image, resize to (img_size×img_size), normalize to [-1,1], and return a torch tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return img_t


def postprocess_and_save(tensor: torch.Tensor, save_path: str):
    """
    Given a tensor in [-1,1], convert to [0,255] uint8, then save as PNG.
    """
    out = tensor.squeeze(0).cpu().detach()  # [3,H,W]
    out = (out + 1.0) * 127.5
    out = out.clamp(0, 255).permute(1, 2, 0).numpy().astype("uint8")
    img = Image.fromarray(out)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"✔ Sketch saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference: photo → sketch")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to generator checkpoint (e.g. checkpoints/pix2pix_cufs/checkpoint_epoch_100.pth)"
    )
    parser.add_argument(
        "--input_photo", type=str, required=True,
        help="Path to a single face photo (JPEG/PNG)."
    )
    parser.add_argument(
        "--output_sketch", type=str, required=True,
        help="Where to save the resulting sketch (PNG)."
    )
    parser.add_argument(
        "--img_size", type=int, default=256,
        help="Size to which photo is resized (default: 256)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = load_generator(args.checkpoint, device)

    img_t = preprocess_image(args.input_photo, args.img_size, device)

    with torch.no_grad():
        fake_sketch = netG(img_t)  # [1,3,256,256]

    postprocess_and_save(fake_sketch, args.output_sketch)


if __name__ == "__main__":
    main()
