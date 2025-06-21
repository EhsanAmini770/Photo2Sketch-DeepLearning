#!/usr/bin/env python3
# generate_samples.py - Generate sample sketches from test photos

import os
import torch
from PIL import Image
from torchvision import transforms
import argparse
import glob
from networks import ResnetGenerator

def load_generator(checkpoint_path: str, device: torch.device):
    """Load the generator from checkpoint"""
    netG = ResnetGenerator(in_channels=3, out_channels=3, ngf=64, n_blocks=9).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(ckpt["netG_state_dict"])
    netG.eval()
    return netG

def preprocess_image(img_path: str, img_size: int, device: torch.device):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t, img

def postprocess_tensor(tensor: torch.Tensor):
    """Convert tensor back to PIL Image"""
    out = tensor.squeeze(0).cpu().detach()
    out = (out + 1.0) * 127.5
    out = out.clamp(0, 255).permute(1, 2, 0).numpy().astype("uint8")
    return Image.fromarray(out)

def create_comparison_image(original: Image.Image, sketch: Image.Image, save_path: str):
    """Create side-by-side comparison image"""
    width, height = original.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original, (0, 0))
    comparison.paste(sketch, (width, 0))
    comparison.save(save_path)
    return comparison

def generate_samples(checkpoint_path: str, test_photos_dir: str, output_dir: str, 
                    num_samples: int = 5, img_size: int = 256):
    """Generate sample sketches from test photos"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load generator
    print(f"Loading generator from: {checkpoint_path}")
    netG = load_generator(checkpoint_path, device)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    sketches_dir = os.path.join(output_dir, "sketches")
    comparisons_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(sketches_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Get test photos
    photo_files = glob.glob(os.path.join(test_photos_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_photos_dir, "*.png")) + \
                  glob.glob(os.path.join(test_photos_dir, "*.jpeg"))
    
    if not photo_files:
        print(f"No photos found in {test_photos_dir}")
        return
    
    # Limit to requested number of samples
    photo_files = photo_files[:num_samples]
    
    print(f"Generating sketches for {len(photo_files)} photos...")
    
    with torch.no_grad():
        for i, photo_path in enumerate(photo_files):
            print(f"Processing {i+1}/{len(photo_files)}: {os.path.basename(photo_path)}")
            
            # Load and preprocess image
            img_tensor, original_img = preprocess_image(photo_path, img_size, device)
            
            # Generate sketch
            sketch_tensor = netG(img_tensor)
            sketch_img = postprocess_tensor(sketch_tensor)
            
            # Save individual sketch
            base_name = os.path.splitext(os.path.basename(photo_path))[0]
            sketch_path = os.path.join(sketches_dir, f"{base_name}_sketch.png")
            sketch_img.save(sketch_path)
            
            # Create and save comparison
            comparison_path = os.path.join(comparisons_dir, f"{base_name}_comparison.png")
            create_comparison_image(original_img, sketch_img, comparison_path)
            
            print(f"  Saved: {sketch_path}")
            print(f"  Saved: {comparison_path}")
    
    print(f"\nGeneration complete! Results saved to: {output_dir}")
    print(f"  Individual sketches: {sketches_dir}")
    print(f"  Side-by-side comparisons: {comparisons_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate sample sketches from test photos")
    parser.add_argument(
        "--checkpoint", type=str, 
        default="checkpoints/pix2pix_cufs/checkpoint_epoch_90.pth",
        help="Path to generator checkpoint"
    )
    parser.add_argument(
        "--test_photos", type=str,
        default="dataset/CUFS/test_photos",
        help="Directory containing test photos"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="sample_results",
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples to generate (default: 5)"
    )
    parser.add_argument(
        "--img_size", type=int, default=256,
        help="Image size for processing (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Check if test photos directory exists
    if not os.path.exists(args.test_photos):
        print(f"Error: Test photos directory not found: {args.test_photos}")
        return
    
    # Generate samples
    generate_samples(
        checkpoint_path=args.checkpoint,
        test_photos_dir=args.test_photos,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        img_size=args.img_size
    )

if __name__ == "__main__":
    main()
