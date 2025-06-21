#!/usr/bin/env python3
"""
Example script demonstrating how to use the Photo to Sketch model
"""

import os
import torch
from PIL import Image
import argparse

from inference import load_generator, preprocess_image, postprocess_and_save

def main():
    """
    Example usage of the photo to sketch conversion model
    """
    
    # Default paths
    default_checkpoint = "checkpoints/pix2pix_cufs_enhanced/checkpoint_epoch_100.pth"
    default_output_dir = "example_outputs"
    
    parser = argparse.ArgumentParser(description="Example Photo to Sketch Conversion")
    parser.add_argument(
        "--checkpoint", type=str, default=default_checkpoint,
        help=f"Path to model checkpoint (default: {default_checkpoint})"
    )
    parser.add_argument(
        "--input_photo", type=str, required=True,
        help="Path to input photo"
    )
    parser.add_argument(
        "--output_dir", type=str, default=default_output_dir,
        help=f"Output directory (default: {default_output_dir})"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Please download or train a model first.")
        return
    
    # Check if input photo exists
    if not os.path.exists(args.input_photo):
        print(f"❌ Input photo not found: {args.input_photo}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    print("📥 Loading model...")
    try:
        netG = load_generator(args.checkpoint, device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process image
    print("🖼️  Processing image...")
    try:
        # Generate output filename
        input_name = os.path.splitext(os.path.basename(args.input_photo))[0]
        output_path = os.path.join(args.output_dir, f"{input_name}_sketch.png")
        
        # Preprocess
        img_tensor = preprocess_image(args.input_photo, 256, device)
        
        # Generate sketch
        with torch.no_grad():
            sketch_tensor = netG(img_tensor)
        
        # Save result
        postprocess_and_save(sketch_tensor, output_path)
        
        print(f"✅ Sketch generated successfully!")
        print(f"📁 Output saved to: {output_path}")
        
        # Display some info about the result
        result_img = Image.open(output_path)
        print(f"📊 Output size: {result_img.size}")
        print(f"📊 Output mode: {result_img.mode}")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return

if __name__ == "__main__":
    main()
