#!/usr/bin/env python3
# discriminator_evaluation.py - Use trained discriminator to evaluate if sketch is "fake" or "real"

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import argparse
import numpy as np

from networks import PatchDiscriminator

def load_discriminator(checkpoint_path: str, device: torch.device):
    """
    Load the trained discriminator from checkpoint.
    Returns: discriminator in eval() mode.
    """
    # Create discriminator (input is 6 channels: photo + sketch concatenated)
    netD = PatchDiscriminator(in_channels=3, ndf=64, n_layers=3).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    netD.load_state_dict(ckpt["netD_state_dict"])
    netD.eval()
    return netD

def preprocess_image(img_path: str, img_size: int, device: torch.device):
    """
    Load an RGB image, resize to (img_size×img_size), normalize to [-1,1], and return a torch tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return img_t

def evaluate_with_discriminator(photo_path: str, sketch_path: str, checkpoint_path: str, 
                              img_size: int = 256) -> dict:
    """
    Use the trained discriminator to evaluate if the (photo, sketch) pair looks real or fake.
    
    Args:
        photo_path: Path to the original photo
        sketch_path: Path to the generated sketch
        checkpoint_path: Path to the trained model checkpoint
        img_size: Image size for processing
    
    Returns:
        Dictionary with evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load discriminator
    print(f"Loading discriminator from: {checkpoint_path}")
    netD = load_discriminator(checkpoint_path, device)
    
    # Load and preprocess images
    print(f"Loading photo: {photo_path}")
    photo_tensor = preprocess_image(photo_path, img_size, device)
    
    print(f"Loading sketch: {sketch_path}")
    sketch_tensor = preprocess_image(sketch_path, img_size, device)
    
    # Concatenate photo and sketch (as the discriminator expects)
    # During training, discriminator sees: [photo, real_sketch] for real pairs
    # and [photo, fake_sketch] for fake pairs
    input_tensor = torch.cat((photo_tensor, sketch_tensor), dim=1)  # [1, 6, H, W]
    
    with torch.no_grad():
        # Get discriminator output
        output = netD(input_tensor)  # [1, 1, H', W'] patch map
        
        # Convert to probability using sigmoid
        prob_map = torch.sigmoid(output)
        
        # Calculate statistics
        mean_prob = prob_map.mean().item()
        max_prob = prob_map.max().item()
        min_prob = prob_map.min().item()
        std_prob = prob_map.std().item()
        
        # Get patch-wise statistics
        patch_shape = prob_map.shape[2:]  # (H', W')
        total_patches = patch_shape[0] * patch_shape[1]
        
        # Count patches that think it's real (>0.5)
        real_patches = (prob_map > 0.5).sum().item()
        fake_patches = total_patches - real_patches
        
        real_percentage = (real_patches / total_patches) * 100
        fake_percentage = (fake_patches / total_patches) * 100
    
    results = {
        'mean_probability': mean_prob,
        'max_probability': max_prob,
        'min_probability': min_prob,
        'std_probability': std_prob,
        'patch_shape': patch_shape,
        'total_patches': total_patches,
        'real_patches': real_patches,
        'fake_patches': fake_patches,
        'real_percentage': real_percentage,
        'fake_percentage': fake_percentage,
        'overall_prediction': 'REAL' if mean_prob > 0.5 else 'FAKE',
        'confidence': abs(mean_prob - 0.5) * 2  # 0 = uncertain, 1 = very confident
    }
    
    return results

def interpret_discriminator_results(results: dict) -> str:
    """
    Interpret the discriminator results and provide human-readable assessment.
    """
    mean_prob = results['mean_probability']
    confidence = results['confidence']
    real_percentage = results['real_percentage']
    
    interpretation = []
    
    # Overall assessment
    if mean_prob > 0.8:
        interpretation.append("🟢 HIGHLY REALISTIC - Discriminator strongly believes this is a real sketch!")
    elif mean_prob > 0.6:
        interpretation.append("🟡 REALISTIC - Discriminator thinks this is likely a real sketch.")
    elif mean_prob > 0.4:
        interpretation.append("🟠 UNCERTAIN - Discriminator is unsure if this is real or fake.")
    elif mean_prob > 0.2:
        interpretation.append("🔴 LIKELY FAKE - Discriminator thinks this is probably generated.")
    else:
        interpretation.append("🔴 CLEARLY FAKE - Discriminator strongly believes this is generated.")
    
    # Confidence assessment
    if confidence > 0.8:
        interpretation.append(f"   Confidence: VERY HIGH ({confidence:.1%})")
    elif confidence > 0.6:
        interpretation.append(f"   Confidence: HIGH ({confidence:.1%})")
    elif confidence > 0.4:
        interpretation.append(f"   Confidence: MODERATE ({confidence:.1%})")
    else:
        interpretation.append(f"   Confidence: LOW ({confidence:.1%})")
    
    # Patch analysis
    if real_percentage > 80:
        interpretation.append(f"   Patch Analysis: {real_percentage:.1f}% of patches look real - EXCELLENT!")
    elif real_percentage > 60:
        interpretation.append(f"   Patch Analysis: {real_percentage:.1f}% of patches look real - GOOD")
    elif real_percentage > 40:
        interpretation.append(f"   Patch Analysis: {real_percentage:.1f}% of patches look real - MIXED")
    else:
        interpretation.append(f"   Patch Analysis: {real_percentage:.1f}% of patches look real - POOR")
    
    return "\n".join(interpretation)

def print_detailed_report(results: dict, photo_path: str, sketch_path: str):
    """Print a detailed discriminator evaluation report"""
    
    print("\n" + "="*80)
    print("🤖 DISCRIMINATOR EVALUATION REPORT")
    print("="*80)
    
    print(f"\n📁 Original Photo: {os.path.basename(photo_path)}")
    print(f"🎨 Generated Sketch: {os.path.basename(sketch_path)}")
    
    print(f"\n🎯 DISCRIMINATOR VERDICT: {results['overall_prediction']}")
    print(f"📊 Mean Probability: {results['mean_probability']:.3f}")
    print(f"🎲 Confidence Level: {results['confidence']:.1%}")
    
    print(f"\n📈 DETAILED STATISTICS:")
    print(f"   • Mean Probability: {results['mean_probability']:.3f}")
    print(f"   • Max Probability:  {results['max_probability']:.3f}")
    print(f"   • Min Probability:  {results['min_probability']:.3f}")
    print(f"   • Std Deviation:    {results['std_probability']:.3f}")
    
    print(f"\n🔍 PATCH ANALYSIS:")
    print(f"   • Patch Grid Size: {results['patch_shape'][0]}×{results['patch_shape'][1]}")
    print(f"   • Total Patches: {results['total_patches']}")
    print(f"   • 'Real' Patches: {results['real_patches']} ({results['real_percentage']:.1f}%)")
    print(f"   • 'Fake' Patches: {results['fake_patches']} ({results['fake_percentage']:.1f}%)")
    
    print(f"\n🧠 INTERPRETATION:")
    interpretation = interpret_discriminator_results(results)
    for line in interpretation.split('\n'):
        print(f"   {line}")
    
    print(f"\n💡 WHAT THIS MEANS:")
    if results['overall_prediction'] == 'REAL':
        print("   ✅ Your generated sketch successfully fooled the discriminator!")
        print("   ✅ This indicates high-quality generation that looks realistic.")
        print("   ✅ The AI model has learned to create convincing sketches.")
    else:
        print("   ⚠️  The discriminator detected that this sketch is generated.")
        print("   ⚠️  This could mean the sketch has some artificial characteristics.")
        print("   ⚠️  However, this doesn't necessarily mean it's a bad sketch!")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate sketch using trained discriminator")
    parser.add_argument(
        "--photo", type=str, required=True,
        help="Path to the original photo"
    )
    parser.add_argument(
        "--sketch", type=str, required=True,
        help="Path to the generated sketch"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--img_size", type=int, default=256,
        help="Image size for processing (default: 256)"
    )
    
    args = parser.parse_args()
    
    try:
        # Evaluate with discriminator
        results = evaluate_with_discriminator(
            args.photo, args.sketch, args.checkpoint, args.img_size
        )
        
        # Print detailed report
        print_detailed_report(results, args.photo, args.sketch)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
