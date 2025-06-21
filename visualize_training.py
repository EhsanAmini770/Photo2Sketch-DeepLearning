#!/usr/bin/env python3
# visualize_training.py

import re
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, List, Tuple, Optional

def load_enhanced_metrics(metrics_file: str) -> Optional[Dict[str, List[float]]]:
    """
    Load enhanced training metrics from JSON file.
    Returns None if file doesn't exist or is invalid.
    """
    if not os.path.exists(metrics_file):
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['epochs', 'generator_loss', 'discriminator_loss']
        if not all(field in data for field in required_fields):
            return None

        return data
    except (json.JSONDecodeError, KeyError):
        return None

def parse_log_file(log_path: str) -> Dict[str, List[float]]:
    """
    Parse the training log file to extract loss values.
    Returns a dictionary with epoch numbers, G_loss, and D_loss values.
    """
    epochs = []
    g_losses = []
    d_losses = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Pattern to match epoch summary lines like:
    # → Epoch 1 finished in 34.9s | avgD: 0.3918 | avgG: 21.5677
    # The arrow is actually Unicode characters, so we'll match the text after it
    epoch_pattern = r'Epoch (\d+) finished in [\d.]+s \| avgD: ([\d.]+) \| avgG: ([\d.]+)'

    for line in lines:
        match = re.search(epoch_pattern, line)
        if match:
            epoch = int(match.group(1))
            avg_d = float(match.group(2))
            avg_g = float(match.group(3))

            epochs.append(epoch)
            d_losses.append(avg_d)
            g_losses.append(avg_g)

    return {
        'epochs': epochs,
        'generator_loss': g_losses,
        'discriminator_loss': d_losses
    }

def load_training_data(log_path: str, enhanced_dir: str = None) -> Dict[str, List[float]]:
    """
    Load training data from either enhanced JSON metrics or legacy log file.
    Prioritizes enhanced metrics if available.
    """
    # Try to load enhanced metrics first
    if enhanced_dir:
        metrics_file = os.path.join(enhanced_dir, "logs", "training_metrics.json")
        enhanced_data = load_enhanced_metrics(metrics_file)
        if enhanced_data:
            print(f"✔ Loaded enhanced metrics from: {metrics_file}")
            return enhanced_data

    # Fall back to parsing log file
    if os.path.exists(log_path):
        print(f"✔ Parsing legacy log file: {log_path}")
        return parse_log_file(log_path)

    print(f"❌ No training data found!")
    return {}

def plot_training_curves(data: Dict[str, List[float]], save_path: str = None, show_plot: bool = True):
    """
    Create comprehensive training loss plots.
    """
    epochs = data['epochs']
    g_loss = data['generator_loss']
    d_loss = data['discriminator_loss']

    # Check if we have enhanced metrics
    has_enhanced = 'generator_gan_loss' in data and 'generator_l1_loss' in data
    has_timing = 'epoch_times' in data

    # Create figure with appropriate subplots
    if has_enhanced:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Enhanced Training Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()  # Flatten for easier indexing
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()  # Flatten for easier indexing
        fig.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Generator Loss
    axes[0].plot(epochs, g_loss, 'b-', linewidth=2, label='Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Generator Loss')
    axes[0].set_title('Generator Loss Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Discriminator Loss
    axes[1].plot(epochs, d_loss, 'r-', linewidth=2, label='Discriminator Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Discriminator Loss')
    axes[1].set_title('Discriminator Loss Over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Combined Loss Plot
    axes[2].plot(epochs, g_loss, 'b-', linewidth=2, label='Generator Loss', alpha=0.8)
    axes[2].plot(epochs, d_loss, 'r-', linewidth=2, label='Discriminator Loss', alpha=0.8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Generator vs Discriminator Loss')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Plot 4: Moving Average or Enhanced Components
    if has_enhanced:
        # Plot Generator Loss Components
        g_gan_loss = data['generator_gan_loss']
        g_l1_loss = data['generator_l1_loss']
        axes[3].plot(epochs, g_gan_loss, 'g-', linewidth=2, label='GAN Loss', alpha=0.8)
        axes[3].plot(epochs, g_l1_loss, 'm-', linewidth=2, label='L1 Loss', alpha=0.8)
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss')
        axes[3].set_title('Generator Loss Components')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        # Plot 5: Training Time (if available)
        if has_timing:
            epoch_times = data['epoch_times']
            axes[4].plot(epochs, epoch_times, 'orange', linewidth=2, marker='o', markersize=3)
            axes[4].set_xlabel('Epoch')
            axes[4].set_ylabel('Time (seconds)')
            axes[4].set_title('Training Time per Epoch')
            axes[4].grid(True, alpha=0.3)

        # Plot 6: Loss Ratio
        loss_ratio = np.array(g_loss) / np.array(d_loss)
        axes[5].plot(epochs, loss_ratio, 'purple', linewidth=2)
        axes[5].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Balance')
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('G_loss / D_loss')
        axes[5].set_title('Loss Ratio (G/D)')
        axes[5].grid(True, alpha=0.3)
        axes[5].legend()
    else:
        # Plot 4: Moving Average for legacy format
        window_size = 5
        if len(g_loss) >= window_size:
            g_loss_ma = np.convolve(g_loss, np.ones(window_size)/window_size, mode='valid')
            d_loss_ma = np.convolve(d_loss, np.ones(window_size)/window_size, mode='valid')
            epochs_ma = epochs[window_size-1:]

            axes[3].plot(epochs_ma, g_loss_ma, 'b-', linewidth=2, label=f'Generator MA({window_size})')
            axes[3].plot(epochs_ma, d_loss_ma, 'r-', linewidth=2, label=f'Discriminator MA({window_size})')
        else:
            axes[3].plot(epochs, g_loss, 'b-', linewidth=2, label='Generator Loss')
            axes[3].plot(epochs, d_loss, 'r-', linewidth=2, label='Discriminator Loss')

        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss')
        axes[3].set_title('Smoothed Loss Curves')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create directory if there is one
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✔ Training curves saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig

def print_training_statistics(data: Dict[str, List[float]]):
    """
    Print comprehensive training statistics.
    """
    epochs = data['epochs']
    g_loss = data['generator_loss']
    d_loss = data['discriminator_loss']

    # Check if enhanced metrics are available
    has_enhanced = 'generator_gan_loss' in data and 'generator_l1_loss' in data
    has_timing = 'epoch_times' in data

    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    if has_enhanced:
        print("(Enhanced Metrics)")
    print("="*60)

    print(f"Total Epochs Trained: {len(epochs)}")
    print(f"Epoch Range: {min(epochs)} - {max(epochs)}")

    # Timing information
    if has_timing:
        epoch_times = data['epoch_times']
        total_time = sum(epoch_times)
        avg_time = np.mean(epoch_times)
        print(f"Total Training Time: {total_time/3600:.2f} hours")
        print(f"Average Time per Epoch: {avg_time:.1f} seconds")

    print(f"\nGenerator Loss:")
    print(f"  Initial: {g_loss[0]:.4f}")
    print(f"  Final: {g_loss[-1]:.4f}")
    print(f"  Minimum: {min(g_loss):.4f} (Epoch {epochs[g_loss.index(min(g_loss))]})")
    print(f"  Maximum: {max(g_loss):.4f} (Epoch {epochs[g_loss.index(max(g_loss))]})")
    print(f"  Average: {np.mean(g_loss):.4f}")
    print(f"  Std Dev: {np.std(g_loss):.4f}")

    # Enhanced generator loss breakdown
    if has_enhanced:
        g_gan_loss = data['generator_gan_loss']
        g_l1_loss = data['generator_l1_loss']
        print(f"\n  Generator Loss Components:")
        print(f"    GAN Loss - Final: {g_gan_loss[-1]:.4f}, Average: {np.mean(g_gan_loss):.4f}")
        print(f"    L1 Loss - Final: {g_l1_loss[-1]:.4f}, Average: {np.mean(g_l1_loss):.4f}")
        print(f"    L1/Total Ratio: {(g_l1_loss[-1]/g_loss[-1])*100:.1f}%")

    print(f"\nDiscriminator Loss:")
    print(f"  Initial: {d_loss[0]:.4f}")
    print(f"  Final: {d_loss[-1]:.4f}")
    print(f"  Minimum: {min(d_loss):.4f} (Epoch {epochs[d_loss.index(min(d_loss))]})")
    print(f"  Maximum: {max(d_loss):.4f} (Epoch {epochs[d_loss.index(max(d_loss))]})")
    print(f"  Average: {np.mean(d_loss):.4f}")
    print(f"  Std Dev: {np.std(d_loss):.4f}")

    # Calculate improvement
    g_improvement = ((g_loss[0] - g_loss[-1]) / g_loss[0]) * 100
    d_improvement = ((d_loss[0] - d_loss[-1]) / d_loss[0]) * 100

    print(f"\nImprovement:")
    print(f"  Generator Loss: {g_improvement:.1f}% reduction")
    print(f"  Discriminator Loss: {d_improvement:.1f}% {'reduction' if d_improvement > 0 else 'increase'}")

    # Loss balance analysis
    final_ratio = g_loss[-1] / d_loss[-1]
    print(f"\nLoss Balance:")
    print(f"  Final G/D Ratio: {final_ratio:.1f}")
    if final_ratio > 100:
        print("  Status: Generator struggling (ratio too high)")
    elif final_ratio < 10:
        print("  Status: Well balanced")
    else:
        print("  Status: Acceptable balance")

    # Find convergence point (where losses stabilize)
    if len(g_loss) > 20:
        last_20_g = g_loss[-20:]
        last_20_d = d_loss[-20:]
        g_stability = np.std(last_20_g)
        d_stability = np.std(last_20_d)

        print(f"\nStability (last 20 epochs):")
        print(f"  Generator Loss Std Dev: {g_stability:.4f}")
        print(f"  Discriminator Loss Std Dev: {d_stability:.4f}")

        if g_stability < 0.5:
            print("  Generator: Converged (stable)")
        else:
            print("  Generator: Still improving")

        if d_stability < 0.05:
            print("  Discriminator: Converged (stable)")
        else:
            print("  Discriminator: Still adapting")

    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Visualize training progress from log file or enhanced metrics")
    parser.add_argument(
        "--log_file", type=str, default="log",
        help="Path to the training log file (default: log)"
    )
    parser.add_argument(
        "--enhanced_dir", type=str, default="checkpoints/pix2pix_cufs_enhanced",
        help="Path to enhanced training directory with JSON metrics"
    )
    parser.add_argument(
        "--save_path", type=str, default="training_curves.png",
        help="Path to save the plot (default: training_curves.png)"
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Don't display the plot, only save it"
    )
    parser.add_argument(
        "--stats_only", action="store_true",
        help="Only print statistics, don't create plots"
    )
    parser.add_argument(
        "--legacy_only", action="store_true",
        help="Force use of legacy log file parsing only"
    )

    args = parser.parse_args()

    # Load training data (enhanced or legacy)
    print(f"📊 Loading training data...")
    if args.legacy_only:
        if not os.path.exists(args.log_file):
            print(f"❌ Error: Log file '{args.log_file}' not found!")
            return
        data = parse_log_file(args.log_file)
    else:
        data = load_training_data(args.log_file, args.enhanced_dir)

    if not data or not data.get('epochs'):
        print("❌ Error: No training data found!")
        return

    print(f"✔ Found training data for {len(data['epochs'])} epochs")

    # Check if enhanced metrics are available
    has_enhanced = 'generator_gan_loss' in data and 'generator_l1_loss' in data
    if has_enhanced:
        print("✔ Enhanced metrics detected - showing detailed analysis")
    else:
        print("ℹ Using legacy format - basic analysis only")

    # Print statistics
    print_training_statistics(data)

    # Create plots unless stats_only is specified
    if not args.stats_only:
        print(f"\n📈 Creating training curves...")
        plot_training_curves(
            data,
            save_path=args.save_path,
            show_plot=not args.no_show
        )

if __name__ == "__main__":
    main()
