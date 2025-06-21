#!/usr/bin/env python3
# analyze_training.py - Comprehensive training analysis

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingAnalyzer:
    """Comprehensive analysis of training progress"""
    
    def __init__(self, log_file: str = "log", checkpoint_dir: str = None):
        self.log_file = log_file
        self.checkpoint_dir = checkpoint_dir
        self.data = self.parse_log_file()
        
    def parse_log_file(self) -> Dict[str, List[float]]:
        """Parse training log and extract all metrics"""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file '{self.log_file}' not found!")
            return {}
        
        epochs = []
        g_losses = []
        d_losses = []
        batch_losses_g = []
        batch_losses_d = []
        epoch_times = []
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        # Patterns for different log entries
        epoch_pattern = r'Epoch (\d+) finished in ([\d.]+)s \| avgD: ([\d.]+) \| avgG: ([\d.]+)'
        batch_pattern = r'\[Epoch \d+/\d+\] \[Batch \d+/\d+\] D_loss: ([\d.]+) \| G_loss: ([\d.]+)'
        
        for line in lines:
            # Parse epoch summaries
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                time_taken = float(epoch_match.group(2))
                avg_d = float(epoch_match.group(3))
                avg_g = float(epoch_match.group(4))
                
                epochs.append(epoch)
                epoch_times.append(time_taken)
                d_losses.append(avg_d)
                g_losses.append(avg_g)
            
            # Parse batch losses for detailed analysis
            batch_match = re.search(batch_pattern, line)
            if batch_match:
                d_loss = float(batch_match.group(1))
                g_loss = float(batch_match.group(2))
                batch_losses_d.append(d_loss)
                batch_losses_g.append(g_loss)
        
        return {
            'epochs': epochs,
            'generator_loss': g_losses,
            'discriminator_loss': d_losses,
            'epoch_times': epoch_times,
            'batch_g_losses': batch_losses_g,
            'batch_d_losses': batch_losses_d
        }
    
    def analyze_convergence(self) -> Dict[str, any]:
        """Analyze training convergence patterns"""
        if not self.data or len(self.data['epochs']) < 10:
            return {}
        
        epochs = self.data['epochs']
        g_loss = self.data['generator_loss']
        d_loss = self.data['discriminator_loss']
        
        # Calculate moving averages
        window_sizes = [5, 10, 20]
        convergence_analysis = {}
        
        for window in window_sizes:
            if len(g_loss) >= window:
                g_ma = np.convolve(g_loss, np.ones(window)/window, mode='valid')
                d_ma = np.convolve(d_loss, np.ones(window)/window, mode='valid')
                
                # Calculate variance in recent epochs
                recent_g_var = np.var(g_ma[-min(10, len(g_ma)):])
                recent_d_var = np.var(d_ma[-min(10, len(d_ma)):])
                
                convergence_analysis[f'window_{window}'] = {
                    'g_variance': recent_g_var,
                    'd_variance': recent_d_var,
                    'g_trend': 'decreasing' if g_ma[-1] < g_ma[0] else 'increasing',
                    'd_trend': 'decreasing' if d_ma[-1] < d_ma[0] else 'increasing'
                }
        
        # Find potential convergence point
        convergence_epoch = None
        if len(g_loss) > 20:
            # Look for point where loss variance becomes small
            for i in range(20, len(g_loss)):
                recent_g = g_loss[i-10:i]
                if np.std(recent_g) < 0.5:  # Threshold for convergence
                    convergence_epoch = epochs[i]
                    break
        
        return {
            'convergence_epoch': convergence_epoch,
            'window_analysis': convergence_analysis,
            'final_g_loss': g_loss[-1],
            'final_d_loss': d_loss[-1],
            'total_improvement_g': ((g_loss[0] - g_loss[-1]) / g_loss[0]) * 100,
            'total_improvement_d': ((d_loss[0] - d_loss[-1]) / d_loss[0]) * 100
        }
    
    def create_comprehensive_plots(self, save_dir: str = "analysis_plots"):
        """Create comprehensive analysis plots"""
        if not self.data or len(self.data['epochs']) < 2:
            print("❌ Insufficient data for plotting")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = self.data['epochs']
        g_loss = self.data['generator_loss']
        d_loss = self.data['discriminator_loss']
        
        # Create main analysis figure
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Main loss curves
        ax1 = plt.subplot(2, 4, 1)
        plt.plot(epochs, g_loss, 'b-', linewidth=2, label='Generator', alpha=0.8)
        plt.plot(epochs, d_loss, 'r-', linewidth=2, label='Discriminator', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Log scale losses
        ax2 = plt.subplot(2, 4, 2)
        plt.semilogy(epochs, g_loss, 'b-', linewidth=2, label='Generator')
        plt.semilogy(epochs, d_loss, 'r-', linewidth=2, label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Curves (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Loss ratio
        ax3 = plt.subplot(2, 4, 3)
        loss_ratio = np.array(g_loss) / np.array(d_loss)
        plt.plot(epochs, loss_ratio, 'g-', linewidth=2)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Balance')
        plt.xlabel('Epoch')
        plt.ylabel('G_loss / D_loss')
        plt.title('Loss Ratio (G/D)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Moving averages
        ax4 = plt.subplot(2, 4, 4)
        window = min(10, len(epochs) // 4)
        if window >= 3:
            g_ma = np.convolve(g_loss, np.ones(window)/window, mode='valid')
            d_ma = np.convolve(d_loss, np.ones(window)/window, mode='valid')
            epochs_ma = epochs[window-1:]
            plt.plot(epochs_ma, g_ma, 'b-', linewidth=2, label=f'Generator MA({window})')
            plt.plot(epochs_ma, d_ma, 'r-', linewidth=2, label=f'Discriminator MA({window})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Loss derivatives (rate of change)
        ax5 = plt.subplot(2, 4, 5)
        if len(g_loss) > 1:
            g_diff = np.diff(g_loss)
            d_diff = np.diff(d_loss)
            plt.plot(epochs[1:], g_diff, 'b-', linewidth=2, label='Generator Δ', alpha=0.7)
            plt.plot(epochs[1:], d_diff, 'r-', linewidth=2, label='Discriminator Δ', alpha=0.7)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.title('Loss Rate of Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Training time analysis
        ax6 = plt.subplot(2, 4, 6)
        if self.data.get('epoch_times'):
            times = self.data['epoch_times']
            plt.plot(epochs, times, 'orange', linewidth=2, marker='o', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.title('Training Time per Epoch')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(epochs, times, 1)
            p = np.poly1d(z)
            plt.plot(epochs, p(epochs), "orange", linestyle='--', alpha=0.7, label=f'Trend: {z[0]:.3f}s/epoch')
            plt.legend()
        
        # Plot 7: Loss distribution
        ax7 = plt.subplot(2, 4, 7)
        plt.hist(g_loss, bins=20, alpha=0.7, label='Generator', color='blue', density=True)
        plt.hist(d_loss, bins=20, alpha=0.7, label='Discriminator', color='red', density=True)
        plt.xlabel('Loss Value')
        plt.ylabel('Density')
        plt.title('Loss Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Stability analysis
        ax8 = plt.subplot(2, 4, 8)
        window = 10
        if len(g_loss) >= window:
            g_rolling_std = []
            d_rolling_std = []
            for i in range(window, len(g_loss)):
                g_rolling_std.append(np.std(g_loss[i-window:i]))
                d_rolling_std.append(np.std(d_loss[i-window:i]))
            
            plt.plot(epochs[window:], g_rolling_std, 'b-', linewidth=2, label='Generator Stability')
            plt.plot(epochs[window:], d_rolling_std, 'r-', linewidth=2, label='Discriminator Stability')
        plt.xlabel('Epoch')
        plt.ylabel('Rolling Std Dev')
        plt.title(f'Training Stability (Window={window})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = os.path.join(save_dir, "comprehensive_training_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✔ Comprehensive analysis saved to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def generate_report(self) -> str:
        """Generate a detailed text report"""
        if not self.data or len(self.data['epochs']) == 0:
            return "❌ No training data available for analysis"
        
        epochs = self.data['epochs']
        g_loss = self.data['generator_loss']
        d_loss = self.data['discriminator_loss']
        
        convergence = self.analyze_convergence()
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TRAINING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Log file: {self.log_file}")
        report.append("")
        
        # Basic statistics
        report.append("TRAINING OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total epochs trained: {len(epochs)}")
        report.append(f"Epoch range: {min(epochs)} - {max(epochs)}")
        if self.data.get('epoch_times'):
            total_time = sum(self.data['epoch_times'])
            avg_time = np.mean(self.data['epoch_times'])
            report.append(f"Total training time: {total_time/3600:.2f} hours")
            report.append(f"Average time per epoch: {avg_time:.1f} seconds")
        report.append("")
        
        # Loss analysis
        report.append("LOSS ANALYSIS")
        report.append("-" * 40)
        report.append(f"Generator Loss:")
        report.append(f"  Initial: {g_loss[0]:.4f}")
        report.append(f"  Final: {g_loss[-1]:.4f}")
        report.append(f"  Best (minimum): {min(g_loss):.4f} (Epoch {epochs[g_loss.index(min(g_loss))]})")
        report.append(f"  Improvement: {convergence.get('total_improvement_g', 0):.1f}%")
        report.append("")
        report.append(f"Discriminator Loss:")
        report.append(f"  Initial: {d_loss[0]:.4f}")
        report.append(f"  Final: {d_loss[-1]:.4f}")
        report.append(f"  Best (minimum): {min(d_loss):.4f} (Epoch {epochs[d_loss.index(min(d_loss))]})")
        report.append(f"  Improvement: {convergence.get('total_improvement_d', 0):.1f}%")
        report.append("")
        
        # Convergence analysis
        if convergence:
            report.append("CONVERGENCE ANALYSIS")
            report.append("-" * 40)
            if convergence.get('convergence_epoch'):
                report.append(f"Estimated convergence: Epoch {convergence['convergence_epoch']}")
            else:
                report.append("No clear convergence point detected")
            
            # Stability analysis
            if len(g_loss) > 20:
                recent_g_std = np.std(g_loss[-20:])
                recent_d_std = np.std(d_loss[-20:])
                report.append(f"Recent stability (last 20 epochs):")
                report.append(f"  Generator std dev: {recent_g_std:.4f}")
                report.append(f"  Discriminator std dev: {recent_d_std:.4f}")
        
        # Recommendations
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if len(g_loss) > 10:
            recent_g_trend = np.mean(g_loss[-5:]) - np.mean(g_loss[-10:-5])
            recent_d_trend = np.mean(d_loss[-5:]) - np.mean(d_loss[-10:-5])
            
            if recent_g_trend > 0:
                report.append("WARNING: Generator loss is increasing recently - consider:")
                report.append("   - Reducing learning rate")
                report.append("   - Adjusting L1 loss weight")
                report.append("   - Checking for mode collapse")
            else:
                report.append("GOOD: Generator loss is decreasing - training progressing well")

            if abs(recent_d_trend) < 0.01:
                report.append("GOOD: Discriminator loss is stable")
            elif recent_d_trend > 0.05:
                report.append("WARNING: Discriminator loss increasing - generator may be winning")
            
            # Loss balance
            final_ratio = g_loss[-1] / d_loss[-1]
            if final_ratio > 100:
                report.append("WARNING: Generator loss much higher than discriminator - consider:")
                report.append("   - Reducing discriminator learning rate")
                report.append("   - Increasing generator training frequency")
            elif final_ratio < 10:
                report.append("GOOD: Good loss balance between generator and discriminator")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Analyze training progress")
    parser.add_argument(
        "--log_file", type=str, default="log",
        help="Training log file to analyze"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/pix2pix_cufs",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--save_dir", type=str, default="analysis_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--report_only", action="store_true",
        help="Generate only text report, no plots"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TrainingAnalyzer(args.log_file, args.checkpoint_dir)
    
    if not analyzer.data:
        print("ERROR: No training data found!")
        return
    
    # Generate report
    print("📋 Generating analysis report...")
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, "training_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Generate plots unless report_only
    if not args.report_only:
        print("\n📊 Creating comprehensive plots...")
        analyzer.create_comprehensive_plots(args.save_dir)

if __name__ == "__main__":
    main()
