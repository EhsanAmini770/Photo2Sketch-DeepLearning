#!/usr/bin/env python3
# monitor_training.py - Real-time training monitor

import os
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Optional

class TrainingMonitor:
    """Real-time training monitor that watches log files and checkpoints"""
    
    def __init__(self, checkpoint_dir: str, log_file: str = None, update_interval: float = 5.0):
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file or "log"
        self.update_interval = update_interval
        
        # Initialize data storage
        self.data = {
            'epochs': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'timestamps': []
        }
        
        # Setup real-time plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16)
        
        # Track last modification time
        self.last_modified = 0
        
    def parse_log_file(self) -> bool:
        """Parse log file and update data. Returns True if new data found."""
        if not os.path.exists(self.log_file):
            return False
        
        # Check if file was modified
        current_modified = os.path.getmtime(self.log_file)
        if current_modified <= self.last_modified:
            return False
        
        self.last_modified = current_modified
        
        # Parse the entire file (simple approach)
        epochs = []
        g_losses = []
        d_losses = []
        timestamps = []
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        # Pattern to match epoch summary lines
        epoch_pattern = r'→ Epoch (\d+) finished in [\d.]+s \| avgD: ([\d.]+) \| avgG: ([\d.]+)'
        
        for line in lines:
            match = re.search(epoch_pattern, line)
            if match:
                epoch = int(match.group(1))
                avg_d = float(match.group(2))
                avg_g = float(match.group(3))
                
                epochs.append(epoch)
                d_losses.append(avg_d)
                g_losses.append(avg_g)
                timestamps.append(datetime.now())
        
        # Update data if we have new information
        if len(epochs) > len(self.data['epochs']):
            self.data['epochs'] = epochs
            self.data['generator_loss'] = g_losses
            self.data['discriminator_loss'] = d_losses
            self.data['timestamps'] = timestamps
            return True
        
        return False
    
    def check_checkpoints(self) -> Dict[str, any]:
        """Check available checkpoints and return info"""
        if not os.path.exists(self.checkpoint_dir):
            return {'count': 0, 'latest': None, 'epochs': []}
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        epochs = []
        for f in checkpoint_files:
            try:
                epoch_num = int(f.replace('checkpoint_epoch_', '').replace('.pth', ''))
                epochs.append(epoch_num)
            except ValueError:
                continue
        
        epochs.sort()
        latest_epoch = max(epochs) if epochs else None
        
        return {
            'count': len(epochs),
            'latest': latest_epoch,
            'epochs': epochs
        }
    
    def update_plots(self):
        """Update all monitoring plots"""
        if len(self.data['epochs']) < 2:
            return
        
        epochs = self.data['epochs']
        g_loss = self.data['generator_loss']
        d_loss = self.data['discriminator_loss']
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Loss curves
        self.axes[0, 0].plot(epochs, g_loss, 'b-', linewidth=2, label='Generator')
        self.axes[0, 0].plot(epochs, d_loss, 'r-', linewidth=2, label='Discriminator')
        self.axes[0, 0].set_title('Training Loss Progress')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss ratio and balance
        if len(epochs) > 1:
            loss_ratio = np.array(g_loss) / np.array(d_loss)
            self.axes[0, 1].plot(epochs, loss_ratio, 'g-', linewidth=2)
            self.axes[0, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Balance')
            self.axes[0, 1].set_title('Generator/Discriminator Loss Ratio')
            self.axes[0, 1].set_xlabel('Epoch')
            self.axes[0, 1].set_ylabel('G_loss / D_loss')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Recent progress (last 20 epochs)
        recent_epochs = epochs[-20:] if len(epochs) > 20 else epochs
        recent_g = g_loss[-20:] if len(g_loss) > 20 else g_loss
        recent_d = d_loss[-20:] if len(d_loss) > 20 else d_loss
        
        self.axes[1, 0].plot(recent_epochs, recent_g, 'b-', linewidth=2, label='Generator')
        self.axes[1, 0].plot(recent_epochs, recent_d, 'r-', linewidth=2, label='Discriminator')
        self.axes[1, 0].set_title('Recent Progress (Last 20 Epochs)')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training statistics
        self.axes[1, 1].text(0.1, 0.9, f'Total Epochs: {len(epochs)}', transform=self.axes[1, 1].transAxes, fontsize=12)
        self.axes[1, 1].text(0.1, 0.8, f'Latest Epoch: {epochs[-1]}', transform=self.axes[1, 1].transAxes, fontsize=12)
        self.axes[1, 1].text(0.1, 0.7, f'Current G Loss: {g_loss[-1]:.4f}', transform=self.axes[1, 1].transAxes, fontsize=12)
        self.axes[1, 1].text(0.1, 0.6, f'Current D Loss: {d_loss[-1]:.4f}', transform=self.axes[1, 1].transAxes, fontsize=12)
        
        if len(g_loss) > 1:
            g_trend = "↓" if g_loss[-1] < g_loss[-2] else "↑"
            d_trend = "↓" if d_loss[-1] < d_loss[-2] else "↑"
            self.axes[1, 1].text(0.1, 0.5, f'G Trend: {g_trend}', transform=self.axes[1, 1].transAxes, fontsize=12)
            self.axes[1, 1].text(0.1, 0.4, f'D Trend: {d_trend}', transform=self.axes[1, 1].transAxes, fontsize=12)
        
        # Check checkpoint info
        checkpoint_info = self.check_checkpoints()
        self.axes[1, 1].text(0.1, 0.3, f'Checkpoints: {checkpoint_info["count"]}', transform=self.axes[1, 1].transAxes, fontsize=12)
        if checkpoint_info['latest']:
            self.axes[1, 1].text(0.1, 0.2, f'Latest Checkpoint: Epoch {checkpoint_info["latest"]}', transform=self.axes[1, 1].transAxes, fontsize=12)
        
        self.axes[1, 1].text(0.1, 0.1, f'Last Update: {datetime.now().strftime("%H:%M:%S")}', transform=self.axes[1, 1].transAxes, fontsize=10)
        self.axes[1, 1].set_title('Training Statistics')
        self.axes[1, 1].set_xlim(0, 1)
        self.axes[1, 1].set_ylim(0, 1)
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def run(self):
        """Run the monitoring loop"""
        print(f"🔍 Starting training monitor...")
        print(f"📁 Watching: {self.checkpoint_dir}")
        print(f"📄 Log file: {self.log_file}")
        print(f"⏱️  Update interval: {self.update_interval}s")
        print(f"🛑 Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                # Check for new data
                if self.parse_log_file():
                    print(f"📊 Updated data - Epoch {self.data['epochs'][-1]} | "
                          f"G: {self.data['generator_loss'][-1]:.4f} | "
                          f"D: {self.data['discriminator_loss'][-1]:.4f}")
                    self.update_plots()
                
                # Wait for next update
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            plt.ioff()
            print("👋 Monitor closed")

def main():
    parser = argparse.ArgumentParser(description="Real-time training monitor")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/pix2pix_cufs",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--log_file", type=str, default="log",
        help="Training log file to monitor"
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Update interval in seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = TrainingMonitor(
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        update_interval=args.interval
    )
    
    monitor.run()

if __name__ == "__main__":
    main()
