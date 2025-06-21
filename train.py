#!/usr/bin/env python3
# train_with_visualization.py - Enhanced training script with real-time visualization

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from dataset import CUFSPairedDataset
from networks import ResnetGenerator, PatchDiscriminator

# ────────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (tweak as needed)
# ────────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "dataset/CUFS"
SAVE_ROOT = "checkpoints"
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 100
LR = 2e-4
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_L1 = 100.0

SAVE_EVERY = 1
PLOT_EVERY = 5  # Update plots every N epochs
LOG_EVERY = 50  # Log batch progress every N batches


class TrainingLogger:
    """Enhanced logging and visualization for training"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.log_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize metrics storage
        self.metrics = {
            "epochs": [],
            "generator_loss": [],
            "discriminator_loss": [],
            "generator_gan_loss": [],
            "generator_l1_loss": [],
            "batch_times": [],
            "epoch_times": [],
        }

        # Setup logging file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"training_log_{timestamp}.txt")
        self.metrics_file = os.path.join(self.log_dir, "training_metrics.json")

        # Initialize plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Real-time Training Progress")

    def log_message(self, message: str):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)

        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def log_epoch(
        self,
        epoch: int,
        g_loss: float,
        d_loss: float,
        g_gan_loss: float,
        g_l1_loss: float,
        epoch_time: float,
    ):
        """Log epoch metrics"""
        self.metrics["epochs"].append(epoch)
        self.metrics["generator_loss"].append(g_loss)
        self.metrics["discriminator_loss"].append(d_loss)
        self.metrics["generator_gan_loss"].append(g_gan_loss)
        self.metrics["generator_l1_loss"].append(g_l1_loss)
        self.metrics["epoch_times"].append(epoch_time)

        # Save metrics to JSON
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Log to console and file
        self.log_message(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Time: {epoch_time:.1f}s | "
            f"G_loss: {g_loss:.4f} | "
            f"D_loss: {d_loss:.4f} | "
            f"G_GAN: {g_gan_loss:.4f} | "
            f"G_L1: {g_l1_loss:.4f}"
        )

    def update_plots(self):
        """Update real-time training plots"""
        if len(self.metrics["epochs"]) < 2:
            return

        epochs = self.metrics["epochs"]
        g_loss = self.metrics["generator_loss"]
        d_loss = self.metrics["discriminator_loss"]
        g_gan_loss = self.metrics["generator_gan_loss"]
        g_l1_loss = self.metrics["generator_l1_loss"]

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Plot 1: Generator vs Discriminator Loss
        self.axes[0, 0].plot(epochs, g_loss, "b-", label="Generator", linewidth=2)
        self.axes[0, 0].plot(epochs, d_loss, "r-", label="Discriminator", linewidth=2)
        self.axes[0, 0].set_title("Generator vs Discriminator Loss")
        self.axes[0, 0].set_xlabel("Epoch")
        self.axes[0, 0].set_ylabel("Loss")
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Generator Loss Components
        self.axes[0, 1].plot(epochs, g_gan_loss, "g-", label="GAN Loss", linewidth=2)
        self.axes[0, 1].plot(epochs, g_l1_loss, "m-", label="L1 Loss", linewidth=2)
        self.axes[0, 1].set_title("Generator Loss Components")
        self.axes[0, 1].set_xlabel("Epoch")
        self.axes[0, 1].set_ylabel("Loss")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Loss Smoothed (Moving Average)
        if len(epochs) >= 5:
            window = min(5, len(epochs))
            g_smooth = np.convolve(g_loss, np.ones(window) / window, mode="valid")
            d_smooth = np.convolve(d_loss, np.ones(window) / window, mode="valid")
            epochs_smooth = epochs[window - 1 :]

            self.axes[1, 0].plot(
                epochs_smooth, g_smooth, "b-", label="Generator (MA)", linewidth=2
            )
            self.axes[1, 0].plot(
                epochs_smooth, d_smooth, "r-", label="Discriminator (MA)", linewidth=2
            )
        else:
            self.axes[1, 0].plot(epochs, g_loss, "b-", label="Generator", linewidth=2)
            self.axes[1, 0].plot(
                epochs, d_loss, "r-", label="Discriminator", linewidth=2
            )

        self.axes[1, 0].set_title("Smoothed Loss Curves")
        self.axes[1, 0].set_xlabel("Epoch")
        self.axes[1, 0].set_ylabel("Loss")
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Training Time per Epoch
        if len(self.metrics["epoch_times"]) > 0:
            self.axes[1, 1].plot(
                epochs, self.metrics["epoch_times"], "orange", linewidth=2
            )
            self.axes[1, 1].set_title("Training Time per Epoch")
            self.axes[1, 1].set_xlabel("Epoch")
            self.axes[1, 1].set_ylabel("Time (seconds)")
            self.axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)  # Small pause to update the plot

        # Save plot
        plot_path = os.path.join(
            self.log_dir, f"training_progress_epoch_{epochs[-1]}.png"
        )
        self.fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    def save_final_plots(self):
        """Save final comprehensive plots"""
        if len(self.metrics["epochs"]) == 0:
            return

        # Save final plot
        final_plot_path = os.path.join(self.log_dir, "final_training_curves.png")
        self.fig.savefig(final_plot_path, dpi=300, bbox_inches="tight")

        self.log_message(f"Final training plots saved to: {final_plot_path}")
        self.log_message(f"Training metrics saved to: {self.metrics_file}")


def init_weights(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_checkpoint(netG, netD, optimG, optimD, epoch, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "netG_state_dict": netG.state_dict(),
            "netD_state_dict": netD.state_dict(),
            "optimG_state_dict": optimG.state_dict(),
            "optimD_state_dict": optimD.state_dict(),
        },
        os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
    )


def train_with_visualization():
    """Enhanced training function with real-time visualization"""

    # Setup save directory
    save_dir = os.path.join(SAVE_ROOT, "pix2pix_cufs_enhanced")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(save_dir)
    logger.log_message("Starting enhanced training with visualization")
    logger.log_message(f"Device: {DEVICE}")
    logger.log_message(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

    # Prepare dataset
    train_dataset = CUFSPairedDataset(DATA_ROOT, mode="train", img_size=IMG_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    logger.log_message(f"Dataset loaded: {len(train_loader)} batches per epoch")

    # Initialize networks
    netG = ResnetGenerator(in_channels=3, out_channels=3, ngf=64, n_blocks=9).to(DEVICE)
    netD = PatchDiscriminator(in_channels=3, ndf=64, n_layers=3).to(DEVICE)

    init_weights(netG)
    init_weights(netD)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    # Optimizers
    optimG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

    logger.log_message("Networks and optimizers initialized")

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        running_G_loss = 0.0
        running_D_loss = 0.0
        running_G_GAN_loss = 0.0
        running_G_L1_loss = 0.0

        for i, data in enumerate(train_loader):
            real_photo = data["photo"].to(DEVICE)
            real_sketch = data["sketch"].to(DEVICE)

            # Update Discriminator
            netD.zero_grad()

            # Real pair
            real_input_D = torch.cat((real_photo, real_sketch), dim=1)
            output_real = netD(real_input_D)
            real_labels = torch.ones_like(output_real)
            loss_D_real = criterion_GAN(output_real, real_labels)

            # Fake pair
            with torch.no_grad():
                fake_sketch = netG(real_photo)
            fake_input_D = torch.cat((real_photo, fake_sketch.detach()), dim=1)
            output_fake = netD(fake_input_D)
            fake_labels = torch.zeros_like(output_fake)
            loss_D_fake = criterion_GAN(output_fake, fake_labels)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimD.step()

            # Update Generator
            netG.zero_grad()
            fake_sketch = netG(real_photo)
            fake_input_for_G = torch.cat((real_photo, fake_sketch), dim=1)
            output_D_for_G = netD(fake_input_for_G)
            real_labels_for_G = torch.ones_like(output_D_for_G)

            loss_G_GAN = criterion_GAN(output_D_for_G, real_labels_for_G)
            loss_G_L1 = criterion_L1(fake_sketch, real_sketch) * LAMBDA_L1
            loss_G = 0.5 * loss_G_GAN + loss_G_L1

            loss_G.backward()
            optimG.step()

            # Accumulate losses
            running_G_loss += loss_G.item()
            running_D_loss += loss_D.item()
            running_G_GAN_loss += loss_G_GAN.item()
            running_G_L1_loss += loss_G_L1.item()

            # Log batch progress
            if (i + 1) % LOG_EVERY == 0:
                logger.log_message(
                    f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i+1}/{len(train_loader)}] "
                    f"D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f} | "
                    f"G_GAN: {loss_G_GAN.item():.4f} | G_L1: {loss_G_L1.item():.4f}"
                )

        # Calculate epoch averages
        epoch_time = time.time() - epoch_start_time
        avg_G = running_G_loss / len(train_loader)
        avg_D = running_D_loss / len(train_loader)
        avg_G_GAN = running_G_GAN_loss / len(train_loader)
        avg_G_L1 = running_G_L1_loss / len(train_loader)

        # Log epoch results
        logger.log_epoch(epoch, avg_G, avg_D, avg_G_GAN, avg_G_L1, epoch_time)

        # Update plots
        if epoch % PLOT_EVERY == 0:
            logger.update_plots()

        # Save checkpoint
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(netG, netD, optimG, optimD, epoch, save_dir)
            logger.log_message(f"Checkpoint saved: epoch {epoch}")

    # Save final plots and cleanup
    logger.save_final_plots()
    logger.log_message("Training completed successfully!")

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show final plot


if __name__ == "__main__":
    train_with_visualization()
