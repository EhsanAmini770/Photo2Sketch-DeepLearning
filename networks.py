#!/usr/bin/env python3
# networks.py

import torch
import torch.nn as nn

###############################################################################
# Helper blocks:  
###############################################################################

def conv2d_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=True, activation=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv2d_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=True, activation=True, dropout=False):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    if dropout:
        layers.append(nn.Dropout(0.5))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

###############################################################################
# Generator:  (ResNet‐9blocks)  
###############################################################################

class ResnetGenerator(nn.Module):
    """
    Input: 3×256×256 face photo  
    Output: 3×256×256 sketch  
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9):
        super().__init__()
        # Initial conv + downsampling:
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            # Downsampling:
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        ]

        # ResNet blocks:
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * 4)]

        # Upsampling:
        model += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


###############################################################################
# PatchGAN Discriminator (70×70)  
###############################################################################

class PatchDiscriminator(nn.Module):
    """
    Input: concatenated (photo, sketch) → 6×256×256  
    Output: patch validity map (1×N×N)  
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        layers = []
        # First conv (no norm on first layer):
        layers += [
            conv2d_block(in_channels * 2, ndf, kernel_size=4, stride=2, padding=1, norm=False, activation=True)
        ]
        # Middle layers:
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                conv2d_block(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=(1 if n == n_layers - 1 else 2),
                    padding=1,
                    norm=True,
                    activation=True
                )
            ]

        # Final conv:
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)  # no activation / norm
        ]

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)  # patch map (batch × 1 × H′ × W′)
