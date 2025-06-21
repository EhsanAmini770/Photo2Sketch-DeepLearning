#!/usr/bin/env python3
"""
Setup script for Photo to Sketch Conversion project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="photo-to-sketch",
    version="1.0.0",
    author="Ehsan Amini",
    author_email="your-email@example.com",
    description="Deep Learning based Photo to Sketch Conversion using GANs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/photo-to-sketch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "photo2sketch=inference:main",
            "train-photo2sketch=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
