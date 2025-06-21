#!/usr/bin/env python3
# download_dataset.py

import os
import urllib.request
import tarfile
import sys

DATASET_URL = (
    "https://github.com/chaofengc/Face-Sketch-SCG/"
    "releases/download/v0.1/dataset.tgz"
)
OUTPUT_DIR = "dataset"
ARCHIVE_PATH = os.path.join(OUTPUT_DIR, "dataset.tgz")
CUFS_SUBDIR = "CUFS"


def download_dataset(url: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"▶ Downloading dataset archive to '{dst_path}' …")
    urllib.request.urlretrieve(url, dst_path)
    print("✔ Download complete.")


def extract_only_cufs(archive_path: str, output_folder: str, subdir: str):
    print(f"▶ Extracting only '{subdir}/' from archive …")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(f"dataset/{subdir}/")]
        if not members:
            print(f"⚠ No members found under 'dataset/{subdir}/' in the archive.")
            sys.exit(1)

        for m in members:
            # Strip off the leading "dataset/" so we end up with folders under output_folder/subdir
            m.name = m.name.split("dataset/", 1)[1]
            tar.extract(m, path=output_folder)

    print(f"✔ Extraction complete: '{output_folder}/{subdir}/' now exists.")


def main():
    # 1) Download if necessary
    if not os.path.exists(ARCHIVE_PATH):
        download_dataset(DATASET_URL, ARCHIVE_PATH)
    else:
        print(f"ℹ Archive already exists at '{ARCHIVE_PATH}'; skipping download.")

    # 2) Extract only CUFS
    extract_only_cufs(ARCHIVE_PATH, OUTPUT_DIR, CUFS_SUBDIR)

    # 3) Remove the .tgz file to save space
    print(f"▶ Removing '{ARCHIVE_PATH}' …")
    os.remove(ARCHIVE_PATH)
    print("✔ Done.\n"
          f"You now have:\n"
          f"  dataset/{CUFS_SUBDIR}/\n"
          f"    ├── train_photos/\n"
          f"    ├── train_sketches/\n"
          f"    ├── test_photos/\n"
          f"    └── test_sketches/")


if __name__ == "__main__":
    main()
