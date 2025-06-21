#!/usr/bin/env python3
# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CUFSPairedDataset(Dataset):
    """
    PyTorch Dataset for CUFS paired (photo, sketch) images.
    Expects:
      root/train_photos/  ← RGB face photos
      root/train_sketches/ ← corresponding pencil sketches
      root/test_photos/
      root/test_sketches/
    Filenames must match (e.g. "1.jpg" in both folders).
    """

    def __init__(self, root: str, mode: str = "train", img_size: int = 256):
        """
        root: path to "dataset/CUFS/"
        mode: "train" or "test"
        img_size: target size (square)
        """
        assert mode in ("train", "test"), "mode must be 'train' or 'test'"
        self.root = root
        self.mode = mode
        self.img_size = img_size

        self.photos_dir = os.path.join(root, f"{mode}_photos")
        self.sketches_dir = os.path.join(root, f"{mode}_sketches")

        self.filenames = sorted(os.listdir(self.photos_dir))
        # Filter out non-image files:
        self.filenames = [fn for fn in self.filenames if fn.lower().endswith((".jpg", ".png", ".jpeg"))]

        # Transforms: resize → random crop (train only) → toTensor → normalize
        if mode == "train":
            self.transform_photo = T.Compose([
                T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                # Optional: random jitter/crop for augmentation:
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
            ])
            self.transform_sketch = T.Compose([
                T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
            ])
        else:  # test
            self.transform_photo = T.Compose([
                T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
            ])
            self.transform_sketch = T.Compose([
                T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
            ])


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        fn = self.filenames[idx]
        photo_path = os.path.join(self.photos_dir, fn)
        sketch_path = os.path.join(self.sketches_dir, fn)

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")

        photo = self.transform_photo(photo)
        sketch = self.transform_sketch(sketch)

        return {"photo": photo, "sketch": sketch, "filename": fn}
