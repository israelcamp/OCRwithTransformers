from dataclasses import dataclass, field
import os
import random
from typing import Any
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision as tv


class SynthDataset(Dataset):
    def __init__(self, images_dir, annotation_file, height=32):
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.image_files = self._load_data()
        self.height = height

    def _load_data(self):
        with open(self.annotation_file, "r") as f:
            lines = f.read().splitlines()

        image_files = [line.split(" ")[0] for line in lines]
        return image_files

    def __len__(self):
        return len(self.image_files)

    def read_image_file_and_label(self, image_file):
        label = image_file.split("_")[1]
        image_path = os.path.join(self.images_dir, image_file)

        image = Image.open(image_path).convert("L")
        w, h = image.size
        ratio = w / float(h)
        nw = round(self.height * ratio)
        nw = min(nw, 400)

        image = image.resize((nw, self.height), Image.BICUBIC)

        return image, label

    def __getitem__(self, idx):
        image_file = self.image_files[idx]

        try:
            image, label = self.read_image_file_and_label(image_file)
        except:
            print(f"Error reading image {image_file} idx {idx}")
            return self.__getitem__(
                random.randint(0, len(self.image_files) - 1)
            )

        return image, label


class MaxPoolImagePad:
    def __init__(self):
        self.pool = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def __call__(self, x):
        return self.pool(x)


@dataclass
class SynthDataModule:
    train_dataset: Any = field(metadata="Training dataset")
    val_dataset: Any = field(metadata="Validation dataset")
    tokenizer: Any = field(metadata="tokenizer")
    train_bs: int = field(default=16, metadata="Training batch size")
    valid_bs: int = field(default=16, metadata="Eval batch size")
    num_workers: int = field(default=2)
    max_width: int = field(default=None)

    pooler: Any = field(default_factory=MaxPoolImagePad)

    @staticmethod
    def expand_image(img, h, w):
        expanded = Image.new("L", (w, h), color=(0,))  # black
        expanded.paste(img)
        return expanded

    def collate_fn(self, samples):
        images = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        image_widths = [im.width for im in images]
        max_width = (
            self.max_width if self.max_width is not None else max(image_widths)
        )

        attention_images = []
        for w in image_widths:
            attention_images.append([1] * w + [0] * (max_width - w))
        attention_images = self.pooler(
            torch.tensor(attention_images).float()
        ).long()

        h = images[0].size[1]
        to_tensor = tv.transforms.ToTensor()
        images = [
            to_tensor(self.expand_image(im, h=h, w=max_width)) for im in images
        ]

        tokens = self.tokenizer.batch_encode_plus(
            labels, padding="longest", return_tensors="pt"
        )
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")

        return torch.stack(images), input_ids, attention_mask, attention_images

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
