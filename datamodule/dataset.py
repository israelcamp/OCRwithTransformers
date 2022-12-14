import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image
from pydantic import DirectoryPath, PositiveInt
from torch.utils.data import Dataset

# Utils


def get_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    np_image = np.array(pil_image)
    return np_image


# Dataset


@dataclass
class TextRecDataset(Dataset):

    images_dir: DirectoryPath = field(metadata="Dir of images")
    img2label: dict = field(metadata="Dict mapping images to labels")
    height: PositiveInt = field(default=32, metadata="Height of images")
    tfms: Any = field(default=None, metadata="Image augmentations")

    def __post_init__(
        self,
    ):
        super().__init__()
        self.labeltuple = sorted(
            [(k, v) for k, v in self.img2label.items()], key=lambda x: x[0]
        )

    def __len__(
        self,
    ):
        return len(self.labeltuple)

    @staticmethod
    def expand_image(img, h, w):
        expanded = Image.new("RGB", (w, h), color=3 * (255,))  # white
        expanded.paste(img)
        return expanded

    def get_image(self, image_name: str):
        image_path = os.path.join(self.images_dir, f"{image_name}.png")
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        ratio = self.height / h  # how the height will change
        nw = round(w * ratio)

        image = image.resize((nw, self.height))

        if nw < 40:
            image = self.expand_image(image, self.height, 40)

        if self.tfms is not None:
            image = np.array(image)
            image = self.tfms(image=image)
            image = Image.fromarray(image)
        return image

    def __getitem__(self, idx):
        image_name, label = self.labeltuple[idx]
        image = self.get_image(image_name)
        outputs = (image, label)
        return outputs


@dataclass
class TestTextRecDataset(TextRecDataset):
    def __getitem__(self, idx):
        image_name, label = self.labeltuple[idx]
        image = self.get_image(image_name)
        outputs = (image, label, image_name)
        return outputs
