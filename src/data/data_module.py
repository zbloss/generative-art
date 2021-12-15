import os
from glob import glob
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 64):
        super().__init__()

        self.data_dir = data_dir
        assert os.path.isdir(
            self.data_dir
        ), f"self.data_dir is not a directory: {self.data_dir}"
        self.transformation_stack = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
                transforms.transforms.ColorJitter(hue=0.5),
                transforms.transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        jpg_image_files = glob(os.path.join(self.data_dir, "**/*.jpg"), recursive=True)
        png_image_files = glob(os.path.join(self.data_dir, "**/*.png"), recursive=True)
        self.image_files = jpg_image_files + png_image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_file = self.image_files[idx]
        image = Image.open(image_file).convert("RGB")
        image = self.transformation_stack(image)
        image = np.asarray(image)
        image = torch.from_numpy(image).float()

        return image


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 1,
        image_size: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def prepare_data(self):
        assert os.path.isdir(
            self.data_dir
        ), f"self.data_dir is not a directory: {self.data_dir}"

    def setup(self):

        dataset = ImageDataset(self.data_dir, self.image_size)
        number_of_images_in_dataset = len(dataset)
        number_of_images_for_training = int(number_of_images_in_dataset * 0.7)
        number_of_images_for_testing = int(number_of_images_in_dataset * 0.15)
        number_of_images_for_validation = (
            number_of_images_in_dataset
            - number_of_images_for_training
            - number_of_images_for_testing
        )

        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset,
            (
                number_of_images_for_training,
                number_of_images_for_testing,
                number_of_images_for_validation,
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True
        )
