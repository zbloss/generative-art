import os
from glob import glob
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
import numpy as np

class CelebDataset(Dataset):
    
    def __init__(self, data_dir: str, image_size: int = 64):
        super().__init__()
        
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir), f'self.data_dir is not a directory: {self.data_dir}'
        self.transformation_stack = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_files = glob(os.path.join(self.data_dir, '*.jpg'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        image = self.transformation_stack(image)
        image = np.asarray(image)
        image = torch.from_numpy(image).float()
                
        return image

class CelebDataModule(LightningDataModule):
    
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
        assert os.path.isdir(self.data_dir), f'self.data_dir is not a directory: {self.data_dir}'

    def setup(self):
        
        dataset = CelebDataset(self.data_dir, self.image_size)
        number_of_images_in_dataset = len(dataset)
        number_of_images_for_training = int(number_of_images_in_dataset*0.8)
        number_of_images_for_testing = int(number_of_images_in_dataset*0.15)
        number_of_images_for_validation = number_of_images_in_dataset - \
                                          number_of_images_for_training - \
                                          number_of_images_for_testing
        
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset,
            (
                number_of_images_for_training, 
                number_of_images_for_testing, 
                number_of_images_for_validation
            )
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )