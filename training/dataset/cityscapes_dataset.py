import logging
import random
from copy import deepcopy
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from PIL import Image as PILImage
from pathlib import Path

MAX_RETRIES = 100


class CityscapesDataset(VisionDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transforms=None,
        multiplier=1,
        target_segments_available=True,
    ):
        """
        Args:
            root_dir (str): Root directory of the Cityscapes dataset.
            split (str): Dataset split, 'train', 'val', or 'test'.
            transforms (callable, optional): Transformations to be applied to the data.
            multiplier (int): Repeat factor for dataset samples.
            target_segments_available (bool): If True, ensures that target masks are always provided.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms or transforms.Compose([])
        self.target_segments_available = target_segments_available

        # Load the image and annotation file paths
        self.image_dir = self.root_dir / "leftImg8bit" / self.split
        self.label_dir = self.root_dir / "gtFine" / self.split
        self.image_paths = sorted(self.image_dir.glob("**/*_leftImg8bit.png"))
        self.label_paths = sorted(self.label_dir.glob("**/*_instanceIds.png"))

        assert len(self.image_paths) == len(
            self.label_paths
        ), "Mismatch between image and label counts."

        self.repeat_factors = torch.ones(len(self.image_paths), dtype=torch.float32)
        self.repeat_factors *= multiplier

    def _get_datapoint(self, idx):
        
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = PILImage.open(image_path).convert("RGB")
        label = PILImage.open(label_path)

        datapoint = self.construct(image, label, image_path.name)
        # print(f"+++++++++++: {datapoint['image'].shape}, Label shape: {datapoint['masks'].shape}")
        
        if isinstance(self.transforms, list):
            for transform in self.transforms:
                datapoint = transform(datapoint)
        elif callable(self.transforms):
            datapoint = self.transforms(datapoint)
        return datapoint
            
    def construct(self, image, label, file_name):
        """
        Constructs a single datapoint.
        Args:
            image (PILImage): RGB image.
            label (PILImage): Instance segmentation label.
            file_name (str): File name of the image.
        Returns:
            dict: A dictionary containing the image, label, and metadata.
        """
        # # Convert to tensors
        # image_tensor = F.to_tensor(image)
        # label_tensor = torch.from_numpy(np.array(label, dtype=np.int32))

        # print(type(image), type(label)) #<class 'PIL.Image.Image'> <class 'PIL.PngImagePlugin.PngImageFile'>
        # print(f"Image type: {type(image)}, Image mode: {image.mode}, Image size: {image.size}")
        # print(f"Label type: {type(label)}, Label mode: {label.mode}, Label size: {label.size}")

        # image_array = np.array(image)  # Convert RGB image to NumPy array
        # label_array = np.array(label, dtype=np.int32)  # Explicit dtype for the label
        
        #Image shape: (1024, 2048, 3), Label shape: (1024, 2048)

        #resize image and label
        image_array = np.array(image.resize((1024, 1024)))                   #H, W, 3
        #change to 3, H, W
        # change (3, H, W) to (1, 3 , H, W)

        label_array = np.array(label.resize((1024, 1024), PILImage.NEAREST)) #H, W
        label_array = np.expand_dims(label_array, axis=0)


        #label array should be bool type
        # label_array = label_array.astype(np.bool)
        #dummy label array to figure out the shape
        # label_array


        # print(f"Image shape: {image_array.shape}, Label shape: {label_array.shape}")

        


        # Metadata
        height, width = 1024, 1024  # Cityscapes image dimensions


        return {
            "image": image_array,
            "masks": label_array,
            "file_name": file_name,
            "size": (height, width),
        }

    def __getitem__(self, idx):
        # print("idx", idx)
        

        # print("-------------------------------")
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    # Example usage

    
    dataset = CityscapesDataset(
        root_dir="/media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes",
        split="train",
        transforms=None,
        multiplier=1,
    )
    print(f"Loaded Cityscapes dataset with {len(dataset)} samples.")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}")

    #test get dara point
    sample = dataset._get_datapoint(0)
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}")

    #test construct
    sample = dataset.construct(sample['image'], sample['label'], sample['file_name'])
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}")
