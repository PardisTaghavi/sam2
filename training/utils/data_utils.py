# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)
    
@tensorclass
class BatchedImageDatapoint:
    """
    This class represents a batch of images with associated annotations and metadata.
    Attributes:
        img_batch: A [BxCxHxW] tensor containing the image data for each image in the batch.
        labels: A [BxHxW] tensor containing the segmentation labels for each image in the batch.
        metadata: Metadata about the batch, including file names and image sizes.
        dict_key: A string key used to identify the batch.
    """
    
    img_batch: torch.FloatTensor  # [B, C, H, W]
    masks: torch.IntTensor       # [B, H, W]
    metadata: dict
    dict_key: str

    def pin_memory(self, device=None):
        """Pins memory for faster data transfer to GPU."""
        return self.apply(torch.Tensor.pin_memory, device=device)
    
    @property
    def num_frames(self) -> int:
        """Returns the number of frames per video."""
        return 1

    @property
    def batch_size(self) -> int:
        """Returns the batch size (number of images in the batch)."""
        return self.img_batch.shape[0]

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Flattens the img_batch tensor to shape [(B)xCxHxW] into [(B)x(HxW)xC]
        """
        return self.img_batch#.flatten(2).transpose(1, 2)

    @property
    def flat_labels(self) -> torch.IntTensor:
        """
        Flattens the labels tensor to shape [B, H, W] into [B, H*W]
        """
        # return self.labels#.flatten(1)
        return self.masks


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0 
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )

import torch
from typing import List, Dict

from typing import List
import torch

def collate_fn_img(
    batch: List[Dict],
    dict_key: str
) -> BatchedImageDatapoint:
    """
    Collates a batch of data from image dataset.
    
    Args:
        batch: A list of dictionaries, each containing an 'image', 'label', 'file_name', and 'size'.
        dict_key (str): A string key used to identify the batch.

    Returns:
        A BatchedImageDatapoint containing the batched images, labels, and metadata.
    """
    
    # Initialize lists to store the batch data
    img_batch = []
    labels_batch = []
    metadata = []
    
    # Loop over each item in the batch
    for sample in batch:
        image_array = sample["image"]
        label_array = sample["masks"]
        
        # Convert image and label to tensors
        img_batch.append(torch.tensor(image_array).permute(2, 0, 1))  # Convert to CxHxW
        labels_batch.append(torch.tensor(label_array))  # Keep labels as HxW
        
        
        # Collect metadata such as file name and size
        metadata.append({
            "file_name": sample["file_name"],
            "size": sample["size"]
        })
    
    # Stack images and labels into tensors
    img_batch = torch.stack(img_batch, dim=0)  # Shape [B, C, H, W]
    labels_batch = torch.stack(labels_batch, dim=0)  # Shape [B, H, W]
    print(f"Imagexxxxxxxxxxxxxxxxx shape: {img_batch.shape}, Label shape: {labels_batch.shape}")
    
    # Return the BatchedImageDatapoint object
    return BatchedImageDatapoint(
        img_batch=img_batch,
        masks=labels_batch,
        metadata=metadata,
        dict_key=dict_key
    )
