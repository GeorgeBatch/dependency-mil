################################################################################
# Imports
################################################################################

import os
import json

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

from source.constants import (
    ALL_IMG_NORMS,
    DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH,
)

STANDARD_INPUT_SIZE = 224

################################################################################
# Transforms
################################################################################

def transform_uniform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    return v2.Compose(
        [
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
        ]
    )


def transform_imagenet():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return v2.Compose(
        [
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
        ]
    )


def transform_none():
    return v2.Compose(
        [
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

################################################################################
# Transforms from LC25000-clean
################################################################################

def get_norm_constants(img_norm: str = 'imagenet'):
    # Source: https://github.com/mahmoodlab/UNI/blob/main/uni/get_encoder/get_encoder.py
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'openai_clip': {'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)},
        'H-optimus-0': {'mean': (0.707223, 0.578729, 0.703617), 'std': (0.211883, 0.230117, 0.177517)},  # taken from HuggingFace model card
        'hibou': {'mean': (0.7068, 0.5755, 0.722), 'std': (0.195, 0.2316, 0.181)},  # from AutoImageProcessor.from_pretrained("histai/hibou-b") or "histai/hibou-L"
    }
    try:
        constants = constants_zoo[img_norm]
    except KeyError as e:
        print(f"Key {e} not found in constants_zoo of `data.get_norm_constants()`. Trying to load from dataset-specific constants.")
        with open(DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH, 'r') as f:
            dataset_specific_constants = json.load(f)
            constants = dataset_specific_constants[img_norm]
        print(f"Succesfully loaded constants for {img_norm} from dataset-specific constants.")
    return constants.get('mean'), constants.get('std')


def get_data_transform(img_norm: str = 'imagenet', mean=None, std=None):
    """
    Returns a torchvision transform for preprocessing input data.

    Args:
        img_norm (str): The type of image normalization to apply. Defaults to 'imagenet'.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.

    Raises:
        AssertionError: If an invalid normalization type is provided.

    """
    if img_norm == 'resize_only':
        transform = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(STANDARD_INPUT_SIZE),
            # Normalize expects float input
            v2.ToDtype(torch.float32, scale=True),
        ])
    elif img_norm == 'manual':
        # used when mean and std are provided as arguments
        assert mean is not None and std is not None, "Mean and std must be provided for dataset-specific normalization."
        transform = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
        ])
    else:
        assert img_norm in ALL_IMG_NORMS, f"Invalid normalization type: {img_norm}. Should be one of {ALL_IMG_NORMS}."
        mean, std = get_norm_constants(img_norm)
        transform = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
        ])

    return transform