"""
Preprocessing module for DermNet representation analysis.
Handles image deduplication, transforms, and PyTorch Dataset creation.
"""

import os
import hashlib
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger("dermnet_analysis")


# ---- Deduplication ----

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def remove_duplicates(df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Remove exact duplicate images based on SHA-256 hash.

    Args:
        df: DataFrame with 'image_path' column.
        output_path: Path to save the duplicates report CSV.

    Returns:
        DataFrame with duplicates removed.
    """
    logger.info("Computing SHA-256 hashes for duplicate detection...")
    hashes = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Hashing images"):
        try:
            h = compute_file_hash(row["image_path"])
        except Exception as e:
            logger.warning(f"Could not hash {row['image_path']}: {e}")
            h = None
        hashes.append(h)

    df = df.copy()
    df["file_hash"] = hashes

    # Find duplicates
    duplicated_mask = df.duplicated(subset="file_hash", keep="first")
    duplicates = df[duplicated_mask]

    if len(duplicates) > 0:
        logger.info(f"Found {len(duplicates)} duplicate images")
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            duplicates[["image_path", "label", "filename", "file_hash"]].to_csv(
                output_path, index=False
            )
            logger.info(f"Duplicate report saved to {output_path}")
        df = df[~duplicated_mask].reset_index(drop=True)
    else:
        logger.info("No duplicate images found")
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(columns=["image_path", "label", "filename", "file_hash"]).to_csv(
                output_path, index=False
            )

    # Drop the hash column
    df = df.drop(columns=["file_hash"])
    return df


# ---- Transforms ----

def get_transform(model_name: str, image_size: int = 224) -> transforms.Compose:
    """
    Get the appropriate preprocessing transform for each model.

    Args:
        model_name: One of "dinov2", "resnet50", "clip"
        image_size: Target image size.

    Returns:
        torchvision.transforms.Compose
    """
    # ImageNet normalization (used by DINOv2 and ResNet50)
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # CLIP normalization
    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    if model_name in ("dinov2", "resnet50"):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    elif model_name == "clip":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            clip_normalize,
        ])
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ---- PyTorch Dataset ----

class DermNetDataset(Dataset):
    """
    PyTorch Dataset for loading dermatological images.
    Handles corrupt images gracefully by returning a black image.
    """

    def __init__(self, image_paths: list, labels: list = None, transform=None):
        """
        Args:
            image_paths: List of image file paths.
            labels: Optional list of string labels.
            transform: torchvision transform to apply.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a black placeholder for corrupt images
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx], idx
        return img, idx


def create_dataloader(
    image_paths: list,
    labels: list = None,
    model_name: str = "dinov2",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for feature extraction.

    Args:
        image_paths: List of image paths.
        labels: Optional list of labels.
        model_name: Model name for selecting the right transform.
        image_size: Input image size.
        batch_size: Batch size for DataLoader.
        num_workers: Number of worker processes.

    Returns:
        DataLoader instance.
    """
    transform = get_transform(model_name, image_size)
    dataset = DermNetDataset(image_paths, labels, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
