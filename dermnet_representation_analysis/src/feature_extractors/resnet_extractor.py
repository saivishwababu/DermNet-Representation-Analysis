"""
ResNet50 Feature Extractor.
Extracts global average pooled features from pretrained ResNet50 (ImageNet).
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from ..preprocessing import create_dataloader

logger = logging.getLogger("dermnet_analysis")


class ResNet50Extractor:
    """Extract visual features using ResNet50 pretrained on ImageNet."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.model = None
        self.model_name = "resnet50"

    def load_model(self):
        """Load ResNet50 and remove the classification head."""
        logger.info("Loading ResNet50 (ImageNet pretrained)...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final FC layer — keep up to avgpool
        self.model = nn.Sequential(
            *list(resnet.children())[:-1],  # Everything up to and including avgpool
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"ResNet50 loaded on {self.device}")

    @torch.no_grad()
    def extract_features(
        self,
        image_paths: list,
        labels: list = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Extract ResNet50 features (global average pooled, 2048-d).

        Args:
            image_paths: List of image file paths.
            labels: Optional list of labels.
            batch_size: Batch size.
            num_workers: DataLoader workers.
            image_size: Input image size.

        Returns:
            Feature array of shape (N, 2048).
        """
        if self.model is None:
            self.load_model()

        dataloader = create_dataloader(
            image_paths,
            labels=labels,
            model_name="resnet50",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        all_features = []
        for batch in tqdm(dataloader, desc="ResNet50 extraction"):
            images = batch[0].to(self.device)
            features = self.model(images)       # (B, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
            all_features.append(features.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        logger.info(f"ResNet50 features extracted: {features.shape}")
        return features

    def extract_and_save(
        self,
        image_paths: list,
        labels: list,
        output_dir: str,
        force: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract features and save to disk. Skip if files exist and force=False.

        Returns:
            Feature array.
        """
        feature_path = os.path.join(output_dir, "resnet50_features.npy")

        if os.path.exists(feature_path) and not force:
            logger.info(f"Loading existing ResNet50 features from {feature_path}")
            return np.load(feature_path)

        features = self.extract_features(image_paths, labels, **kwargs)

        os.makedirs(output_dir, exist_ok=True)
        np.save(feature_path, features)
        logger.info(f"ResNet50 features saved to {feature_path}")

        return features
