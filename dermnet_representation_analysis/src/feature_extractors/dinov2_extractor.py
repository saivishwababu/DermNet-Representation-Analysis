"""
DINOv2 Feature Extractor.
Extracts CLS token embeddings from the pretrained DINOv2 ViT-B/14 model.
"""

import os
import logging
import numpy as np
import torch
from tqdm import tqdm

from ..preprocessing import create_dataloader

logger = logging.getLogger("dermnet_analysis")


class DINOv2Extractor:
    """Extract visual features using DINOv2 (ViT-B/14)."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.model = None
        self.model_name = "dinov2"

    def load_model(self):
        """Load the DINOv2 model from torch hub."""
        logger.info("Loading DINOv2 ViT-B/14 model...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitb14",
            pretrained=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"DINOv2 loaded on {self.device}")

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
        Extract DINOv2 CLS token features for all images.

        Args:
            image_paths: List of image file paths.
            labels: Optional list of labels.
            batch_size: Batch size.
            num_workers: DataLoader workers.
            image_size: Input image size.

        Returns:
            Feature array of shape (N, 768).
        """
        if self.model is None:
            self.load_model()

        dataloader = create_dataloader(
            image_paths,
            labels=labels,
            model_name="dinov2",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        all_features = []
        for batch in tqdm(dataloader, desc="DINOv2 extraction"):
            images = batch[0].to(self.device)
            features = self.model(images)  # CLS token output: (B, 768)
            all_features.append(features.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        logger.info(f"DINOv2 features extracted: {features.shape}")
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
        feature_path = os.path.join(output_dir, "dinov2_features.npy")

        if os.path.exists(feature_path) and not force:
            logger.info(f"Loading existing DINOv2 features from {feature_path}")
            return np.load(feature_path)

        features = self.extract_features(image_paths, labels, **kwargs)

        os.makedirs(output_dir, exist_ok=True)
        np.save(feature_path, features)
        logger.info(f"DINOv2 features saved to {feature_path}")

        return features
