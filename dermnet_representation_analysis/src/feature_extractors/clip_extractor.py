"""
CLIP Feature Extractor.
Extracts image embeddings from pretrained CLIP ViT-B/32 using open_clip.
"""

import os
import logging
import numpy as np
import torch
from tqdm import tqdm

from ..preprocessing import create_dataloader

logger = logging.getLogger("dermnet_analysis")


class CLIPExtractor:
    """Extract visual features using CLIP ViT-B/32."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.model = None
        self.model_name = "clip"

    def load_model(self):
        """Load CLIP model using open_clip."""
        logger.info("Loading CLIP ViT-B/32 model...")
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=self.device,
        )
        self.model.eval()
        logger.info(f"CLIP loaded on {self.device}")

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
        Extract CLIP image features (512-d).

        We use our own preprocessing pipeline (with CLIP normalization)
        to handle the images consistently across all models.

        Args:
            image_paths: List of image file paths.
            labels: Optional list of labels.
            batch_size: Batch size.
            num_workers: DataLoader workers.
            image_size: Input image size.

        Returns:
            Feature array of shape (N, 512).
        """
        if self.model is None:
            self.load_model()

        dataloader = create_dataloader(
            image_paths,
            labels=labels,
            model_name="clip",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        all_features = []
        for batch in tqdm(dataloader, desc="CLIP extraction"):
            images = batch[0].to(self.device)
            features = self.model.encode_image(images)  # (B, 512)
            # L2 normalize CLIP embeddings
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().float().numpy())

        features = np.concatenate(all_features, axis=0)
        logger.info(f"CLIP features extracted: {features.shape}")
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
        feature_path = os.path.join(output_dir, "clip_features.npy")

        if os.path.exists(feature_path) and not force:
            logger.info(f"Loading existing CLIP features from {feature_path}")
            return np.load(feature_path)

        features = self.extract_features(image_paths, labels, **kwargs)

        os.makedirs(output_dir, exist_ok=True)
        np.save(feature_path, features)
        logger.info(f"CLIP features saved to {feature_path}")

        return features
