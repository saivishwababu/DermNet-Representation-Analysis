"""
Data Loader for DermNet dataset.
Scans the dataset directory, extracts labels from folder names,
validates images, and produces a structured DataFrame index.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("dermnet_analysis")


def scan_dataset(root_path: str, image_extensions: list = None) -> pd.DataFrame:
    """
    Scan the DermNet dataset directory and build an image index DataFrame.

    Supports two structures:
        1. root/train/<class>/<image>  +  root/test/<class>/<image>
        2. root/<class>/<image>

    Args:
        root_path: Absolute or relative path to the dataset root.
        image_extensions: List of valid image extensions (e.g., [".jpg", ".png"]).

    Returns:
        DataFrame with columns: image_path, label, split, filename
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # Normalize extensions to lowercase
    image_extensions = [ext.lower() for ext in image_extensions]

    records = []
    root = Path(root_path).resolve()

    # Detect structure: check for train/ and test/ subdirs
    train_dir = root / "train"
    test_dir = root / "test"
    has_splits = train_dir.is_dir()

    if has_splits:
        # Structure 1: train/test split
        for split_name, split_dir in [("train", train_dir), ("test", test_dir)]:
            if not split_dir.is_dir():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        records.append({
                            "image_path": str(img_file),
                            "label": label,
                            "split": split_name,
                            "filename": img_file.name,
                        })
    else:
        # Structure 2: flat class folders
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    records.append({
                        "image_path": str(img_file),
                        "label": label,
                        "split": "all",
                        "filename": img_file.name,
                    })

    df = pd.DataFrame(records)
    logger.info(f"Found {len(df)} images across {df['label'].nunique()} classes")
    return df


def validate_images(df: pd.DataFrame, remove_corrupt: bool = True) -> pd.DataFrame:
    """
    Validate that all image files can be opened by PIL.
    Optionally removes corrupt entries from the DataFrame.

    Args:
        df: DataFrame with 'image_path' column.
        remove_corrupt: If True, drop rows with corrupt images.

    Returns:
        Cleaned DataFrame.
    """
    corrupt_indices = []
    logger.info("Validating images for corruption...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        try:
            img = Image.open(row["image_path"])
            img.verify()  # Verify the image integrity
        except Exception as e:
            logger.warning(f"Corrupt image: {row['image_path']} — {e}")
            corrupt_indices.append(idx)

    if corrupt_indices:
        logger.warning(f"Found {len(corrupt_indices)} corrupt images")
        if remove_corrupt:
            df = df.drop(corrupt_indices).reset_index(drop=True)
            logger.info(f"Removed corrupt images. Remaining: {len(df)}")
    else:
        logger.info("All images validated successfully")

    return df


def print_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Print and return dataset statistics.

    Returns:
        Dictionary with dataset statistics for report generation.
    """
    stats = {
        "total_images": len(df),
        "num_classes": df["label"].nunique(),
    }

    # Per-split counts
    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        stats["train_images"] = split_counts.get("train", 0)
        stats["test_images"] = split_counts.get("test", 0)

    # Class distribution
    class_counts = df["label"].value_counts()
    stats["class_distribution"] = {
        "largest": f"{class_counts.index[0]} ({class_counts.iloc[0]})",
        "smallest": f"{class_counts.index[-1]} ({class_counts.iloc[-1]})",
    }

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Total images:   {stats['total_images']}")
    logger.info(f"  Num classes:    {stats['num_classes']}")
    if "train_images" in stats:
        logger.info(f"  Train images:   {stats['train_images']}")
        logger.info(f"  Test images:    {stats['test_images']}")
    logger.info(f"  Largest class:  {stats['class_distribution']['largest']}")
    logger.info(f"  Smallest class: {stats['class_distribution']['smallest']}")
    logger.info("-" * 60)
    logger.info("Per-class distribution:")
    for cls_name, count in class_counts.items():
        logger.info(f"    {cls_name:60s} {count:>5d}")
    logger.info("=" * 60)

    return stats


def save_dataset_index(df: pd.DataFrame, output_path: str):
    """Save the dataset index DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset index saved to {output_path}")


def load_dataset(config: dict, project_root: str) -> tuple:
    """
    Full dataset loading pipeline: scan → validate → stats → save.

    Args:
        config: Loaded YAML config dict.
        project_root: Project root directory.

    Returns:
        Tuple of (DataFrame, stats_dict)
    """
    dataset_root = os.path.join(project_root, config["dataset"]["root_path"])
    extensions = config["dataset"].get("image_extensions")

    # Scan
    df = scan_dataset(dataset_root, extensions)

    # Validate
    df = validate_images(df, remove_corrupt=True)

    # Optional: subsample
    max_per_class = config["dataset"].get("max_samples_per_class")
    if max_per_class is not None:
        logger.info(f"Subsampling to max {max_per_class} images per class")
        df = df.groupby("label").apply(
            lambda x: x.sample(n=min(len(x), max_per_class), random_state=42)
        ).reset_index(drop=True)

    # Stats
    stats = print_dataset_stats(df)

    # Save index
    tables_dir = os.path.join(project_root, config["outputs"]["tables_dir"])
    save_dataset_index(df, os.path.join(tables_dir, "dataset_index.csv"))

    return df, stats
