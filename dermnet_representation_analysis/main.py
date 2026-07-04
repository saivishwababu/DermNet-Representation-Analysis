#!/usr/bin/env python3
"""
Representation Analysis of Dermatological Images
Using Self-Supervised Visual Features

Main pipeline orchestrator.
Usage:
    python main.py --config config.yaml                   # Full pipeline
    python main.py --config config.yaml --extract_features
    python main.py --config config.yaml --run_clustering
    python main.py --config config.yaml --run_analysis
    python main.py --config config.yaml --generate_report
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, setup_logging, get_device, set_seed, ensure_dirs, resolve_path, generate_report
from src.data_loader import load_dataset
from src.preprocessing import remove_duplicates
from src.feature_extractors import DINOv2Extractor, ResNet50Extractor, CLIPExtractor
from src.dimensionality_reduction import run_dimensionality_reduction
from src.clustering import run_all_clustering
from src.evaluation import evaluate_all
from src.confusion_analysis import run_confusion_analysis
from src.qualitative_analysis import run_qualitative_analysis
from src.visualization import (
    plot_umap_by_label,
    plot_umap_by_cluster,
    plot_pca_variance,
    plot_centroid_heatmap,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Representation Analysis of Dermatological Images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="Run feature extraction only",
    )
    parser.add_argument(
        "--run_clustering",
        action="store_true",
        help="Run clustering only (requires extracted features)",
    )
    parser.add_argument(
        "--run_analysis",
        action="store_true",
        help="Run representation and confusion analysis",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate final markdown report",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Force re-extraction of features even if cached",
    )

    return parser.parse_args()


def step_load_data(config, logger):
    """Step 1: Load and prepare the dataset."""
    logger.info("=" * 70)
    logger.info("STEP 1: Loading Dataset")
    logger.info("=" * 70)

    df, stats = load_dataset(config, PROJECT_ROOT)

    # Deduplication
    if config["preprocessing"].get("remove_duplicates", True):
        dup_path = resolve_path(
            os.path.join(config["outputs"]["tables_dir"], "duplicates.csv"),
            PROJECT_ROOT,
        )
        df = remove_duplicates(df, output_path=dup_path)

    return df, stats


def step_extract_features(config, df, logger, force=False):
    """Step 2: Extract features from all models."""
    logger.info("=" * 70)
    logger.info("STEP 2: Feature Extraction")
    logger.info("=" * 70)

    device = get_device(config["feature_extraction"].get("device", "auto"))
    logger.info(f"Using device: {device}")

    image_paths = df["image_path"].tolist()
    labels_str = df["label"].tolist()

    # Encode string labels to integers
    unique_labels = sorted(set(labels_str))
    label_to_idx = {name: i for i, name in enumerate(unique_labels)}
    labels_int = np.array([label_to_idx[l] for l in labels_str])

    features_dir = resolve_path(config["outputs"]["features_dir"], PROJECT_ROOT)
    os.makedirs(features_dir, exist_ok=True)

    batch_size = config["feature_extraction"].get("batch_size", 32)
    num_workers = config["preprocessing"].get("num_workers", 4)

    # Save labels and paths (shared across all models)
    labels_path = os.path.join(features_dir, "labels.npy")
    paths_csv = os.path.join(features_dir, "image_paths.csv")
    label_names_path = os.path.join(features_dir, "label_names.npy")

    np.save(labels_path, labels_int)
    np.save(label_names_path, np.array(unique_labels))
    pd.DataFrame({"image_path": image_paths}).to_csv(paths_csv, index=False)

    features_dict = {}

    # --- DINOv2 ---
    logger.info("-" * 40)
    logger.info("Extracting DINOv2 features...")
    dinov2 = DINOv2Extractor(device=device)
    features_dict["dinov2"] = dinov2.extract_and_save(
        image_paths, labels_str, features_dir, force=force,
        batch_size=batch_size, num_workers=num_workers,
        image_size=config["feature_extraction"]["models"]["dinov2"]["image_size"],
    )
    del dinov2  # Free memory

    # --- ResNet50 ---
    logger.info("-" * 40)
    logger.info("Extracting ResNet50 features...")
    resnet = ResNet50Extractor(device=device)
    features_dict["resnet50"] = resnet.extract_and_save(
        image_paths, labels_str, features_dir, force=force,
        batch_size=batch_size, num_workers=num_workers,
        image_size=config["feature_extraction"]["models"]["resnet50"]["image_size"],
    )
    del resnet

    # --- CLIP ---
    logger.info("-" * 40)
    logger.info("Extracting CLIP features...")
    clip_ext = CLIPExtractor(device=device)
    features_dict["clip"] = clip_ext.extract_and_save(
        image_paths, labels_str, features_dir, force=force,
        batch_size=batch_size, num_workers=num_workers,
        image_size=config["feature_extraction"]["models"]["clip"]["image_size"],
    )
    del clip_ext

    return features_dict, labels_int, unique_labels, image_paths


def step_dimensionality_reduction(config, features_dict, logger):
    """Step 3: Apply dimensionality reduction."""
    logger.info("=" * 70)
    logger.info("STEP 3: Dimensionality Reduction")
    logger.info("=" * 70)

    features_dir = resolve_path(config["outputs"]["features_dir"], PROJECT_ROOT)
    dr_results = {}

    for model_name, features in features_dict.items():
        logger.info(f"--- {model_name.upper()} ---")
        dr_results[model_name] = run_dimensionality_reduction(
            features, model_name, config, features_dir
        )

    return dr_results


def step_clustering(config, dr_results, n_classes, logger):
    """Step 4: Run clustering on PCA-reduced features."""
    logger.info("=" * 70)
    logger.info("STEP 4: Clustering")
    logger.info("=" * 70)

    cluster_results = {}
    for model_name, dr_data in dr_results.items():
        logger.info(f"--- {model_name.upper()} ---")
        cluster_results[model_name] = run_all_clustering(
            dr_data["pca"], n_classes, config
        )

    return cluster_results


def step_evaluation(config, dr_results, cluster_results, labels_int, logger):
    """Step 5: Evaluate clustering results."""
    logger.info("=" * 70)
    logger.info("STEP 5: Evaluation")
    logger.info("=" * 70)

    pca_features = {name: dr["pca"] for name, dr in dr_results.items()}

    output_path = resolve_path(
        os.path.join(config["outputs"]["tables_dir"], "clustering_results.csv"),
        PROJECT_ROOT,
    )

    results_df = evaluate_all(
        pca_features, cluster_results, labels_int, output_path
    )

    logger.info("\n" + results_df.to_string(index=False))
    return results_df


def step_visualization(config, dr_results, cluster_results, labels_int, label_names, logger):
    """Step 6: Create visualizations."""
    logger.info("=" * 70)
    logger.info("STEP 6: Visualization")
    logger.info("=" * 70)

    plots_dir = resolve_path(config["outputs"]["plots_dir"], PROJECT_ROOT)
    os.makedirs(plots_dir, exist_ok=True)

    # PCA variance plot
    variance_ratios = {
        name: dr["pca_variance"] for name, dr in dr_results.items()
    }
    plot_pca_variance(variance_ratios, plots_dir, config)

    for model_name, dr_data in dr_results.items():
        # UMAP by label
        plot_umap_by_label(
            dr_data["umap"], labels_int, label_names,
            model_name, plots_dir, config,
        )

        # UMAP by cluster (for each clustering method)
        if model_name in cluster_results:
            for method_name, cluster_labels in cluster_results[model_name].items():
                plot_umap_by_cluster(
                    dr_data["umap"], cluster_labels,
                    model_name, method_name, plots_dir, config,
                )

                # Confusion matrix
                plot_confusion_matrix(
                    labels_int, cluster_labels, label_names,
                    model_name, method_name, plots_dir, config,
                )


def step_analysis(config, features_dict, labels_int, label_names,
                  cluster_results, image_paths, logger):
    """Step 7: Representation and confusion analysis."""
    logger.info("=" * 70)
    logger.info("STEP 7: Representation & Confusion Analysis")
    logger.info("=" * 70)

    plots_dir = resolve_path(config["outputs"]["plots_dir"], PROJECT_ROOT)

    # Confusion analysis (includes representation geometry)
    repr_df, confused_df, dist_matrices = run_confusion_analysis(
        features_dict, labels_int, label_names,
        cluster_results, config, PROJECT_ROOT,
    )

    # Plot centroid heatmaps
    for model_name, dist_matrix in dist_matrices.items():
        plot_centroid_heatmap(
            dist_matrix, label_names, model_name, plots_dir, config
        )

    # Qualitative analysis (NN retrieval + example grids)
    nn_df = run_qualitative_analysis(
        features_dict, labels_int, label_names, image_paths,
        confused_df, config, PROJECT_ROOT,
    )

    return repr_df, confused_df, nn_df


def step_generate_report(config, dataset_stats, logger):
    """Step 8: Generate final markdown report."""
    logger.info("=" * 70)
    logger.info("STEP 8: Report Generation")
    logger.info("=" * 70)

    tables_dir = resolve_path(config["outputs"]["tables_dir"], PROJECT_ROOT)

    report_path = generate_report(
        config,
        dataset_stats,
        clustering_results_path=os.path.join(tables_dir, "clustering_results.csv"),
        representation_analysis_path=os.path.join(tables_dir, "representation_analysis.csv"),
        confused_pairs_path=os.path.join(tables_dir, "confused_disease_pairs.csv"),
        nn_results_path=os.path.join(tables_dir, "nearest_neighbor_results.csv"),
        project_root=PROJECT_ROOT,
    )

    logger.info(f"Final report generated: {report_path}")


def main():
    args = parse_args()

    # Load config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    # Override force flag
    if args.force_extract:
        config["feature_extraction"]["force_extract"] = True

    # Setup
    logger = setup_logging()
    set_seed(config.get("random_seed", 42))
    ensure_dirs(config, PROJECT_ROOT)

    logger.info("=" * 70)
    logger.info("  Representation Analysis of Dermatological Images")
    logger.info("  Using Self-Supervised Visual Features")
    logger.info("=" * 70)

    # Determine which steps to run
    run_all = not any([
        args.extract_features,
        args.run_clustering,
        args.run_analysis,
        args.generate_report,
    ])

    # --- Step 1: Load Data ---
    df, dataset_stats = step_load_data(config, logger)

    if args.extract_features or run_all:
        # --- Step 2: Extract Features ---
        features_dict, labels_int, label_names, image_paths = step_extract_features(
            config, df, logger, force=config["feature_extraction"].get("force_extract", False)
        )
    else:
        # Load cached features and metadata
        features_dir = resolve_path(config["outputs"]["features_dir"], PROJECT_ROOT)
        labels_int = np.load(os.path.join(features_dir, "labels.npy"))
        label_names = list(np.load(os.path.join(features_dir, "label_names.npy"), allow_pickle=True))
        image_paths = pd.read_csv(os.path.join(features_dir, "image_paths.csv"))["image_path"].tolist()

        features_dict = {}
        for model_name in ["dinov2", "resnet50", "clip"]:
            fpath = os.path.join(features_dir, f"{model_name}_features.npy")
            if os.path.exists(fpath):
                features_dict[model_name] = np.load(fpath)
                logger.info(f"Loaded {model_name} features: {features_dict[model_name].shape}")

    if args.extract_features and not run_all:
        logger.info("Feature extraction complete. Exiting.")
        return

    # --- Step 3: Dimensionality Reduction ---
    if run_all or args.run_clustering or args.run_analysis:
        dr_results = step_dimensionality_reduction(config, features_dict, logger)

    # --- Step 4 & 5: Clustering + Evaluation ---
    if args.run_clustering or run_all:
        n_classes = len(set(labels_int))
        cluster_results = step_clustering(config, dr_results, n_classes, logger)
        results_df = step_evaluation(config, dr_results, cluster_results, labels_int, logger)

    # --- Step 6: Visualization ---
    if run_all or args.run_clustering:
        step_visualization(
            config, dr_results, cluster_results, labels_int, label_names, logger
        )

    # --- Step 7: Analysis ---
    if args.run_analysis or run_all:
        if not run_all and not args.run_clustering:
            # Need to run clustering first for analysis
            n_classes = len(set(labels_int))
            cluster_results = step_clustering(config, dr_results, n_classes, logger)

        repr_df, confused_df, nn_df = step_analysis(
            config, features_dict, labels_int, label_names,
            cluster_results, image_paths, logger,
        )

    # --- Step 8: Report ---
    if args.generate_report or run_all:
        step_generate_report(config, dataset_stats, logger)

    logger.info("=" * 70)
    logger.info("  Pipeline complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
