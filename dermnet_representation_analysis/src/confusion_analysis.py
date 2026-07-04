"""
Confusion Analysis module.
Analyzes the embedding space geometry: class centroids, intra-class compactness,
inter-class distances, and identifies confused disease pairs.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from collections import Counter

logger = logging.getLogger("dermnet_analysis")


def compute_class_centroids(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple:
    """
    Compute class centroids in the embedding space.

    Args:
        features: Feature array (N, D).
        labels: Integer label array (N,).

    Returns:
        Tuple of (centroids array (K, D), unique_labels array (K,))
    """
    unique_labels = sorted(set(labels))
    centroids = np.zeros((len(unique_labels), features.shape[1]))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroids[i] = features[mask].mean(axis=0)

    return centroids, np.array(unique_labels)


def compute_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance between class centroids.

    Returns:
        Distance matrix (K, K).
    """
    return cosine_distances(centroids)


def compute_intra_class_compactness(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    unique_labels: np.ndarray,
) -> dict:
    """
    Compute intra-class compactness as average cosine distance to centroid.

    Returns:
        Dict mapping label index to average intra-class distance.
    """
    compactness = {}
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_features = features[mask]
        centroid = centroids[i].reshape(1, -1)
        distances = cosine_distances(class_features, centroid).flatten()
        compactness[label] = {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "n_samples": int(mask.sum()),
        }
    return compactness


def analyze_representation(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    model_name: str,
) -> tuple:
    """
    Full representation geometry analysis for one model.

    Args:
        features: Feature array (N, D).
        labels: Integer label array (N,).
        label_names: List of class name strings.
        model_name: Model name string.

    Returns:
        Tuple of (analysis_df, distance_matrix, centroids)
    """
    logger.info(f"Analyzing representation geometry for {model_name}...")

    # Compute centroids
    centroids, unique_labels = compute_class_centroids(features, labels)

    # Pairwise centroid distances
    dist_matrix = compute_centroid_distances(centroids)

    # Intra-class compactness
    compactness = compute_intra_class_compactness(
        features, labels, centroids, unique_labels
    )

    # Build analysis table
    records = []
    for i, label_idx in enumerate(unique_labels):
        # Find nearest neighbor class (excluding self)
        distances = dist_matrix[i].copy()
        distances[i] = np.inf  # Exclude self
        nn_idx = np.argmin(distances)
        nn_label = unique_labels[nn_idx]

        name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        nn_name = label_names[nn_label] if nn_label < len(label_names) else str(nn_label)

        records.append({
            "representation": model_name,
            "disease_class": name,
            "n_samples": compactness[label_idx]["n_samples"],
            "intra_class_distance": round(compactness[label_idx]["mean_distance"], 4),
            "intra_class_std": round(compactness[label_idx]["std_distance"], 4),
            "nearest_neighbor_class": nn_name,
            "centroid_distance_to_nearest": round(distances[nn_idx], 4),
        })

    analysis_df = pd.DataFrame(records)
    return analysis_df, dist_matrix, centroids


def find_confused_pairs(
    dist_matrix: np.ndarray,
    label_names: list,
    unique_labels: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    model_name: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identify the most confused disease pairs based on:
    1. Centroid proximity in embedding space
    2. Cluster overlap (same cluster assignment)

    Args:
        dist_matrix: Pairwise centroid distance matrix (K, K).
        label_names: List of class names.
        unique_labels: Array of unique label indices.
        cluster_labels: Predicted cluster labels (N,).
        true_labels: Ground-truth labels (N,).
        model_name: Model name.
        top_n: Number of top confused pairs to return.

    Returns:
        DataFrame of confused disease pairs.
    """
    K = len(unique_labels)
    pairs = []

    # Compute cluster overlap for each pair of classes
    for i in range(K):
        for j in range(i + 1, K):
            label_i = unique_labels[i]
            label_j = unique_labels[j]

            name_i = label_names[label_i] if label_i < len(label_names) else str(label_i)
            name_j = label_names[label_j] if label_j < len(label_names) else str(label_j)

            centroid_dist = dist_matrix[i, j]

            # Compute cluster overlap: fraction of images from these two classes
            # that end up in the same cluster
            mask_i = true_labels == label_i
            mask_j = true_labels == label_j
            clusters_i = Counter(cluster_labels[mask_i])
            clusters_j = Counter(cluster_labels[mask_j])

            shared_clusters = set(clusters_i.keys()) & set(clusters_j.keys())
            overlap_count = sum(
                min(clusters_i[c], clusters_j[c]) for c in shared_clusters
            )
            total = mask_i.sum() + mask_j.sum()
            overlap_ratio = overlap_count / total if total > 0 else 0

            pairs.append({
                "representation": model_name,
                "class_a": name_i,
                "class_b": name_j,
                "centroid_distance": round(centroid_dist, 4),
                "cluster_overlap_ratio": round(overlap_ratio, 4),
                "confusion_score": round(
                    overlap_ratio + (1.0 - centroid_dist), 4
                ),  # Higher = more confused
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values("confusion_score", ascending=False)

    return pairs_df.head(top_n).reset_index(drop=True)


def run_confusion_analysis(
    features_dict: dict,
    labels: np.ndarray,
    label_names: list,
    cluster_results_dict: dict,
    config: dict,
    project_root: str,
) -> tuple:
    """
    Run the full confusion analysis for all representations.

    Args:
        features_dict: {model_name: raw_features_array}
        labels: Ground-truth integer labels.
        label_names: List of class name strings.
        cluster_results_dict: {model_name: {method: cluster_labels}}
        config: Config dict.
        project_root: Project root path.

    Returns:
        Tuple of (representation_analysis_df, confused_pairs_df, distance_matrices)
    """
    tables_dir = os.path.join(project_root, config["outputs"]["tables_dir"])
    os.makedirs(tables_dir, exist_ok=True)

    all_repr_dfs = []
    all_confused_dfs = []
    distance_matrices = {}

    top_n = config.get("analysis", {}).get("confusion", {}).get("top_n_confused_pairs", 10)

    for model_name, features in features_dict.items():
        # Representation analysis
        repr_df, dist_matrix, centroids = analyze_representation(
            features, labels, label_names, model_name
        )
        all_repr_dfs.append(repr_df)
        distance_matrices[model_name] = dist_matrix

        # Confused pairs (use first available clustering method)
        if model_name in cluster_results_dict:
            # Prefer kmeans for confusion analysis
            methods = cluster_results_dict[model_name]
            method_name = "kmeans" if "kmeans" in methods else list(methods.keys())[0]
            cluster_labels = methods[method_name]

            unique_labels = sorted(set(labels))
            confused_df = find_confused_pairs(
                dist_matrix,
                label_names,
                np.array(unique_labels),
                cluster_labels,
                labels,
                model_name,
                top_n=top_n,
            )
            all_confused_dfs.append(confused_df)

    # Combine and save
    repr_combined = pd.concat(all_repr_dfs, ignore_index=True)
    repr_path = os.path.join(tables_dir, "representation_analysis.csv")
    repr_combined.to_csv(repr_path, index=False)
    logger.info(f"Representation analysis saved to {repr_path}")

    if all_confused_dfs:
        confused_combined = pd.concat(all_confused_dfs, ignore_index=True)
        confused_path = os.path.join(tables_dir, "confused_disease_pairs.csv")
        confused_combined.to_csv(confused_path, index=False)
        logger.info(f"Confused pairs saved to {confused_path}")
    else:
        confused_combined = pd.DataFrame()

    return repr_combined, confused_combined, distance_matrices
