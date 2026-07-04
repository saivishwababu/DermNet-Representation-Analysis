"""
Evaluation module.
Computes clustering and representation quality metrics.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from collections import Counter

logger = logging.getLogger("dermnet_analysis")


def compute_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Compute cluster purity.

    Purity = (1/N) * sum over clusters of max_class_count_in_cluster.
    """
    contingency = {}
    for true, pred in zip(true_labels, cluster_labels):
        if pred not in contingency:
            contingency[pred] = Counter()
        contingency[pred][true] += 1

    total = len(true_labels)
    purity = sum(max(counts.values()) for counts in contingency.values()) / total
    return purity


def evaluate_clustering(
    features: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    model_name: str,
    method_name: str,
) -> dict:
    """
    Compute all clustering evaluation metrics.

    Args:
        features: Feature array used for clustering (N, D).
        true_labels: Ground-truth integer labels (N,).
        cluster_labels: Predicted cluster labels (N,).
        model_name: Name of the representation model.
        method_name: Name of the clustering algorithm.

    Returns:
        Dict with metric names and values.
    """
    # Filter out noise points (label == -1) for metrics that need it
    valid_mask = cluster_labels >= 0
    if valid_mask.sum() < len(cluster_labels):
        logger.info(
            f"Filtering {(~valid_mask).sum()} noise points for metric computation"
        )
        features_valid = features[valid_mask]
        true_valid = true_labels[valid_mask]
        cluster_valid = cluster_labels[valid_mask]
    else:
        features_valid = features
        true_valid = true_labels
        cluster_valid = cluster_labels

    n_unique_clusters = len(set(cluster_valid))

    metrics = {
        "representation": model_name,
        "clustering_method": method_name,
        "n_clusters": n_unique_clusters,
    }

    # Internal metrics (don't need ground truth)
    if n_unique_clusters > 1:
        try:
            metrics["silhouette"] = round(
                silhouette_score(features_valid, cluster_valid, sample_size=min(5000, len(features_valid))),
                4,
            )
        except Exception as e:
            logger.warning(f"Silhouette score failed: {e}")
            metrics["silhouette"] = np.nan

        try:
            metrics["davies_bouldin"] = round(
                davies_bouldin_score(features_valid, cluster_valid), 4
            )
        except Exception as e:
            logger.warning(f"Davies-Bouldin score failed: {e}")
            metrics["davies_bouldin"] = np.nan

        try:
            metrics["calinski_harabasz"] = round(
                calinski_harabasz_score(features_valid, cluster_valid), 2
            )
        except Exception as e:
            logger.warning(f"Calinski-Harabasz score failed: {e}")
            metrics["calinski_harabasz"] = np.nan
    else:
        metrics["silhouette"] = np.nan
        metrics["davies_bouldin"] = np.nan
        metrics["calinski_harabasz"] = np.nan

    # External metrics (need ground truth)
    metrics["ari"] = round(
        adjusted_rand_score(true_valid, cluster_valid), 4
    )
    metrics["nmi"] = round(
        normalized_mutual_info_score(true_valid, cluster_valid), 4
    )
    metrics["purity"] = round(compute_purity(true_valid, cluster_valid), 4)

    logger.info(
        f"{model_name} + {method_name}: "
        f"Silhouette={metrics['silhouette']}, ARI={metrics['ari']}, "
        f"NMI={metrics['nmi']}, Purity={metrics['purity']}"
    )

    return metrics


def evaluate_all(
    features_dict: dict,
    cluster_results_dict: dict,
    true_labels: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    Evaluate all model × clustering method combinations.

    Args:
        features_dict: {model_name: pca_features_array}
        cluster_results_dict: {model_name: {method_name: cluster_labels}}
        true_labels: Ground-truth integer labels.
        output_path: Path to save results CSV.

    Returns:
        DataFrame with all metrics.
    """
    all_metrics = []

    for model_name, clustering_results in cluster_results_dict.items():
        pca_features = features_dict[model_name]
        for method_name, cluster_labels in clustering_results.items():
            metrics = evaluate_clustering(
                pca_features,
                true_labels,
                cluster_labels,
                model_name,
                method_name,
            )
            all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Reorder columns for readability
    col_order = [
        "representation", "clustering_method", "n_clusters",
        "silhouette", "davies_bouldin", "calinski_harabasz",
        "ari", "nmi", "purity",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Clustering results saved to {output_path}")

    return df
