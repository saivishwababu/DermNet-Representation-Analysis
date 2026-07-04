"""
Clustering module.
Implements K-Means, Agglomerative, and HDBSCAN clustering as
post-hoc evaluation tools for representation analysis.
"""

import logging
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

logger = logging.getLogger("dermnet_analysis")


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run K-Means clustering.

    Args:
        features: Feature array (N, D).
        n_clusters: Number of clusters.
        n_init: Number of initializations.
        max_iter: Maximum iterations.
        random_state: Random seed.

    Returns:
        Cluster labels array (N,).
    """
    logger.info(f"Running K-Means with k={n_clusters}...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(features)
    logger.info(f"K-Means complete. Inertia: {kmeans.inertia_:.2f}")
    return labels


def run_agglomerative(
    features: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> np.ndarray:
    """
    Run Agglomerative (Hierarchical) Clustering.

    Args:
        features: Feature array (N, D).
        n_clusters: Number of clusters.
        linkage: Linkage criterion.

    Returns:
        Cluster labels array (N,).
    """
    logger.info(f"Running Agglomerative Clustering with k={n_clusters}, linkage={linkage}...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(features)
    logger.info("Agglomerative Clustering complete.")
    return labels


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Run HDBSCAN clustering.

    Args:
        features: Feature array (N, D).
        min_cluster_size: Minimum cluster size.
        min_samples: Minimum samples.

    Returns:
        Cluster labels array (N,). Note: -1 indicates noise points.
    """
    try:
        import hdbscan
    except ImportError:
        logger.warning("hdbscan not installed, skipping HDBSCAN clustering")
        return None

    logger.info(
        f"Running HDBSCAN (min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples})..."
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(features)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(
        f"HDBSCAN complete. Found {n_clusters} clusters, "
        f"{n_noise} noise points ({n_noise / len(labels) * 100:.1f}%)"
    )
    return labels


def run_all_clustering(
    features: np.ndarray,
    n_classes: int,
    config: dict,
) -> dict:
    """
    Run all configured clustering algorithms.

    Args:
        features: PCA-reduced feature array (N, D).
        n_classes: Number of ground-truth classes (used as k for K-Means/Agglomerative).
        config: Config dict with clustering parameters.

    Returns:
        Dict mapping algorithm name to cluster labels.
    """
    cluster_config = config["clustering"]
    results = {}

    algorithms = cluster_config.get("algorithms", ["kmeans", "agglomerative"])

    if "kmeans" in algorithms:
        km_config = cluster_config.get("kmeans", {})
        results["kmeans"] = run_kmeans(
            features,
            n_clusters=n_classes,
            n_init=km_config.get("n_init", 10),
            max_iter=km_config.get("max_iter", 300),
            random_state=km_config.get("random_state", 42),
        )

    if "agglomerative" in algorithms:
        agg_config = cluster_config.get("agglomerative", {})
        results["agglomerative"] = run_agglomerative(
            features,
            n_clusters=n_classes,
            linkage=agg_config.get("linkage", "ward"),
        )

    if "hdbscan" in algorithms:
        hdb_config = cluster_config.get("hdbscan", {})
        hdb_labels = run_hdbscan(
            features,
            min_cluster_size=hdb_config.get("min_cluster_size", 15),
            min_samples=hdb_config.get("min_samples", 5),
        )
        if hdb_labels is not None:
            results["hdbscan"] = hdb_labels

    return results
