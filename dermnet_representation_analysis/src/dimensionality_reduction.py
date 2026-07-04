"""
Dimensionality Reduction module.
Applies PCA, UMAP, and optionally t-SNE to feature representations.
"""

import os
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("dermnet_analysis")


def apply_pca(features: np.ndarray, n_components: int = 50) -> tuple:
    """
    Apply PCA for dimensionality reduction.

    Args:
        features: Feature array (N, D).
        n_components: Number of principal components.

    Returns:
        Tuple of (reduced_features, pca_model, explained_variance_ratio)
    """
    n_components = min(n_components, features.shape[1], features.shape[0])
    logger.info(f"Applying PCA: {features.shape[1]}D → {n_components}D")

    # Standardize features before PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features_scaled)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        f"PCA explained variance: {cumulative_var[-1]:.4f} "
        f"({n_components} components)"
    )

    return reduced, pca, pca.explained_variance_ratio_


def apply_umap(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply UMAP for 2D visualization.

    Args:
        features: Feature array (N, D). Typically PCA-reduced.
        n_components: UMAP output dimensions (usually 2).
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.
        metric: Distance metric.
        random_state: Random seed.

    Returns:
        UMAP-reduced array (N, n_components).
    """
    import umap

    logger.info(
        f"Applying UMAP: {features.shape[1]}D → {n_components}D "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})"
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )

    reduced = reducer.fit_transform(features)
    logger.info(f"UMAP complete: {reduced.shape}")
    return reduced


def apply_tsne(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply t-SNE for 2D visualization.

    Args:
        features: Feature array (N, D). Should be PCA-reduced first.
        n_components: Output dimensions (usually 2).
        perplexity: t-SNE perplexity.
        random_state: Random seed.

    Returns:
        t-SNE-reduced array (N, n_components).
    """
    from sklearn.manifold import TSNE

    logger.info(
        f"Applying t-SNE: {features.shape[1]}D → {n_components}D "
        f"(perplexity={perplexity})"
    )

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=0,
    )

    reduced = tsne.fit_transform(features)
    logger.info(f"t-SNE complete: {reduced.shape}")
    return reduced


def run_dimensionality_reduction(
    features: np.ndarray,
    model_name: str,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Run the full dimensionality reduction pipeline for one model's features.

    Args:
        features: Raw feature array (N, D).
        model_name: Name of the model (e.g., "dinov2").
        config: Config dict with DR parameters.
        output_dir: Directory to save reduced embeddings.

    Returns:
        Dict with keys: "pca", "umap", "tsne" (optional), "pca_variance"
    """
    dr_config = config["dimensionality_reduction"]
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # PCA
    pca_path = os.path.join(output_dir, f"{model_name}_pca.npy")
    pca_var_path = os.path.join(output_dir, f"{model_name}_pca_variance.npy")

    if os.path.exists(pca_path) and not config["feature_extraction"].get("force_extract", False):
        logger.info(f"Loading existing PCA features for {model_name}")
        results["pca"] = np.load(pca_path)
        results["pca_variance"] = np.load(pca_var_path)
    else:
        pca_features, pca_model, variance = apply_pca(
            features, dr_config["pca_components"]
        )
        np.save(pca_path, pca_features)
        np.save(pca_var_path, variance)
        results["pca"] = pca_features
        results["pca_variance"] = variance

    # UMAP (on PCA-reduced features for speed)
    umap_path = os.path.join(output_dir, f"{model_name}_umap.npy")

    if os.path.exists(umap_path) and not config["feature_extraction"].get("force_extract", False):
        logger.info(f"Loading existing UMAP embeddings for {model_name}")
        results["umap"] = np.load(umap_path)
    else:
        umap_features = apply_umap(
            results["pca"],
            n_components=dr_config["umap_components"],
            n_neighbors=dr_config["umap_n_neighbors"],
            min_dist=dr_config["umap_min_dist"],
            metric=dr_config["umap_metric"],
            random_state=config.get("random_seed", 42),
        )
        np.save(umap_path, umap_features)
        results["umap"] = umap_features

    # Optional t-SNE
    if dr_config.get("run_tsne", False):
        tsne_path = os.path.join(output_dir, f"{model_name}_tsne.npy")

        if os.path.exists(tsne_path) and not config["feature_extraction"].get("force_extract", False):
            logger.info(f"Loading existing t-SNE embeddings for {model_name}")
            results["tsne"] = np.load(tsne_path)
        else:
            tsne_features = apply_tsne(
                results["pca"],
                perplexity=dr_config.get("tsne_perplexity", 30),
                random_state=config.get("random_seed", 42),
            )
            np.save(tsne_path, tsne_features)
            results["tsne"] = tsne_features

    return results
