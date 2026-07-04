"""
Visualization module.
Creates publication-quality plots for representation analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("dermnet_analysis")


def _setup_style(config: dict):
    """Apply matplotlib style settings."""
    style = config.get("visualization", {}).get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": config.get("visualization", {}).get("dpi", 150),
    })


def plot_umap_by_label(
    umap_features: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    model_name: str,
    output_dir: str,
    config: dict,
):
    """
    Create UMAP scatter plot colored by ground-truth disease labels.
    """
    _setup_style(config)
    figsize = config.get("visualization", {}).get("figsize_umap", [14, 10])
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)

    # Use a colormap that handles many classes
    if n_classes <= 20:
        cmap = plt.cm.get_cmap("tab20", n_classes)
    else:
        cmap = plt.cm.get_cmap("gist_ncar", n_classes)

    for i, label_idx in enumerate(unique_labels):
        mask = labels == label_idx
        ax.scatter(
            umap_features[mask, 0],
            umap_features[mask, 1],
            c=[cmap(i)],
            label=label_names[label_idx] if label_idx < len(label_names) else str(label_idx),
            s=5,
            alpha=0.6,
        )

    ax.set_title(f"UMAP — {model_name.upper()} Features by Disease Label", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Place legend outside the plot
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        markerscale=3,
        framealpha=0.9,
    )

    plt.tight_layout()
    path = os.path.join(output_dir, f"umap_{model_name}_by_label.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved UMAP label plot: {path}")


def plot_umap_by_cluster(
    umap_features: np.ndarray,
    cluster_labels: np.ndarray,
    model_name: str,
    method_name: str,
    output_dir: str,
    config: dict,
):
    """
    Create UMAP scatter plot colored by cluster assignment.
    """
    _setup_style(config)
    figsize = config.get("visualization", {}).get("figsize_umap", [14, 10])
    fig, ax = plt.subplots(figsize=figsize)

    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        label_text = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
        color = "gray" if cluster_id == -1 else cmap(i)
        ax.scatter(
            umap_features[mask, 0],
            umap_features[mask, 1],
            c=[color],
            label=label_text,
            s=5,
            alpha=0.6,
        )

    ax.set_title(
        f"UMAP — {model_name.upper()} + {method_name} Clusters",
        fontsize=14,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        markerscale=3,
    )

    plt.tight_layout()
    path = os.path.join(output_dir, f"umap_{model_name}_{method_name}_clusters.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved UMAP cluster plot: {path}")


def plot_pca_variance(
    variance_ratios: dict,
    output_dir: str,
    config: dict,
):
    """
    Plot PCA explained variance for all models.

    Args:
        variance_ratios: {model_name: variance_ratio_array}
    """
    _setup_style(config)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"dinov2": "#2196F3", "resnet50": "#FF9800", "clip": "#4CAF50"}

    # Individual variance
    for model_name, variance in variance_ratios.items():
        color = colors.get(model_name, "#666666")
        axes[0].plot(
            range(1, len(variance) + 1),
            variance,
            label=model_name.upper(),
            color=color,
            linewidth=1.5,
        )

    axes[0].set_title("Individual Explained Variance")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative variance
    for model_name, variance in variance_ratios.items():
        color = colors.get(model_name, "#666666")
        cumulative = np.cumsum(variance)
        axes[1].plot(
            range(1, len(variance) + 1),
            cumulative,
            label=model_name.upper(),
            color=color,
            linewidth=1.5,
        )

    axes[1].axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95% variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative Variance Ratio")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "pca_explained_variance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PCA variance plot: {path}")


def plot_centroid_heatmap(
    distance_matrix: np.ndarray,
    label_names: list,
    model_name: str,
    output_dir: str,
    config: dict,
):
    """
    Plot a heatmap of pairwise cosine distances between class centroids.
    """
    _setup_style(config)
    figsize = config.get("visualization", {}).get("figsize_heatmap", [16, 14])
    fig, ax = plt.subplots(figsize=figsize)

    # Shorten label names for readability
    short_names = [name[:35] + "..." if len(name) > 38 else name for name in label_names]

    sns.heatmap(
        distance_matrix,
        xticklabels=short_names,
        yticklabels=short_names,
        cmap="RdYlBu_r",
        annot=False,
        square=True,
        ax=ax,
        cbar_kws={"label": "Cosine Distance"},
    )

    ax.set_title(
        f"Class Centroid Pairwise Cosine Distance — {model_name.upper()}",
        fontsize=14,
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"centroid_heatmap_{model_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved centroid heatmap: {path}")


def plot_confusion_matrix(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    label_names: list,
    model_name: str,
    method_name: str,
    output_dir: str,
    config: dict,
):
    """
    Plot confusion matrix between ground-truth labels and cluster assignments.
    Uses Hungarian algorithm for optimal cluster-to-class mapping.
    """
    _setup_style(config)
    figsize = config.get("visualization", {}).get("figsize_confusion", [18, 16])

    # Build contingency matrix
    unique_true = sorted(set(true_labels))
    unique_pred = sorted(set(cluster_labels[cluster_labels >= 0]))

    cm = confusion_matrix(true_labels, cluster_labels, labels=unique_pred)

    # Hungarian matching for best assignment
    if cm.shape[0] <= cm.shape[1]:
        # More clusters than classes — match classes to clusters
        cost_matrix = -cm  # Negate because linear_sum_assignment minimizes
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Reorder columns
        order = list(col_ind) + [i for i in range(cm.shape[1]) if i not in col_ind]
        cm = cm[:, order]
        mapped_cluster_names = [f"C{order[i]}" for i in range(len(order))]
    else:
        mapped_cluster_names = [f"C{i}" for i in unique_pred]

    fig, ax = plt.subplots(figsize=figsize)

    short_names = [name[:30] + "..." if len(name) > 33 else name for name in label_names]

    sns.heatmap(
        cm,
        xticklabels=mapped_cluster_names[:cm.shape[1]],
        yticklabels=short_names[:cm.shape[0]],
        cmap="Blues",
        annot=True if cm.shape[0] <= 25 else False,
        fmt="d" if cm.shape[0] <= 25 else "",
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix — {model_name.upper()} + {method_name}",
        fontsize=14,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("True Disease Class")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name}_{method_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {path}")


def plot_nn_grid(
    query_path: str,
    neighbor_paths: list,
    neighbor_labels: list,
    query_label: str,
    output_path: str,
):
    """
    Plot a nearest-neighbor image grid for a single query image.
    """
    from PIL import Image

    n_neighbors = len(neighbor_paths)
    fig, axes = plt.subplots(1, n_neighbors + 1, figsize=(3 * (n_neighbors + 1), 3.5))

    # Query image
    try:
        img = Image.open(query_path).convert("RGB").resize((200, 200))
    except Exception:
        img = Image.new("RGB", (200, 200), (128, 128, 128))
    axes[0].imshow(img)
    axes[0].set_title(f"Query\n{query_label[:25]}", fontsize=8, color="blue")
    axes[0].axis("off")

    # Neighbor images
    for i, (path, label) in enumerate(zip(neighbor_paths, neighbor_labels)):
        try:
            img = Image.open(path).convert("RGB").resize((200, 200))
        except Exception:
            img = Image.new("RGB", (200, 200), (128, 128, 128))
        axes[i + 1].imshow(img)
        color = "green" if label == query_label else "red"
        axes[i + 1].set_title(f"NN-{i+1}\n{label[:25]}", fontsize=8, color=color)
        axes[i + 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_confused_pair_examples(
    paths_a: list,
    paths_b: list,
    label_a: str,
    label_b: str,
    output_path: str,
    n_examples: int = 5,
):
    """
    Plot example images from two confused disease classes side by side.
    """
    from PIL import Image

    n_a = min(n_examples, len(paths_a))
    n_b = min(n_examples, len(paths_b))
    n_cols = max(n_a, n_b)

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Confused Pair: {label_a[:40]} vs {label_b[:40]}",
        fontsize=11,
        fontweight="bold",
    )

    for i in range(n_cols):
        # Row 1: Class A
        if i < n_a:
            try:
                img = Image.open(paths_a[i]).convert("RGB").resize((200, 200))
            except Exception:
                img = Image.new("RGB", (200, 200), (128, 128, 128))
            axes[0, i].imshow(img)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel(label_a[:30], fontsize=9, rotation=0, labelpad=100)

        # Row 2: Class B
        if i < n_b:
            try:
                img = Image.open(paths_b[i]).convert("RGB").resize((200, 200))
            except Exception:
                img = Image.new("RGB", (200, 200), (128, 128, 128))
            axes[1, i].imshow(img)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel(label_b[:30], fontsize=9, rotation=0, labelpad=100)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info(f"Saved confused pair examples: {output_path}")
