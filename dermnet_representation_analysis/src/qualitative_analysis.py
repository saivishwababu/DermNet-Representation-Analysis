"""
Qualitative Analysis module.
Nearest-neighbor retrieval, image grids, and visual explanations
for embedding space structure.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter

logger = logging.getLogger("dermnet_analysis")


def nearest_neighbor_retrieval(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    image_paths: list,
    model_name: str,
    top_k: int = 5,
    num_query_samples: int = 10,
    random_state: int = 42,
) -> tuple:
    """
    For each class, sample query images and retrieve top-k nearest neighbors.

    Args:
        features: Feature array (N, D).
        labels: Integer label array (N,).
        label_names: List of class name strings.
        image_paths: List of image file paths.
        model_name: Model name.
        top_k: Number of nearest neighbors to retrieve.
        num_query_samples: Number of query samples per class.
        random_state: Random seed.

    Returns:
        Tuple of (results_df, nn_details_list)
        - results_df: Summary with top-1 and top-5 accuracy per class
        - nn_details_list: List of dicts with query/neighbor details for plotting
    """
    rng = np.random.RandomState(random_state)
    unique_labels = sorted(set(labels))

    class_results = []
    nn_details = []

    for label_idx in unique_labels:
        mask = labels == label_idx
        class_indices = np.where(mask)[0]
        class_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)

        # Sample query indices
        n_samples = min(num_query_samples, len(class_indices))
        query_indices = rng.choice(class_indices, n_samples, replace=False)

        top1_correct = 0
        topk_correct = 0

        for q_idx in query_indices:
            q_feature = features[q_idx].reshape(1, -1)

            # Compute distances to all other images
            distances = cosine_distances(q_feature, features).flatten()
            distances[q_idx] = np.inf  # Exclude self

            # Get top-k nearest neighbors
            nn_indices = np.argsort(distances)[:top_k]
            nn_labels = labels[nn_indices]
            nn_paths = [image_paths[i] for i in nn_indices]
            nn_label_names = [
                label_names[l] if l < len(label_names) else str(l)
                for l in nn_labels
            ]

            # Check accuracy
            if nn_labels[0] == label_idx:
                top1_correct += 1
            if label_idx in nn_labels:
                topk_correct += 1

            nn_details.append({
                "model": model_name,
                "query_path": image_paths[q_idx],
                "query_label": class_name,
                "query_label_idx": label_idx,
                "nn_paths": nn_paths,
                "nn_labels": nn_label_names,
                "nn_distances": distances[nn_indices].tolist(),
                "top1_match": nn_labels[0] == label_idx,
            })

        class_results.append({
            "representation": model_name,
            "disease_class": class_name,
            "n_queries": n_samples,
            "top1_accuracy": round(top1_correct / n_samples, 4),
            f"top{top_k}_accuracy": round(topk_correct / n_samples, 4),
        })

    results_df = pd.DataFrame(class_results)

    # Overall summary
    overall_top1 = results_df["top1_accuracy"].mean()
    overall_topk = results_df[f"top{top_k}_accuracy"].mean()
    logger.info(
        f"{model_name} NN retrieval: "
        f"mean top-1 acc = {overall_top1:.4f}, "
        f"mean top-{top_k} acc = {overall_topk:.4f}"
    )

    return results_df, nn_details


def run_qualitative_analysis(
    features_dict: dict,
    labels: np.ndarray,
    label_names: list,
    image_paths: list,
    confused_pairs_df: pd.DataFrame,
    config: dict,
    project_root: str,
) -> pd.DataFrame:
    """
    Run the full qualitative analysis pipeline.

    Args:
        features_dict: {model_name: raw_features_array}
        labels: Ground-truth integer labels.
        label_names: List of class name strings.
        image_paths: List of image paths.
        confused_pairs_df: DataFrame of confused pairs.
        config: Config dict.
        project_root: Project root path.

    Returns:
        Combined nearest-neighbor results DataFrame.
    """
    from .visualization import plot_nn_grid, plot_confused_pair_examples

    tables_dir = os.path.join(project_root, config["outputs"]["tables_dir"])
    nn_grids_dir = os.path.join(project_root, config["outputs"]["nn_grids_dir"])
    confused_dir = os.path.join(project_root, config["outputs"]["confused_pairs_dir"])
    os.makedirs(nn_grids_dir, exist_ok=True)
    os.makedirs(confused_dir, exist_ok=True)

    analysis_config = config.get("analysis", {}).get("nearest_neighbors", {})
    top_k = analysis_config.get("top_k", 5)
    num_queries = analysis_config.get("num_query_samples", 10)
    n_example_imgs = config.get("analysis", {}).get("confusion", {}).get("example_images_per_pair", 5)

    all_nn_dfs = []

    for model_name, features in features_dict.items():
        logger.info(f"Running qualitative analysis for {model_name}...")

        # Nearest-neighbor retrieval
        nn_df, nn_details = nearest_neighbor_retrieval(
            features, labels, label_names, image_paths,
            model_name, top_k=top_k, num_query_samples=num_queries,
        )
        all_nn_dfs.append(nn_df)

        # Plot NN grids for a few examples per model
        model_nn_dir = os.path.join(nn_grids_dir, model_name)
        os.makedirs(model_nn_dir, exist_ok=True)

        # Select up to 3 examples per class (limit total plots)
        plotted = 0
        max_plots = min(30, len(nn_details))
        for detail in nn_details[:max_plots]:
            plot_nn_grid(
                query_path=detail["query_path"],
                neighbor_paths=detail["nn_paths"],
                neighbor_labels=detail["nn_labels"],
                query_label=detail["query_label"],
                output_path=os.path.join(
                    model_nn_dir,
                    f"nn_{detail['query_label'][:20]}_{plotted}.png"
                ),
            )
            plotted += 1

    # Combine NN results
    nn_combined = pd.concat(all_nn_dfs, ignore_index=True)
    nn_path = os.path.join(tables_dir, "nearest_neighbor_results.csv")
    nn_combined.to_csv(nn_path, index=False)
    logger.info(f"NN results saved to {nn_path}")

    # Plot confused pair examples
    if not confused_pairs_df.empty:
        # Get paths indexed by label name
        label_to_paths = {}
        for i, path in enumerate(image_paths):
            name = label_names[labels[i]] if labels[i] < len(label_names) else str(labels[i])
            if name not in label_to_paths:
                label_to_paths[name] = []
            label_to_paths[name].append(path)

        for idx, row in confused_pairs_df.iterrows():
            if idx >= 10:  # Limit to top 10 pairs
                break
            class_a = row["class_a"]
            class_b = row["class_b"]
            model = row["representation"]

            paths_a = label_to_paths.get(class_a, [])[:n_example_imgs]
            paths_b = label_to_paths.get(class_b, [])[:n_example_imgs]

            if paths_a and paths_b:
                safe_name = f"{model}_{idx:02d}_{class_a[:15]}_{class_b[:15]}"
                safe_name = safe_name.replace(" ", "_").replace("/", "_")
                plot_confused_pair_examples(
                    paths_a, paths_b, class_a, class_b,
                    os.path.join(confused_dir, f"{safe_name}.png"),
                    n_examples=n_example_imgs,
                )

    return nn_combined
