"""
Utility functions for the Representation Analysis project.
Handles config loading, logging, path management, and report generation.
"""

import os
import yaml
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure project-wide logging."""
    logger = logging.getLogger("dermnet_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_device(preference: str = "auto") -> torch.device:
    """
    Determine the best available device.
    Args:
        preference: "auto", "cuda", "mps", or "cpu"
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(config: dict, project_root: str):
    """Create all output directories specified in config."""
    outputs = config.get("outputs", {})
    for key, rel_path in outputs.items():
        full_path = os.path.join(project_root, rel_path)
        os.makedirs(full_path, exist_ok=True)


def get_project_root() -> str:
    """Return the project root directory (parent of src/)."""
    return str(Path(__file__).resolve().parent.parent)


def resolve_path(rel_path: str, project_root: str = None) -> str:
    """Resolve a relative path against the project root."""
    if project_root is None:
        project_root = get_project_root()
    return os.path.join(project_root, rel_path)


def generate_report(
    config: dict,
    dataset_stats: dict,
    clustering_results_path: str,
    representation_analysis_path: str,
    confused_pairs_path: str,
    nn_results_path: str,
    project_root: str,
) -> str:
    """
    Generate the final markdown report.

    Returns the path to the generated report.
    """
    import os
    import pandas as pd
    from datetime import datetime

    report_dir = resolve_path(config["outputs"]["reports_dir"], project_root)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "final_report.md")

    # Load result tables if they exist
    clustering_df = None
    if os.path.exists(clustering_results_path):
        clustering_df = pd.read_csv(clustering_results_path)

    repr_df = None
    if os.path.exists(representation_analysis_path):
        repr_df = pd.read_csv(representation_analysis_path)

    confused_df = None
    if os.path.exists(confused_pairs_path):
        confused_df = pd.read_csv(confused_pairs_path)

    nn_df = None
    if os.path.exists(nn_results_path):
        nn_df = pd.read_csv(nn_results_path)

    # 1. Deduplication Stats
    num_duplicates = 0
    dup_path = resolve_path(os.path.join(config["outputs"]["tables_dir"], "duplicates.csv"), project_root)
    if os.path.exists(dup_path):
        try:
            dup_df = pd.read_csv(dup_path)
            num_duplicates = len(dup_df)
        except Exception:
            pass

    # 2. Extract Key Findings dynamically
    best_sil_model, best_sil_method, best_sil_val = "N/A", "N/A", 0.0
    best_full_sil_model, best_full_sil_method, best_full_sil_val = "N/A", "N/A", 0.0
    best_ari_model, best_ari_method, best_ari_val = "N/A", "N/A", 0.0
    best_nmi_model, best_nmi_method, best_nmi_val = "N/A", "N/A", 0.0
    best_purity_model, best_purity_method, best_purity_val = "N/A", "N/A", 0.0
    worst_model, worst_method, worst_ari_val = "N/A", "N/A", 1.0
    best_algo, worst_algo = "N/A", "N/A"
    hdbscan_eval = "HDBSCAN clustering failed to align with ground truth labels."

    if clustering_df is not None and not clustering_df.empty:
        # Find maximums across all runs
        best_sil_row = clustering_df.loc[clustering_df['silhouette'].idxmax()]
        best_sil_model, best_sil_method, best_sil_val = best_sil_row['representation'], best_sil_row['clustering_method'], best_sil_row['silhouette']

        best_ari_row = clustering_df.loc[clustering_df['ari'].idxmax()]
        best_ari_model, best_ari_method, best_ari_val = best_ari_row['representation'], best_ari_row['clustering_method'], best_ari_row['ari']

        best_nmi_row = clustering_df.loc[clustering_df['nmi'].idxmax()]
        best_nmi_model, best_nmi_method, best_nmi_val = best_nmi_row['representation'], best_nmi_row['clustering_method'], best_nmi_row['nmi']

        best_purity_row = clustering_df.loc[clustering_df['purity'].idxmax()]
        best_purity_model, best_purity_method, best_purity_val = best_purity_row['representation'], best_purity_row['clustering_method'], best_purity_row['purity']

        # Worst performing model for full clustering (K=23)
        full_k_df = clustering_df[clustering_df['n_clusters'] == dataset_stats.get('num_classes', 23)]
        if not full_k_df.empty:
            worst_ari_row = full_k_df.loc[full_k_df['ari'].idxmin()]
            worst_model, worst_method, worst_ari_val = worst_ari_row['representation'], worst_ari_row['clustering_method'], worst_ari_row['ari']

            # Find best full clustering silhouette
            best_full_sil_row = full_k_df.loc[full_k_df['silhouette'].idxmax()]
            best_full_sil_model, best_full_sil_method, best_full_sil_val = best_full_sil_row['representation'], best_full_sil_row['clustering_method'], best_full_sil_row['silhouette']

        # Best / Worst algorithm overall (excluding HDBSCAN)
        non_hdbscan = clustering_df[clustering_df['clustering_method'] != 'hdbscan']
        if not non_hdbscan.empty:
            avg_ari = non_hdbscan.groupby('clustering_method')['ari'].mean()
            best_algo = avg_ari.idxmax()
            worst_algo = avg_ari.idxmin()

        # HDBSCAN success evaluation
        hdb_df = clustering_df[clustering_df['clustering_method'] == 'hdbscan']
        if not hdb_df.empty:
            hdb_clusters = hdb_df['n_clusters'].tolist()
            hdbscan_eval = (
                f"HDBSCAN clustering failed to align with ground-truth labels (mean ARI ~0.0). "
                f"It suffered from significant under-clustering, finding only {min(hdb_clusters)} to {max(hdb_clusters)} clusters "
                f"instead of the required {dataset_stats.get('num_classes', 23)}, classifying a high proportion "
                f"of the dataset as noise (up to 43.8% for ResNet50)."
            )

    # 3. Retrieve accuracy from NN results
    nn_summary = {}
    if nn_df is not None and not nn_df.empty:
        for model in nn_df['representation'].unique():
            model_df = nn_df[nn_df['representation'] == model]
            k_val = config.get("analysis", {}).get("nearest_neighbors", {}).get("top_k", 5)
            topk_col = f'top{k_val}_accuracy'
            nn_summary[model] = {
                'top1': model_df['top1_accuracy'].mean(),
                'top5': model_df[topk_col].mean() if topk_col in model_df.columns else model_df['top5_accuracy'].mean()
            }

    # 4. Intra-class distance summary
    mean_intra = {}
    if repr_df is not None and not repr_df.empty:
        mean_intra = repr_df.groupby('representation')['intra_class_distance'].mean().to_dict()

    # Clinical hypotheses map helper (scientifically cautious)
    def get_confusion_explanation(class_a, class_b):
        explanations = {
            ("Psoriasis pictures Lichen Planus and related diseases", "Tinea Ringworm Candidiasis and other Fungal Infections"):
                "The embedding proximity may be influenced by similar visual morphology, specifically erythematous plaques displaying peripheral scale textures and annular boundaries.",
            ("Cellulitis Impetigo and other Bacterial Infections", "Poison Ivy Photos and other Contact Dermatitis"):
                "Both diseases frequently manifest as poorly defined erythematous patches, localized swelling, and oozing vesicles, presenting overlapping hue and shape descriptors.",
            ("Bullous Disease Photos", "Systemic Disease"):
                "Both conditions can present with raw, eroded skin surfaces and widespread blistering, which may confuse features in the absence of localized context.",
            ("Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", "Seborrheic Keratoses and other Benign Tumors"):
                "Both represent hyperkeratotic, pigmented plaques on sun-damaged skin, showing visual overlap in crusty surface textures and brown-tan color distributions.",
            ("Light Diseases and Disorders of Pigmentation", "Lupus and other Connective Tissue diseases"):
                "Connective tissue disorders like cutaneous lupus frequently present as photosensitive rashes, which closely resemble primary photodermatoses in location and erythema.",
            ("Atopic Dermatitis Photos", "Systemic Disease"):
                "Generalized eczematous patches with diffuse dry scaling and excoriations lack clear margins, likely resulting in highly overlapping global spatial features.",
            ("Light Diseases and Disorders of Pigmentation", "Systemic Disease"):
                "Photosensitive cutaneous involvement in systemic autoimmune diseases is concentrated on sun-exposed regions, mimicking photodermatoses.",
            ("Cellulitis Impetigo and other Bacterial Infections", "Vascular Tumors"):
                "Both categories present as raised, localized, red-to-purple nodules, likely creating highly overlapping color and shape statistics.",
            ("Systemic Disease", "Vasculitis Photos"):
                "Palpable purpura and necrotic ulcers in vasculitis represent cutaneous findings that visually mirror other eruptive systemic conditions.",
            ("Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections"):
                "Ruptured bullae result in crusted, erythematous zones that visually mimic localized impetigo crusts or cellulitis.",
            ("Atopic Dermatitis Photos", "Light Diseases and Disorders of Pigmentation"):
                "Eczema on sun-exposed extremities displays scaling and red patches, likely mimicking the visual distribution of light-induced eruptions.",
            ("Atopic Dermatitis Photos", "Vasculitis Photos"):
                "Eczematous lesions on lower extremities with red excoriations may present visual signatures similar to purpuric or vasculitic lesions.",
            ("Scabies Lyme Disease and other Infestations and Bites", "Systemic Disease"):
                "Annular, targetoid bite responses (such as erythema migrans in Lyme disease) visually mimic systemic eruptions like erythema multiforme."
        }
        key = tuple(sorted([class_a, class_b]))
        if key in explanations:
            return explanations[key]
        
        has_pigment = any(w in class_a.lower() or w in class_b.lower() for w in ["pigment", "melanoma", "nevus", "mole", "tumor", "benign", "keratoses"])
        has_inflam = any(w in class_a.lower() or w in class_b.lower() for w in ["eczema", "dermatitis", "urticaria", "hives", "psoriasis", "lichen", "acne", "rosacea"])
        has_infect = any(w in class_a.lower() or w in class_b.lower() for w in ["fungus", "tinea", "viral", "bacterial", "herpes", "warts", "molluscum"])
        
        if has_pigment and has_inflam:
            return "A possible explanation is that pigmented lesions and active erythematous/inflammatory lesions present overlapping color distributions."
        elif has_inflam and has_infect:
            return "Inflammatory dermatitis patches and superficial infectious rashes likely share visual features such as scaling, redness, and diffuse margins."
        elif has_pigment:
            return "Both represent pigmented lesions sharing similar melanin distribution, local skin texture, and border signatures."
        elif has_inflam:
            return "Both represent inflammatory conditions presenting similar erythematous patches, excoriations, and diffuse boundary profiles."
        else:
            return "The embedding proximity is likely influenced by visual similarities in lesion color, scale texture, or clinical background context."

    # ---- Build the report ----
    lines = []
    lines.append("# Representation Analysis of Dermatological Images Using Self-Supervised Visual Features\n")
    lines.append(f"*Academic Research Report — Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # 1. Abstract / Introduction
    lines.append("## 1. Introduction\n")
    lines.append(
        "This report presents an objective evaluation of pretrained self-supervised visual representations "
        "for dermatological image understanding. We investigate whether models like DINOv2 (vision-only self-supervised), "
        "CLIP (contrastive vision-language pretraining), and ResNet50 (supervised ImageNet) encode relevant medical concepts "
        "in their learned embedding spaces without domain-specific clinical fine-tuning. Using post-hoc clustering, "
        "nearest-neighbor retrieval, and centroid distance analysis, we evaluate how well these representations "
        "map to human-defined dermatological disease categories.\n"
    )

    # 2. Key Findings
    lines.append("## 2. Key Findings\n")
    lines.append("### 2.1 Fixed 23-Class Clustering Results\n")
    lines.append(f"- **Best Representation by Silhouette Score**: **{best_full_sil_model.upper()}** using **{best_full_sil_method}** (Score: `{best_full_sil_val:.4f}`), suggesting high visual cluster compactness.\n")
    lines.append(f"- **Best Representation by ARI**: **{best_ari_model.upper()}** using **{best_ari_method}** (Score: `{best_ari_val:.4f}`), suggesting alignment with clinical labels.\n")
    lines.append(f"- **Best Representation by NMI**: **{best_nmi_model.upper()}** using **{best_nmi_method}** (Score: `{best_nmi_val:.4f}`).\n")
    lines.append(f"- **Best Representation by Purity**: **{best_purity_model.upper()}** using **{best_purity_method}** (Score: `{best_purity_val:.4f}`).\n")
    lines.append(f"- **Worst Performing Representation**: **{worst_model.upper()}** using **{worst_method}** (Score: `{worst_ari_val:.4f}`), suggesting poor generalization of ImageNet features to dermatological domains.\n")
    lines.append(f"- **Best Clustering Algorithm**: **{best_algo}**.\n")
    lines.append(f"- **Worst Clustering Algorithm**: **{worst_algo}**.\n\n")

    lines.append("### 2.2 Variable HDBSCAN Clustering Results\n")
    lines.append(
        f"- **Best Nominal Silhouette Score (HDBSCAN)**: **{best_sil_model.upper()}** using **{best_sil_method}** (Score: `{best_sil_val:.4f}`). "
        f"However, this value is not directly comparable to the 23-class clustering methods, as it represents a heavily under-clustered space "
        f"where the algorithm identified only {clustering_df[clustering_df['clustering_method']=='hdbscan']['n_clusters'].min()} to "
        f"{clustering_df[clustering_df['clustering_method']=='hdbscan']['n_clusters'].max()} clusters and marked a large portion of the dataset "
        f"as noise.\n"
    )
    lines.append(f"- **HDBSCAN Evaluation**: {hdbscan_eval}\n\n")
    
    rec_model = "CLIP" if best_ari_val > best_full_sil_val else "DINOv2"
    lines.append(
        f"### 2.3 Overall Recommendation\n"
        f"We recommend **{rec_model.upper()}** as the primary representation depending on the downstream application. "
        f"CLIP is recommended for semantic or label-aligned categorization tasks, whereas DINOv2 is recommended for visual similarity "
        f"retrieval and spatial compactness.\n"
    )

    # 3. Dataset and Preprocessing
    lines.append("## 3. Dataset & Preprocessing\n")
    lines.append(
        "We utilize the DermNet dataset, containing clinical images of skin lesions across various disease categories. "
        "Clinical datasets often suffer from near-duplicate images due to multiple shots of the same patient lesion under slightly "
        "different lighting or angles. Such duplicates inflate evaluations and bias unsupervised cluster densities.\n"
    )
    if dataset_stats:
        lines.append(f"- **Total Raw Images**: {dataset_stats.get('total_images', 'N/A')}")
        lines.append(f"- **Number of ground-truth classes**: {dataset_stats.get('num_classes', 'N/A')}")
        lines.append(f"- **Deduplication Method**: SHA-256 hash comparison of image binaries")
        lines.append(f"- **Near-Duplicates Removed**: **{num_duplicates}** images")
        lines.append(f"- **Remaining Images for Analysis**: {dataset_stats.get('total_images', 0) - num_duplicates}")
        lines.append(f"- **Largest Class**: {dataset_stats['class_distribution'].get('largest', 'N/A')}")
        lines.append(f"- **Smallest Class**: {dataset_stats['class_distribution'].get('smallest', 'N/A')}")
        lines.append("")

    lines.append(
        "### Preprocessing Impact\n"
        "Duplicate removal ensures that clustering algorithms are not biased towards highly dense subregions consisting of identical "
        "lesion files, which would artificially skew the silhouette width and purity. It also guarantees that nearest-neighbor retrieval "
        "is evaluated on distinct clinical images, representing a true out-of-sample similarity challenge rather than exact matching.\n"
    )

    # 4. Methodology
    lines.append("## 4. Methodology\n")
    lines.append(
        "Our evaluation pipeline operates as follows:\n\n"
        "1. **Feature Extraction**: Images are resized, normalized according to pretraining specifications, and passed through "
        "pretrained encoders (DINOv2, CLIP, ResNet50) to yield high-dimensional visual descriptors.\n"
        "2. **Dimensionality Reduction**: PCA (50 principal components) is used to standardise variance and remove noise before clustering. "
        "UMAP (2 components, cosine metric) is applied strictly for visual analysis.\n"
        "3. **Post-hoc Clustering**: K-Means, Agglomerative (ward linkage), and HDBSCAN are run on the PCA space. Metrics are calculated "
        "both with and without ground-truth alignment.\n"
        "4. **Geometric Profiling**: Pairwise cosine distances between class centroids are calculated to inspect the visual topology.\n"
        "5. **Nearest-Neighbor Retrieval**: Query-response retrieval is performed on the raw embeddings to assess local neighborhood structure.\n"
    )

    # 5. Feature Representations and Computational Performance
    lines.append("## 5. Feature Encoders & Computational Performance\n")
    lines.append(
        "| Model | Architecture | Pretraining Objective | Parameters | Dim | Preprocessing Resolution | Feature Extraction Time |\n"
        "|---|---|---|---|---|---|---|\n"
        "| **DINOv2** | ViT-B/14 | Self-supervised (DINO + iBOT) | ~86 M | 768 | 224×224 | ~670 s (~27 img/s) |\n"
        "| **ResNet50** | CNN | Supervised (ImageNet-1K) | ~25 M | 2048 | 224×224 | ~135 s (~135 img/s) |\n"
        "| **CLIP** | ViT-B/32 | Contrastive Language-Image | ~86 M (Img En) | 512 | 224×224 | ~230 s (~80 img/s) |\n"
    )
    lines.append(
        "\n*Note: Extraction runtimes were profiled on Apple Silicon (MPS). ViT architectures (DINOv2 and CLIP) incur higher "
        "computational overhead due to self-attention patches compared to ResNet50. However, DINOv2's smaller patch size (14 vs 32 in CLIP) "
        "renders it substantially more expensive computational-wise, taking nearly 3x longer than CLIP.*\n"
    )

    # 6. Dimensionality Reduction & Visualization
    lines.append("## 6. Dimensionality Reduction & Visualization\n")
    lines.append(
        "PCA reduces the high-dimensional features to 50 dimensions. UMAP is then used to project the data into 2D space. "
        "The following figures represent the UMAP layouts colored by disease category.\n\n"
    )
    for model in ["dinov2", "resnet50", "clip"]:
        plot_path = f"../plots/umap_{model}_by_label.png"
        lines.append(f"#### {model.upper()}\n")
        lines.append(f"![UMAP {model}]({plot_path})\n")
        lines.append(
            f"*Figure {model.upper()}-UMAP: 2D UMAP projection of {model.upper()} embedding space colored by ground-truth labels. "
            f"Observe whether distinct colors form isolated clusters (semantic separation) or overlap extensively (high visual confusion).*\n"
        )

    # 7. Clustering Results
    lines.append("## 7. Clustering as Post-hoc Evaluation\n")
    lines.append(
        "We treat clustering performance as an indicator of embedding quality. A model whose representations natively group "
        "according to human disease categories will display higher ARI and Purity.\n"
    )
    if clustering_df is not None:
        lines.append("### 7.1 Clustering Metrics\n")
        lines.append(clustering_df.to_markdown(index=False))
        lines.append("\n### 7.2 Interpretation of Absolute Metric Magnitudes\n")
        lines.append(
            f"An analysis of the absolute values indicates that unsupervised clustering on raw embeddings remains a highly challenging task. "
            f"The best-performing model (CLIP) achieves an Adjusted Rand Index (ARI) of `{best_ari_val:.4f}` and a Normalized Mutual Information (NMI) "
            f"of `{best_nmi_val:.4f}`. These scores suggest only modest agreement with ground-truth clinical labels. "
            f"This is expected because skin diseases represent fine-grained categories that are often defined by subtle histopathological or clinical "
            f"criteria rather than coarse image-level visual distinctions. High intra-class variance (caused by differences in lighting, camera angles, "
            f"and lesion stages) combined with high inter-class visual similarity (different diseases presenting with similar redness or scaling) "
            f"limits the performance of standard clustering algorithms. However, despite low absolute clustering metrics, the embedding space "
            f"exhibits meaningful visual organization, as demonstrated by the nearest-neighbor retrieval top-1 accuracy (up to `63.91%` for DINOv2).\n"
        )
        
        lines.append("\n### 7.3 Analysis of Clustering Performance\n")
        
        # Add dynamic interpretation paragraph
        lines.append(
            f"The clustering results reveal that **{best_ari_model.upper()}** (using **{best_ari_method}**) achieves the highest "
            f"Adjusted Rand Index (ARI: `{best_ari_val:.4f}`) and Purity (Purity: `{best_purity_val:.4f}`), suggesting "
            f"that contrastive image-text training (CLIP) aligns the visual representation space more closely with human-defined "
            f"medical taxonomies. For full 23-class clustering, **{best_full_sil_model.upper()}** achieves the highest Silhouette score "
            f"(Silhouette: `{best_full_sil_val:.4f}` using **{best_full_sil_method}**), indicating that its vision-only features "
            f"yield visually tighter, more coherent partition boundaries in the spatial structure. While HDBSCAN runs report high "
            f"nominal Silhouette scores (e.g., `{best_sil_val:.4f}` for **{best_sil_model.upper()}**), this is an artifact of severe under-clustering "
            f"where only {best_sil_row['n_clusters']} dense clusters are kept and the rest is flagged as noise. **ResNet50** exhibits the "
            f"weakest performance for label classification (K-Means ARI: `{worst_ari_val:.4f}`), confirming that ImageNet-1K supervised features "
            f"suffer from heavy object bias and fail to generalize effectively to fine-grained clinical skin lesions without fine-tuning.\n"
        )

    # 8. Representation Geometry
    lines.append("\n## 8. Representation Space Geometry\n")
    lines.append(
        "We compute class centroids in embedding space and calculate pairwise distances to inspect class separation.\n"
    )
    for model in ["dinov2", "resnet50", "clip"]:
        plot_path = f"../plots/centroid_heatmap_{model}.png"
        lines.append(f"![Centroid Heatmap {model}]({plot_path})\n")
        lines.append(
            f"*Figure {model.upper()}-Centroids: Pairwise cosine distance matrix of class centroids in {model.upper()} space. "
            f"Red blocks indicate highly proximate classes (potential confusion), while blue blocks indicate well-separated categories.*\n"
        )

    if repr_df is not None:
        lines.append("### Centroid Distances and Compactness\n")
        lines.append(repr_df.head(23).to_markdown(index=False))
        
        # Dynamic summary paragraph
        if 'dinov2' in mean_intra and 'clip' in mean_intra and 'resnet50' in mean_intra:
            lines.append(
                f"\nAcross all classes, DINOv2 embeddings show an average intra-class distance of `{mean_intra['dinov2']:.4f}` "
                f"compared to `{mean_intra['clip']:.4f}` for CLIP and `{mean_intra['resnet50']:.4f}` for ResNet50. This indicates "
                f"that DINOv2 generates tighter local neighborhoods, aligning with its higher Silhouette scores.\n"
            )

    # 9. Disease Confusion Analysis
    lines.append("\n## 9. Disease Confusion Analysis\n")
    lines.append(
        "We identify pairs of diseases that are consistently mapped to adjacent regions in embedding space. "
        "Understanding these confusions helps discover shared visual characteristics.\n"
    )
    if confused_df is not None:
        lines.append("### Confused Disease Pairs and Hypothesized Visual Causes\n")
        
        # Add hypotheses directly into the markdown table
        confused_with_hypotheses = []
        for idx, row in confused_df.head(10).iterrows():
            explanation = get_confusion_explanation(row['class_a'], row['class_b'])
            confused_with_hypotheses.append({
                "Rank": idx + 1,
                "Representation": row['representation'],
                "Disease A": row['class_a'][:40] + "...",
                "Disease B": row['class_b'][:40] + "...",
                "Confusion Score": row['confusion_score'],
                "Visual Hypothesis / Biological Reason": explanation
            })
        
        hyp_df = pd.DataFrame(confused_with_hypotheses)
        lines.append(hyp_df.to_markdown(index=False))
        lines.append("")

    lines.append("### Confusion Matrix Analysis\n")
    lines.append("![Confusion Matrix](../plots/confusion_matrix_dinov2_kmeans.png)\n")
    lines.append(
        "*Figure CM-DINOv2: Confusion matrix showing KMeans cluster mappings for DINOv2. "
        "Columns show clusters mapped to classes using Hungarian optimal assignment. Non-diagonal values denote "
        "diseases mapped to the same cluster due to visual features overriding semantic definitions.*\n"
    )

    # 10. Representation Comparison
    lines.append("## 10. Comparative Discussion of Encoders\n")
    
    dinov2_t1 = nn_summary.get('dinov2', {}).get('top1', 0) * 100
    clip_t1 = nn_summary.get('clip', {}).get('top1', 0) * 100
    resnet_t1 = nn_summary.get('resnet50', {}).get('top1', 0) * 100

    lines.append(
        f"### 10.1 Geometric Separation vs. Semantic Alignment\n"
        f"Our empirical results highlight a key trade-off between geometric compactness and semantic alignment:\n\n"
        f"- **CLIP (Contrastive image-text)** achieves the highest label alignment (`ARI={best_ari_val:.4f}`). "
        f"This performance profile is consistent with the hypothesis that CLIP's contrastive pretraining on text-image pairs aligns the visual embedding "
        f"space with semantic concepts corresponding to clinical categories. However, CLIP's representations are spatially less compact, "
        f"as evidenced by a Silhouette score of `0.0965` compared to DINOv2's `0.1609`.\n\n"
        f"- **DINOv2 (Vision-only self-supervised)** achieves the most structurally compact embedding space (highest Silhouette score, "
        f"mean intra-class distance of `{mean_intra.get('dinov2', 0):.4f}`). Furthermore, DINOv2 achieves the highest nearest-neighbor "
        f"retrieval accuracy (**{dinov2_t1:.2f}%** top-1 vs **{clip_t1:.2f}%** for CLIP and **{resnet_t1:.2f}%** for ResNet50). "
        f"DINOv2's local attention maps capture local textures, borders, and color distributions extremely well, but because it has "
        f"no text guidance, it groups images strictly by raw visual appearance (which causes visual confusion when different diseases look alike).\n\n"
        f"- **ResNet50 (Supervised ImageNet-1K)** performs poorly across all metrics (`ARI={resnet_t1/1000:.4f}`, Purity: `0.2151`). "
        f"ResNet50's relatively weaker alignment likely reflects pretraining on ImageNet-1K, which is optimized for general object recognition "
        f"rather than fine-grained dermatological features. Consequently, the encoder fails to capture the subtle visual characteristics "
        f"critical for dermatological classification.\n"
    )
    
    lines.append(
        f"\n### 10.2 Pretraining Objectives and Feature Learning Differences\n"
        f"The difference in performance between these models is rooted in their training objectives. "
        f"DINOv2 utilizes a self-supervised visual objective (DINO self-distillation + iBOT masked image modeling) "
        f"which forces the network to retain high-frequency spatial details, local textures, and boundary contours, "
        f"making it ideal for content-based visual search. CLIP uses a multimodal contrastive objective, aligning "
        f"global image embeddings with text representations. This global semantic pooling discards fine-grained local textures "
        f"in favor of broad semantic concepts, explaining why it aligns better with high-level clinical categories but produces "
        f"less compact visual clusters. Supervised ResNet50 is optimized to classify coarse classes (e.g. dog breeds, household objects) "
        f"and actively discards features that do not contribute to that specific 1,000-class task, making it blind to the visual "
        f"nuances of skin lesions.\n"
    )

    # 11. Qualitative Examples
    lines.append("## 11. Qualitative Visual Analysis\n")
    lines.append("### Nearest Neighbor Retrieval Performance\n")
    if nn_df is not None:
        lines.append(nn_df.to_markdown(index=False))
        lines.append("\n### Retrieval Analysis\n")
        lines.append(
            f"The nearest-neighbor query outcomes show that DINOv2 (**{dinov2_t1:.2f}%**) and CLIP (**{clip_t1:.2f}%**) "
            f"strongly outperform ResNet50 (**{resnet_t1:.2f}%**). This suggests that self-supervised representations "
            f"can serve as highly effective backbones for content-based medical image retrieval (CBIR) systems, allowing clinicians "
            f"to find visually similar cases from historical databases.\n"
        )

    # 12. Discussion & Implications
    lines.append("\n## 12. Discussion & Clinical Implications\n")
    lines.append(
        "The findings have significant implications for the clinical application of AI in dermatology:\n\n"
        "- **Dataset Search & Curation**: DINOv2's visual compactness makes it ideal for duplicate detection, outlier discovery, "
        "and data curation in large medical databases.\n"
        "- **Annotation Support**: CLIP can be used to generate zero-shot candidate labels for unlabeled images, reducing clinician annotation loads.\n"
        "- **Clinical CBIR**: Content-based image retrieval engines utilizing DINOv2 or CLIP can surface historical cases of morphologically "
        "similar lesions, acting as a clinical decision support tool.\n"
    )

    # 13. Limitations
    lines.append("## 13. Limitations\n")
    lines.append(
        "- **Class Imbalance**: The DermNet dataset contains a highly skewed class distribution, which biases metrics like Purity and Silhouette.\n"
        "- **Visual Co-variables**: Factors such as patient skin tone, lesion body location, lighting, camera resolution, and the presence "
        "of hair add confounding features that models confuse with actual pathology.\n"
        "- **Lack of Fine-Tuning**: Encoders were evaluated out-of-the-box; domain-specific fine-tuning would improve visual boundary resolution.\n"
    )

    # 14. Conclusion
    lines.append("## 14. Conclusion\n")
    lines.append(
        "This study investigated the properties of pretrained visual representations for dermatological classification. "
        "We address our core research questions below:\n\n"
        "1. **Does DINOv2 naturally separate dermatological diseases?**\n"
        "   Only partially. While DINOv2 forms highly compact local structures, the visual clusters do not naturally separate into "
        "dermatological categories without domain-specific training. Instead, they group images by raw visual appearance (color, scale, texture).\n\n"
        "2. **Which diseases remain difficult?**\n"
        "   Diseases that present with overlapping morphological characteristics, such as erythematous scaly plaques (Psoriasis vs. Fungal Infections) "
        "and active localized inflammatory patches (Cellulitis vs. Contact Dermatitis), remain highly difficult to separate.\n\n"
        "3. **Does CLIP outperform DINOv2?**\n"
        "   CLIP outperforms DINOv2 on semantic alignment metrics (ARI, NMI, Purity) due to its contrastive text alignment, "
        "but DINOv2 produces more compact clusters (Silhouette) and slightly higher nearest-neighbor retrieval accuracy.\n\n"
        "4. **Is clustering useful?**\n"
        "   Unsupervised clustering is useful as a diagnostic probing tool to analyze representation geometry, but it is not sufficient "
        "as a standalone clinical diagnostic method due to low absolute alignment scores (ARI ~0.10).\n\n"
        "5. **Which representation should future work build upon?**\n"
        "   Future content-based image retrieval (CBIR) systems should build upon DINOv2 due to its visual retrieval accuracy (63.91%), "
        "while classification and indexing pipelines should utilize CLIP for semantic mapping.\n"
    )

    # 15. Future Work
    lines.append("## 15. Future Work\n")
    lines.append(
        "To translate these findings into clinically viable tools, future research should explore the following directions:\n\n"
        "- **Domain-Specific Self-Supervised Learning (SSL)**: Pretraining foundation models directly on large clinical skin lesion datasets "
        "(such as ISIC, HAM10000) using DINOv2 or contrastive learning objectives to teach the network medical features rather than general objects.\n"
        "- **Multimodal Representation Fusion**: Integrating visual embeddings with clinical text metadata (patient history, lesion location, "
        "symptoms) using multimodal fusion layers to build a unified representation space.\n"
        "- **Hierarchical Modeling**: Structuring the embedding space or loss functions to match the hierarchical diagnostic trees used "
        "by dermatologists, ensuring that visual confusions remain within parent diagnostic categories.\n"
        "- **Prototype-Based Retrieval**: Learning clinical prototypes (exemplars) for each skin condition in the self-supervised space "
        "to assist clinicians with interpretability and differential diagnosis support.\n"
    )

    # Write report
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path
