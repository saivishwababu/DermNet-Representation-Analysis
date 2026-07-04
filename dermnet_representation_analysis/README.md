# Representation Analysis of Dermatological Images Using Self-Supervised Visual Features

## Overview

This project analyzes whether pretrained self-supervised visual representations — particularly **DINOv2** — naturally capture meaningful dermatological disease structures without supervised training. Rather than simply comparing clustering algorithms, this work focuses on understanding the geometry and quality of learned visual representations for dermatological images.

## Research Questions

1. **Do DINOv2 embeddings naturally separate dermatological disease categories?**
2. **Which disease categories are well-separated and which are confused in the embedding space?**
3. **Are the confusions visually meaningful** (e.g., similar lesion color, texture, shape, body location)?
4. **How do DINOv2 representations compare with ResNet50 (ImageNet) and CLIP features?**
5. **Which representation is most suitable for unsupervised dermatological image organization?**

## Dataset

This project uses the **DermNet** dermatology image dataset:
- 23 disease categories
- ~19,500 images with train/test splits
- Classes range from 212 (Urticaria Hives) to 1,405 (Psoriasis) images

## Models Compared

| Model | Architecture | Training Paradigm | Embedding Dim |
|-------|-------------|-------------------|---------------|
| DINOv2 | ViT-B/14 | Self-supervised (LVD-142M) | 768 |
| ResNet50 | CNN | Supervised (ImageNet-1K) | 2048 |
| CLIP | ViT-B/32 | Contrastive (image-text pairs) | 512 |

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Path Configuration

Edit `config.yaml` and set the dataset path:

```yaml
dataset:
  root_path: "../Dataset"   # Relative to project directory
```

The project expects the standard DermNet structure:
```
Dataset/
├── train/
│   ├── Acne and Rosacea Photos/
│   ├── Atopic Dermatitis Photos/
│   └── ...
└── test/
    ├── Acne and Rosacea Photos/
    └── ...
```

## Usage

### Full Pipeline

```bash
python main.py --config config.yaml
```

### Individual Steps

```bash
# Extract features from all three models
python main.py --config config.yaml --extract_features

# Run clustering and evaluation
python main.py --config config.yaml --run_clustering

# Run representation analysis and qualitative analysis
python main.py --config config.yaml --run_analysis

# Generate the final markdown report
python main.py --config config.yaml --generate_report

# Force re-extraction of features
python main.py --config config.yaml --extract_features --force_extract
```

## Output Files

### Tables (`outputs/tables/`)

| File | Description |
|------|-------------|
| `dataset_index.csv` | Full image index with paths, labels, splits |
| `duplicates.csv` | Detected duplicate images |
| `clustering_results.csv` | Metrics for all model × clustering combinations |
| `representation_analysis.csv` | Per-class embedding geometry (compactness, nearest class) |
| `confused_disease_pairs.csv` | Top confused disease pairs by centroid proximity |
| `nearest_neighbor_results.csv` | Per-class top-1 and top-5 NN accuracy |

### Plots (`outputs/plots/`)

| File | Description |
|------|-------------|
| `umap_<model>_by_label.png` | UMAP colored by disease labels |
| `umap_<model>_<method>_clusters.png` | UMAP colored by cluster assignments |
| `pca_explained_variance.png` | PCA variance plots for all models |
| `centroid_heatmap_<model>.png` | Pairwise class centroid distance heatmap |
| `confusion_matrix_<model>_<method>.png` | Confusion matrices |
| `confused_pairs_examples/` | Image grids for confused disease pairs |
| `nearest_neighbor_grids/` | NN retrieval example grids |

### Features (`outputs/features/`)

| File | Description |
|------|-------------|
| `dinov2_features.npy` | DINOv2 embeddings (N × 768) |
| `resnet50_features.npy` | ResNet50 embeddings (N × 2048) |
| `clip_features.npy` | CLIP embeddings (N × 512) |
| `labels.npy` | Integer label array |
| `label_names.npy` | Class name mapping |

### Report (`outputs/reports/`)

| File | Description |
|------|-------------|
| `final_report.md` | Complete analysis report with embedded figures |

## How to Interpret Results

### Clustering Metrics

- **Silhouette Score** (higher is better): Measures how well samples match their own cluster vs. nearest cluster.
- **Davies-Bouldin Index** (lower is better): Ratio of within-cluster to between-cluster distances.
- **Calinski-Harabasz Index** (higher is better): Ratio of between-cluster to within-cluster dispersion.
- **ARI** (higher is better, max 1.0): Agreement between cluster labels and true labels, adjusted for chance.
- **NMI** (higher is better, max 1.0): Normalized mutual information between clusters and labels.
- **Purity** (higher is better, max 1.0): Fraction of samples in each cluster that belong to the majority class.

### Representation Analysis

- **Intra-class distance** (lower is better): How tightly a disease class is clustered in embedding space.
- **Centroid distance to nearest class** (higher is better): Separation from the most similar other class.
- **Confused pairs**: Disease pairs with small centroid distance and high cluster overlap suggest visual similarity in the embedding space.

### Nearest-Neighbor Accuracy

- **Top-1 accuracy**: Whether the single nearest neighbor shares the same disease label.
- **Top-5 accuracy**: Whether any of the 5 nearest neighbors shares the same disease label.

High NN accuracy indicates that the representation preserves disease-specific visual structure.

## Project Structure

```
dermnet_representation_analysis/
├── main.py                          # Pipeline orchestrator
├── config.yaml                      # Configuration
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset scanning & validation
│   ├── preprocessing.py             # Deduplication & transforms
│   ├── feature_extractors/
│   │   ├── __init__.py
│   │   ├── dinov2_extractor.py      # DINOv2 ViT-B/14
│   │   ├── resnet_extractor.py      # ResNet50 ImageNet
│   │   └── clip_extractor.py        # CLIP ViT-B/32
│   ├── dimensionality_reduction.py  # PCA, UMAP, t-SNE
│   ├── clustering.py                # K-Means, Agglomerative, HDBSCAN
│   ├── evaluation.py                # Clustering metrics
│   ├── visualization.py             # Publication-quality plots
│   ├── confusion_analysis.py        # Embedding space geometry
│   ├── qualitative_analysis.py      # NN retrieval & examples
│   └── utils.py                     # Config, logging, report gen
├── notebooks/                       # Interactive exploration
└── outputs/                         # All generated outputs
```

## Hardware Notes

- **Mac (CPU/MPS)**: Works out of the box. Feature extraction is slower on CPU (~30–60 min for all three models). MPS (Apple Silicon) provides moderate speedup.
- **GPU (CUDA)**: Significantly faster. Set `device: "auto"` in config.yaml to auto-detect.
- **Memory**: DINOv2 ViT-B/14 requires ~1.5 GB. All models fit comfortably on an 8 GB GPU.

## License

This project is for academic/research purposes only.
