# DermNet Representation Analysis Project

This repository contains the codebase for the **Dermatological Image Representation Analysis Project**. The project evaluates how pretrained vision foundation models (DINOv2, CLIP, and ResNet50) organize dermatological image structures in their representation spaces without supervised clinical fine-tuning.

---

## 📁 Repository Structure

- **`dermnet_representation_analysis/`**: Contains the Python pipelines for dataset deduplication, feature extraction, post-hoc clustering evaluation, PCA dimensional reduction, nearest-neighbor searches, and qualitative plotting.
- **`web/`**: A Next.js 16 (App Router, Tailwind CSS v4, Framer Motion, Recharts) web application presenting an interactive, publication-quality dashboard of findings, interactive UMAP projections, nearest-neighbor retrieval grids, and disease confusion reviews.

---

## 💾 Dataset Setup

To protect storage boundaries, the raw image dataset is **not** pushed to this remote repository.

To run the Python pipelines locally:
1. Download the DermNet dataset from Kaggle:
   👉 **[Kaggle DermNet Dataset](https://www.kaggle.com/datasets/shubhamgoel27/dermnet/data)**
2. Unpack the dataset and place it at the root of this project in a folder named `Dataset/`:
   ```text
   DermNet-Representation-Analysis/
   ├── Dataset/
   │   ├── train/
   │   │   ├── Acne and Rosacea Photos/
   │   │   └── ...
   │   └── test/
   │       ├── Acne and Rosacea Photos/
   │       └── ...
   ├── dermnet_representation_analysis/
   └── web/
   ```

---

## 🚀 Running the Web Dashboard Locally

1. Navigate to the web folder:
   ```bash
   cd web
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Build the static production bundle:
   ```bash
   npm run build
   ```
   The compiled static website will be written to `web/out/`.
