import os
import json
import numpy as np
import pandas as pd

# Paths
project_root = "/Users/saivishwa/Projects/DermNet-main"
features_dir = os.path.join(project_root, "dermnet_representation_analysis/outputs/features")
output_json_path = os.path.join(project_root, "web/public/data/umap_points.json")

# Make sure directory exists
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

# Load data
print("Loading features and UMAP coordinates...")
labels = np.load(os.path.join(features_dir, "labels.npy"))
label_names = np.load(os.path.join(features_dir, "label_names.npy"), allow_pickle=True)

dinov2_umap = np.load(os.path.join(features_dir, "dinov2_umap.npy"))
clip_umap = np.load(os.path.join(features_dir, "clip_umap.npy"))
resnet50_umap = np.load(os.path.join(features_dir, "resnet50_umap.npy"))

image_paths_df = pd.read_csv(os.path.join(features_dir, "image_paths.csv"))
image_paths = image_paths_df['image_path'].tolist()

# Organize by label
label_to_indices = {}
for idx, label in enumerate(labels):
    if label not in label_to_indices:
        label_to_indices[label] = []
    label_to_indices[label].append(idx)

# Sample 50 points per class
sampled_points = []
rng = np.random.RandomState(42)

for label_idx in sorted(label_to_indices.keys()):
    indices = label_to_indices[label_idx]
    n_samples = min(50, len(indices))
    sampled_indices = rng.choice(indices, n_samples, replace=False)
    
    class_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
    
    for idx in sampled_indices:
        full_path = image_paths[idx]
        filename = os.path.basename(full_path)
        
        # Copy the image file to web/public/dataset
        class_folder = os.path.basename(os.path.dirname(full_path))
        dest_dir = os.path.join(project_root, "web/public/dataset", class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        
        import shutil
        if not os.path.exists(dest_path):
            shutil.copy2(full_path, dest_path)
            
        rel_path = f"/dataset/{class_folder}/{filename}"
        
        sampled_points.append({
            "id": int(idx),
            "label_idx": int(label_idx),
            "class_name": class_name,
            "filename": filename,
            "image_path": rel_path,
            "dinov2": [float(dinov2_umap[idx][0]), float(dinov2_umap[idx][1])],
            "clip": [float(clip_umap[idx][0]), float(clip_umap[idx][1])],
            "resnet50": [float(resnet50_umap[idx][0]), float(resnet50_umap[idx][1])],
        })

print(f"Sampled {len(sampled_points)} points. Saving to {output_json_path}...")
with open(output_json_path, "w") as f:
    json.dump(sampled_points, f, indent=2)

print("Done!")
