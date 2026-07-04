import fs from 'fs';
import path from 'path';

// Custom CSV parser that handles quoted strings containing commas
function parseCSV(filePath: string): Record<string, string>[] {
  if (!fs.existsSync(filePath)) {
    console.warn(`File not found: ${filePath}`);
    return [];
  }
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n').map(line => line.trim()).filter(Boolean);
  if (lines.length === 0) return [];

  const headers = parseCSVLine(lines[0]);
  const results: Record<string, string>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    const row: Record<string, string> = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || '';
    });
    results.push(row);
  }
  return results;
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  
  return result.map(v => v.replace(/^"|"$/g, '').trim());
}

// Data Interface definitions
export interface ClusteringResult {
  representation: string;
  clustering_method: string;
  n_clusters: number;
  silhouette: number;
  davies_bouldin: number;
  calinski_harabasz: number;
  ari: number;
  nmi: number;
  purity: number;
}

export interface RepresentationAnalysis {
  representation: string;
  disease_class: string;
  n_samples: number;
  intra_class_distance: number;
  intra_class_std: number;
  nearest_neighbor_class: string;
  centroid_distance_to_nearest: number;
}

export interface ConfusedDiseasePair {
  representation: string;
  class_a: string;
  class_b: string;
  centroid_distance: number;
  cluster_overlap_ratio: number;
  confusion_score: number;
  rank: number;
  image_name: string;
  image_path: string;
}

export interface NearestNeighborResult {
  representation: string;
  disease_class: string;
  n_queries: number;
  top1_accuracy: number;
  top5_accuracy: number;
}

export interface UmapPoint {
  id: number;
  label_idx: number;
  class_name: string;
  filename: string;
  image_path: string;
  dinov2: [number, number];
  clip: [number, number];
  resnet50: [number, number];
}

// Data getters
export function getClusteringResults(): ClusteringResult[] {
  const filePath = path.join(process.cwd(), '../dermnet_representation_analysis/outputs/tables/clustering_results.csv');
  const raw = parseCSV(filePath);
  return raw.map(r => ({
    representation: r.representation,
    clustering_method: r.clustering_method,
    n_clusters: parseInt(r.n_clusters) || 0,
    silhouette: parseFloat(r.silhouette) || 0,
    davies_bouldin: parseFloat(r.davies_bouldin) || 0,
    calinski_harabasz: parseFloat(r.calinski_harabasz) || 0,
    ari: parseFloat(r.ari) || 0,
    nmi: parseFloat(r.nmi) || 0,
    purity: parseFloat(r.purity) || 0,
  }));
}

export function getRepresentationAnalysis(): RepresentationAnalysis[] {
  const filePath = path.join(process.cwd(), '../dermnet_representation_analysis/outputs/tables/representation_analysis.csv');
  const raw = parseCSV(filePath);
  return raw.map(r => ({
    representation: r.representation,
    disease_class: r.disease_class,
    n_samples: parseInt(r.n_samples) || 0,
    intra_class_distance: parseFloat(r.intra_class_distance) || 0,
    intra_class_std: parseFloat(r.intra_class_std) || 0,
    nearest_neighbor_class: r.nearest_neighbor_class,
    centroid_distance_to_nearest: parseFloat(r.centroid_distance_to_nearest) || 0,
  }));
}

export function getConfusedPairs(): ConfusedDiseasePair[] {
  const filePath = path.join(process.cwd(), '../dermnet_representation_analysis/outputs/tables/confused_disease_pairs.csv');
  const raw = parseCSV(filePath);
  return raw.map((r, index) => {
    const rank = index + 1;
    // Map to pre-rendered filename: clip_00_Class_A_Class_B.png
    // e.g. safe_name = f"{model}_{idx:02d}_{class_a[:15]}_{class_b[:15]}"
    const idxStr = String(index).padStart(2, '0');
    const safeA = r.class_a.slice(0, 15).replace(/ /g, '_').replace(/\//g, '_');
    const safeB = r.class_b.slice(0, 15).replace(/ /g, '_').replace(/\//g, '_');
    const imageName = `${r.representation}_${idxStr}_${safeA}_${safeB}.png`;
    
    const imagePath = `/plots/confused_pairs_examples/${imageName}`;

    return {
      representation: r.representation,
      class_a: r.class_a,
      class_b: r.class_b,
      centroid_distance: parseFloat(r.centroid_distance) || 0,
      cluster_overlap_ratio: parseFloat(r.cluster_overlap_ratio) || 0,
      confusion_score: parseFloat(r.confusion_score) || 0,
      rank,
      image_name: imageName,
      image_path: imagePath
    };
  });
}

export function getNearestNeighborResults(): NearestNeighborResult[] {
  const filePath = path.join(process.cwd(), '../dermnet_representation_analysis/outputs/tables/nearest_neighbor_results.csv');
  const raw = parseCSV(filePath);
  return raw.map(r => ({
    representation: r.representation,
    disease_class: r.disease_class,
    n_queries: parseInt(r.n_queries) || 0,
    top1_accuracy: parseFloat(r.top1_accuracy) || 0,
    top5_accuracy: parseFloat(r.top5_accuracy) || 0,
  }));
}

export function getUmapPoints(): UmapPoint[] {
  const filePath = path.join(process.cwd(), 'public/data/umap_points.json');
  if (!fs.existsSync(filePath)) {
    console.warn(`UMAP points JSON not found: ${filePath}`);
    return [];
  }
  const content = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(content);
}

export function getFinalReport(): string {
  const filePath = path.join(process.cwd(), '../dermnet_representation_analysis/outputs/reports/final_report.md');
  if (!fs.existsSync(filePath)) {
    return 'Final report not found.';
  }
  return fs.readFileSync(filePath, 'utf-8');
}

export interface ProjectFile {
  name: string;
  path: string;
  content: string;
  description: string;
}

export function getProjectFiles(): ProjectFile[] {
  const root = path.join(process.cwd(), '../dermnet_representation_analysis');
  const filesToRead = [
    { name: 'main.py', path: 'main.py', desc: 'Main entry point orchestrating the dataset loading, duplicate removal, feature extraction, post-hoc clustering, centroid computations, and qualitative retrievals.' },
    { name: 'config.yaml', path: 'config.yaml', desc: 'Configuration file specifying hyperparameters, clustering methods, UMAP dimensions, PCA components, and folder destinations for all outputs.' },
    { name: 'requirements.txt', path: 'requirements.txt', desc: 'Python dependencies required to execute the pipeline (PyTorch, scikit-learn, numpy, pandas, umap-learn, matplotlib).' },
    { name: 'utils.py', path: 'src/utils.py', desc: 'Helper functions for logging setup, tracking execution times, and handling CUDA/MPS device assignments.' },
    { name: 'qualitative_analysis.py', path: 'src/qualitative_analysis.py', desc: 'Module handling local neighborhood retrieval accuracy (top-1 and top-5 nearest neighbors) and class confusion pairs selection.' },
    { name: 'visualization.py', path: 'src/visualization.py', desc: 'Visualization code plotting UMAP projections, centroid pairwise distance heatmaps, cluster assignment confusion matrices, and nearest-neighbor grids.' }
  ];

  return filesToRead.map(f => {
    const fullPath = path.join(root, f.path);
    let content = '';
    if (fs.existsSync(fullPath)) {
      content = fs.readFileSync(fullPath, 'utf-8');
    } else {
      content = `# File not found: ${f.path}`;
    }
    return {
      name: f.name,
      path: f.path,
      content,
      description: f.desc
    };
  });
}
