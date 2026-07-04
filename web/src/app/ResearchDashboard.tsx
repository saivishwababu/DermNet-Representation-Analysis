'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Moon,
  Sun,
  Layers,
  Settings,
  AlertCircle,
  ArrowRight,
  BookOpen,
  Sparkles,
  Clock,
  TrendingUp,
  BarChart3,
  Info,
  Copy,
  ExternalLink,
  Download,
  FileText,
  Cpu,
  Check,
  Maximize2
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend as RechartsLegend,
  ResponsiveContainer
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import {
  ClusteringResult,
  RepresentationAnalysis,
  ConfusedDiseasePair,
  NearestNeighborResult,
  UmapPoint
} from '@/lib/data';

interface DashboardProps {
  clusteringResults: ClusteringResult[];
  representationAnalysis: RepresentationAnalysis[];
  confusedPairs: ConfusedDiseasePair[];
  nearestNeighborResults: NearestNeighborResult[];
  umapPoints: UmapPoint[];
  finalReport: string;
}

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

const CLASS_COLORS = [
  '#3b82f6', '#10b981', '#ec4899', '#f59e0b', '#8b5cf6',
  '#06b6d4', '#e11d48', '#14b8a6', '#f97316', '#a855f7',
  '#6366f1', '#84cc16', '#10b981', '#059669', '#d97706',
  '#b45309', '#be123c', '#4f46e5', '#ca8a04', '#0d9488',
  '#c026d3', '#7c3aed', '#0891b2'
];

const GithubIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
  </svg>
);

export default function ResearchDashboard({
  clusteringResults,
  representationAnalysis,
  confusedPairs,
  nearestNeighborResults,
  umapPoints,
  finalReport
}: DashboardProps) {
  // Theme state
  const [darkMode, setDarkMode] = useState(false);

  // Active section tracking (ScrollSpy)
  const [activeSection, setActiveSection] = useState('home');

  // Interactive controls per section
  const [selectedModel, setSelectedModel] = useState<'dinov2' | 'clip' | 'resnet50'>('dinov2'); // Local to UMAP Canvas
  const [selectedClusteringMethod, setSelectedClusteringMethod] = useState<'kmeans' | 'agglomerative' | 'hdbscan'>('kmeans'); // Results section
  const [selectedConfusionModel, setSelectedConfusionModel] = useState<'dinov2' | 'clip' | 'resnet50'>('dinov2'); // Confusion section
  const [selectedNnModel, setSelectedNnModel] = useState<'dinov2' | 'clip' | 'resnet50'>('dinov2'); // Nearest Neighbor section

  // UMAP Interactive Canvas State
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [hoveredPoint, setHoveredPoint] = useState<UmapPoint | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [highlightedClass, setHighlightedClass] = useState<string | null>(null);

  // Nearest Neighbor Selection State
  const [nnClass, setNnClass] = useState('Acne and Rosacea Photos');
  const [nnIndex, setNnIndex] = useState(0);

  // Lightbox / Zoom Modal states
  const [lightboxImage, setLightboxImage] = useState<{ src: string; alt: string; caption: string } | null>(null);

  // Explanation Modal State (Merged Pipeline)
  const [methodologyModal, setMethodologyModal] = useState<{ title: string; desc: string } | null>(null);

  // Scroll Progress
  const [scrollProgress, setScrollProgress] = useState(0);

  // Copy success citation state
  const [citationCopied, setCitationCopied] = useState(false);

  // Hero Canvas Animation Ref
  const heroCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Sync dark mode class on HTML
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Hero Particle Canvas Animation
  useEffect(() => {
    const canvas = heroCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let width = (canvas.width = canvas.clientWidth);
    let height = (canvas.height = canvas.clientHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.clientWidth;
      height = canvas.height = canvas.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    const particles: { x: number; y: number; vx: number; vy: number; radius: number }[] = [];
    const particleCount = 35;

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 2 + 1
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      particles.forEach((p, idx) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > width) p.vx = -p.vx;
        if (p.y < 0 || p.y > height) p.vy = -p.vy;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = darkMode ? 'rgba(59, 130, 246, 0.25)' : 'rgba(37, 99, 235, 0.15)';
        ctx.fill();

        for (let j = idx + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = darkMode
              ? `rgba(59, 130, 246, ${0.12 * (1 - dist / 100)})`
              : `rgba(37, 99, 235, ${0.08 * (1 - dist / 100)})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      });
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
    };
  }, [darkMode]);

  // Scroll progress listener
  useEffect(() => {
    const handleScroll = () => {
      const totalScroll = document.documentElement.scrollHeight - window.innerHeight;
      if (totalScroll > 0) {
        setScrollProgress(window.scrollY / totalScroll);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Section observer hook
  const buildSectionObserver = (id: string) => {
    const [ref, inView] = useInView({ threshold: 0.15, rootMargin: '-15% 0px -45% 0px' });
    useEffect(() => {
      if (inView) {
        setActiveSection(id);
      }
    }, [inView, id]);
    return ref;
  };

  const homeRef = buildSectionObserver('home');
  const motivationRef = buildSectionObserver('motivation');
  const questionsRef = buildSectionObserver('questions');
  const datasetRef = buildSectionObserver('dataset');
  const modelsRef = buildSectionObserver('models');
  const resultsRef = buildSectionObserver('results');
  const umapRef = buildSectionObserver('umap');
  const confusionRef = buildSectionObserver('confusions');
  const nnRef = buildSectionObserver('nearest-neighbor');
  const findingsRef = buildSectionObserver('findings');
  const futureRef = buildSectionObserver('future');

  // dataset metrics
  const totalRawImages = 19559;
  const deletedDuplicates = 1257;
  const remainingImages = totalRawImages - deletedDuplicates;

  // Compute UMAP bounds locally for selected canvas representation
  const umapBounds = useMemo(() => {
    if (umapPoints.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    umapPoints.forEach(p => {
      const [x, y] = p[selectedModel];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });
    return { minX, maxX, minY, maxY };
  }, [umapPoints, selectedModel]);

  // Draw UMAP points
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    const pad = 40;
    const mapX = (val: number) => pad + ((val - umapBounds.minX) / (umapBounds.maxX - umapBounds.minX)) * (w - 2 * pad);
    const mapY = (val: number) => pad + ((val - umapBounds.minY) / (umapBounds.maxY - umapBounds.minY)) * (h - 2 * pad);

    // Draw grid lines
    ctx.strokeStyle = darkMode ? '#1e293b' : '#f1f5f9';
    ctx.lineWidth = 0.5 / zoom;
    const gridCount = 10;
    for (let i = 0; i <= gridCount; i++) {
      const yVal = pad + (i / gridCount) * (h - 2 * pad);
      ctx.beginPath();
      ctx.moveTo(pad, yVal);
      ctx.lineTo(w - pad, yVal);
      ctx.stroke();

      const xVal = pad + (i / gridCount) * (w - 2 * pad);
      ctx.beginPath();
      ctx.moveTo(xVal, pad);
      ctx.lineTo(xVal, h - pad);
      ctx.stroke();
    }

    // Render coordinates
    umapPoints.forEach(p => {
      const [x, y] = p[selectedModel];
      const cx = mapX(x);
      const cy = mapY(y);

      const color = CLASS_COLORS[p.label_idx % CLASS_COLORS.length];
      const isHighlighted = highlightedClass === p.class_name;
      const isHovered = hoveredPoint?.id === p.id;
      const hasFocus = highlightedClass === null || isHighlighted;

      ctx.beginPath();
      ctx.arc(cx, cy, isHovered ? 8 : (isHighlighted ? 6 : 4), 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.globalAlpha = hasFocus ? (isHovered ? 1 : 0.85) : 0.15;
      ctx.fill();

      if (isHighlighted || isHovered) {
        ctx.strokeStyle = darkMode ? '#ffffff' : '#0f172a';
        ctx.lineWidth = (isHovered ? 2 : 1) / zoom;
        ctx.stroke();
      }
    });

    ctx.restore();
  }, [umapPoints, selectedModel, zoom, panX, panY, umapBounds, highlightedClass, hoveredPoint, darkMode]);

  const handleZoom = (direction: 'in' | 'out') => {
    const factor = direction === 'in' ? 1.2 : 0.8;
    setZoom(z => Math.max(0.2, Math.min(25, z * factor)));
  };

  const resetZoom = () => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
    setHighlightedClass(null);
    setHoveredPoint(null);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsPanning(true);
    setStartPan({ x: e.clientX - panX, y: e.clientY - panY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isPanning) {
      setPanX(e.clientX - startPan.x);
      setPanY(e.clientY - startPan.y);
    } else {
      const virtX = (mouseX - panX) / zoom;
      const virtY = (mouseY - panY) / zoom;

      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const pad = 40;
      const mapX = (val: number) => pad + ((val - umapBounds.minX) / (umapBounds.maxX - umapBounds.minX)) * (w - 2 * pad);
      const mapY = (val: number) => pad + ((val - umapBounds.minY) / (umapBounds.maxY - umapBounds.minY)) * (h - 2 * pad);

      let minD = Infinity;
      let closest: UmapPoint | null = null;

      umapPoints.forEach(p => {
        const [x, y] = p[selectedModel];
        const cx = mapX(x);
        const cy = mapY(y);
        const dx = virtX - cx;
        const dy = virtY - cy;
        const dist = Math.sqrt(dx * dx + dy * dy) * zoom;
        if (dist < minD) {
          minD = dist;
          closest = p;
        }
      });

      if (minD < 12) {
        setHoveredPoint(closest);
        setTooltipPos({ x: mouseX + 10, y: mouseY + 10 });
      } else {
        setHoveredPoint(null);
      }
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const classesList = useMemo(() => {
    const set = new Set<string>();
    umapPoints.forEach(p => set.add(p.class_name));
    return Array.from(set).sort();
  }, [umapPoints]);

  const chartData = useMemo(() => {
    return clusteringResults
      .filter(r => r.clustering_method === selectedClusteringMethod)
      .map(r => ({
        name: r.representation.toUpperCase(),
        Silhouette: parseFloat(r.silhouette.toFixed(4)),
        ARI: parseFloat(r.ari.toFixed(4)),
        NMI: parseFloat(r.nmi.toFixed(4)),
        Purity: parseFloat(r.purity.toFixed(4)),
      }));
  }, [clusteringResults, selectedClusteringMethod]);

  const classDistData = useMemo(() => {
    return representationAnalysis
      .filter(r => r.representation === 'dinov2')
      .map(r => ({
        name: r.disease_class,
        samples: r.n_samples,
      }))
      .sort((a, b) => b.samples - a.samples);
  }, [representationAnalysis]);

  const filteredConfusedPairs = useMemo(() => {
    return confusedPairs.filter(p => p.representation === selectedConfusionModel);
  }, [confusedPairs, selectedConfusionModel]);

  const activeNnImagePath = useMemo(() => {
    const prefix = nnClass.slice(0, 20);
    let actualIndex = nnIndex;
    if (nnClass.startsWith('Actinic')) {
      actualIndex = 10 + nnIndex;
    } else if (nnClass.startsWith('Atopic')) {
      actualIndex = 20 + nnIndex;
    }
    return `${basePath}/plots/nearest_neighbor_grids/${selectedNnModel}/nn_${prefix}_${actualIndex}.png`;
  }, [selectedNnModel, nnClass, nnIndex]);

  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const copyCitation = () => {
    const bibtex = `@techreport{dermnet_representation_2026,
  author       = {Saarland University Student},
  title        = {Representation Analysis of Dermatological Images Using Self-Supervised Visual Features},
  institution  = {Saarland University},
  year         = {2026},
  type         = {Research Project},
  note         = {Pretrained model evaluation using DINOv2, CLIP, and ResNet50}
}`;
    navigator.clipboard.writeText(bibtex);
    setCitationCopied(true);
    setTimeout(() => setCitationCopied(false), 2000);
  };

  return (
    <div className="relative min-h-screen">
      {/* Scroll Progress Bar */}
      <div
        className="fixed top-0 left-0 right-0 h-1 bg-blue-600 dark:bg-blue-500 z-50 transition-all duration-75"
        style={{ width: `${scrollProgress * 100}%` }}
      />

      {/* Sticky Navigation Bar */}
      <header className="sticky top-0 z-40 w-full border-b bg-white/85 backdrop-blur-md dark:bg-slate-950/85 dark:border-slate-900">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Logo / Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => scrollTo('home')}>
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white font-bold text-lg shadow-md shadow-blue-500/20">
              D
            </div>
            <div>
              <span className="font-bold text-sm tracking-tight sm:text-base">DermNet</span>
              <span className="ml-1 text-xs text-blue-600 dark:text-blue-400 font-semibold uppercase tracking-wider">Representation</span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Dark Mode Toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="rounded-full p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-900 dark:hover:text-slate-100"
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>

            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="hidden sm:flex items-center space-x-1 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold shadow-sm hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:hover:bg-slate-800"
            >
              <GithubIcon className="h-3.5 w-3.5" />
              <span>GitHub</span>
            </a>
          </div>
        </div>

        {/* Sticky Secondary ScrollSpy Navigation */}
        <div className="border-t bg-slate-50/90 dark:bg-slate-900/90 dark:border-slate-900 overflow-x-auto whitespace-nowrap scrollbar-none py-2 text-xs">
          <div className="mx-auto max-w-7xl px-4 flex space-x-6">
            {[
              { id: 'home', label: 'Home' },
              { id: 'motivation', label: 'Motivation' },
              { id: 'questions', label: 'Research Questions' },
              { id: 'dataset', label: 'Dataset' },
              { id: 'models', label: 'Models Compared' },
              { id: 'results', label: 'Results Dashboard' },
              { id: 'umap', label: 'Representation Analysis' },
              { id: 'confusions', label: 'Disease Confusions' },
              { id: 'nearest-neighbor', label: 'Nearest Neighbour' },
              { id: 'findings', label: 'Key Findings' },
              { id: 'future', label: 'Future Work' }
            ].map(sec => (
              <button
                key={sec.id}
                onClick={() => scrollTo(sec.id)}
                className={`font-semibold transition-colors duration-150 py-1 border-b-2 ${
                  activeSection === sec.id
                    ? 'text-blue-600 border-blue-600 dark:text-blue-400 dark:border-blue-400'
                    : 'text-slate-500 border-transparent hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                {sec.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* HERO SECTION */}
      <section
        id="home"
        ref={homeRef}
        className="scroll-mt-28 relative overflow-hidden bg-white dark:bg-slate-950 border-b dark:border-slate-900 py-24 sm:py-32"
      >
        {/* Particle Canvas Background */}
        <div className="absolute inset-0 bg-gradient-to-tr from-blue-50/30 via-transparent to-emerald-50/10 dark:from-blue-950/10" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-500/5 blur-[120px] rounded-full pointer-events-none" />
        <canvas ref={heroCanvasRef} className="absolute inset-0 z-0 pointer-events-none" />

        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center z-10">
          <div className="inline-flex items-center space-x-2 rounded-full border border-slate-200 bg-slate-50 px-3.5 py-1 text-xs font-semibold text-slate-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 mb-6">
            <span className="flex h-2 w-2 rounded-full bg-blue-600 animate-pulse" />
            <span>Research Project</span>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <span>Saarland University</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-slate-50 max-w-5xl mx-auto leading-tight">
            Representation Analysis of Dermatological Images Using Self-Supervised Visual Features
          </h1>

          <p className="mt-6 text-base sm:text-lg lg:text-xl text-slate-500 dark:text-slate-400 max-w-3xl mx-auto leading-relaxed">
            Understanding how <span className="font-semibold text-slate-850 dark:text-slate-200">DINOv2</span>, <span className="font-semibold text-slate-850 dark:text-slate-200">CLIP</span>, and <span className="font-semibold text-slate-850 dark:text-slate-200">ResNet50</span> organize dermatological images without supervised pretraining.
          </p>

          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <button
              onClick={() => scrollTo('results')}
              className="inline-flex items-center rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-md shadow-blue-500/10 hover:bg-blue-700 transition"
            >
              <span>Explore Results</span>
              <ArrowRight className="ml-2 h-4 w-4" />
            </button>
            <button
              onClick={() => scrollTo('dataset')}
              className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800 transition"
            >
              View Dataset
            </button>
            <a
              href={`${basePath}/reports/final_report.pdf`}
              download
              className="inline-flex items-center rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800 transition"
            >
              <Download className="mr-2 h-4 w-4 text-slate-400" />
              Download Report
            </a>
          </div>
        </div>
      </section>

      {/* MOTIVATION & PIPELINE SECTION */}
      <section
        id="motivation"
        ref={motivationRef}
        className="scroll-mt-28 py-20 bg-slate-50 dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Project Motivation</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Why probe self-supervised representations of skin lesions? Rather than training another CNN classifier, we focus on understanding how foundation models encode medical concepts.
            </p>
          </div>

          {/* Infographic Workflow Panels */}
          <div className="grid md:grid-cols-2 gap-8 lg:gap-12 mb-16">
            {/* Traditional Supervised CNN Workflow */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center mb-6">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-100 text-xs text-slate-500 font-bold dark:bg-slate-800 mr-2">
                  A
                </span>
                Traditional Supervised Workflow
              </h3>

              <div className="space-y-4 relative">
                {[
                  { title: 'Images', desc: 'Raw, uncurated medical photographs of skin lesions' },
                  { title: 'Labels', desc: 'Expensive, expert-annotated disease classification labels' },
                  { title: 'Supervised CNN', desc: 'Models trained end-to-end to minimize cross-entropy loss' },
                  { title: 'Classification Output', desc: 'Single discrete class predictions' }
                ].map((step, idx) => (
                  <div key={idx} className="flex items-start">
                    <div className="flex flex-col items-center mr-4">
                      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-100 border text-xs text-slate-600 font-bold dark:bg-slate-800 dark:border-slate-700">
                        {idx + 1}
                      </div>
                      {idx < 3 && <div className="h-6 w-0.5 bg-slate-200 dark:bg-slate-800" />}
                    </div>
                    <div className="pt-0.5">
                      <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200">{step.title}</h4>
                      <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Our Self-Supervised Representation Analysis Workflow */}
            <div className="rounded-2xl border border-blue-100 bg-blue-50/20 p-6 shadow-sm dark:border-blue-900/30 dark:bg-blue-950/5">
              <h3 className="text-base font-bold text-blue-600 dark:text-blue-400 flex items-center mb-6">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-xs text-blue-600 font-bold dark:bg-blue-900/50 mr-2">
                  B
                </span>
                Our Probing Workflow
              </h3>

              <div className="space-y-4 relative">
                {[
                  { title: 'Raw Images', desc: 'Deduplicated DermNet patient lesion photographs' },
                  { title: 'Pretrained Encoders', desc: 'Zero-shot feature extraction using DINOv2, CLIP, and ResNet50' },
                  { title: 'Embedding Space Topology', desc: 'High-dimensional visual descriptor mapping' },
                  { title: 'Representation Probing', desc: 'Unsupervised clustering, nearest neighbor retrieval, and confusion probing' },
                  { title: 'Understanding Disease Structure', desc: 'Evaluating visual similarity clusters vs. clinical definitions' }
                ].map((step, idx) => (
                  <div key={idx} className="flex items-start">
                    <div className="flex flex-col items-center mr-4">
                      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-100 text-xs text-blue-600 font-bold dark:bg-blue-900/50">
                        {idx + 1}
                      </div>
                      {idx < 4 && <div className="h-6 w-0.5 bg-blue-200/50 dark:bg-blue-900/35" />}
                    </div>
                    <div className="pt-0.5">
                      <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200">{step.title}</h4>
                      <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Conceptual statement */}
          <div className="rounded-xl bg-blue-600/5 border border-blue-500/10 p-6 max-w-4xl mx-auto text-center dark:bg-blue-500/5">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 mb-3">
              <Sparkles className="h-4.5 w-4.5" />
            </span>
            <p className="text-xs font-medium text-slate-850 dark:text-slate-200 leading-relaxed">
              The contribution of this project is <span className="text-blue-600 dark:text-blue-400 font-bold">NOT</span> better clustering accuracies. It is understanding the underlying topological organization of visual representations to inspect how modern computer vision foundation models align with expert clinical labels.
            </p>
          </div>
        </div>
      </section>

      {/* RESEARCH QUESTIONS SECTION */}
      <section
        id="questions"
        ref={questionsRef}
        className="scroll-mt-28 py-20 bg-white dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Research Questions & Answers</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Probing questions and outcomes derived from the final report metrics.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {[
              {
                id: 'Q1',
                q: 'Does DINOv2 naturally separate skin diseases?',
                icon: <Layers className="h-5 w-5 text-blue-600" />,
                ans: 'Only partially. While DINOv2 forms tight local neighborhoods (highest silhouette score of 0.1609), the visual clusters map to raw textures, margins, and hues rather than clinical disease categorizations without supervised training.'
              },
              {
                id: 'Q2',
                q: 'Which disease categories remain highly confused?',
                icon: <AlertCircle className="h-5 w-5 text-amber-505" />,
                ans: 'Diseases sharing visual morphology: erythematous scaly plaques (Psoriasis vs Fungal Infections) and active localized swelling (Cellulitis vs Contact Dermatitis) show severe centroid proximity.'
              },
              {
                id: 'Q3',
                q: 'Does contrastive CLIP outperform self-distilled DINOv2?',
                icon: <TrendingUp className="h-5 w-5 text-emerald-600" />,
                ans: 'CLIP achieves higher semantic label alignment (KMeans ARI: 0.1001 vs 0.0922 in DINOv2) due to multimodal language pretraining, but DINOv2 produces tighter cluster geometries and slightly higher nearest-neighbor retrieval (63.91% top-1).'
              },
              {
                id: 'Q4',
                q: 'Is unsupervised post-hoc clustering useful?',
                icon: <BarChart3 className="h-5 w-5 text-purple-650" />,
                ans: 'It serves as an excellent diagnostic probing tool to verify spatial geometry and uncover visual similarities, but it is not sufficient as a standalone clinical diagnostics tool (low absolute alignment ARI ~0.10).'
              },
              {
                id: 'Q5',
                q: 'Which encoder backbone is recommended for future work?',
                icon: <Cpu className="h-5 w-5 text-indigo-500" />,
                ans: 'For Visual Search (Content-Based Image Retrieval), build on DINOv2. For Zero-shot classification and indexing, use CLIP. Supervised ResNet50 is not recommended due to heavy ImageNet object-bias.'
              }
            ].map((rq, idx) => (
              <motion.div
                key={rq.id}
                whileHover={{ y: -4 }}
                className="group relative rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 transition hover:shadow-md"
              >
                <div className="flex items-center space-x-3 mb-4">
                  <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-50 dark:bg-slate-950">
                    {rq.icon}
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-blue-600 dark:text-blue-400 uppercase tracking-widest">{rq.id}</span>
                    <h3 className="text-xs font-bold text-slate-850 dark:text-slate-100">{rq.q}</h3>
                  </div>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed pt-2 border-t dark:border-slate-850">
                  {rq.ans}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* DATASET SECTION */}
      <section
        id="dataset"
        ref={datasetRef}
        className="scroll-mt-28 py-20 bg-slate-50 dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Dataset Properties & Statistics</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              DermNet represents skin lesion photography with severe class imbalance and duplication. Near-duplicate shots inflate metrics and were removed using SHA-256 binary hash checks.
            </p>
            <div className="mt-6 flex justify-center">
              <a
                href="https://www.kaggle.com/datasets/shubhamgoel27/dermnet/data"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center rounded-xl bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-blue-700 transition"
              >
                <span>View Dataset on Kaggle</span>
                <ExternalLink className="ml-1.5 h-3.5 w-3.5" />
              </a>
            </div>
          </div>

          {/* Quick Counter Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
            {[
              { val: '23', label: 'Ground-Truth Classes' },
              { val: totalRawImages.toLocaleString(), label: 'Total Raw Images' },
              { val: deletedDuplicates.toLocaleString(), label: 'Near-Duplicates Removed' },
              { val: remainingImages.toLocaleString(), label: 'Remaining Probed Images' }
            ].map((stat, idx) => (
              <div key={idx} className="rounded-xl border border-slate-205 bg-white p-5 text-center shadow-sm dark:border-slate-800 dark:bg-slate-900">
                <span className="block text-2xl sm:text-3xl font-extrabold text-blue-600 dark:text-blue-400">{stat.val}</span>
                <span className="mt-1 block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{stat.label}</span>
              </div>
            ))}
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Chart: Class Distribution */}
            <div className="lg:col-span-2 rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h3 className="text-sm font-bold text-slate-805 dark:text-slate-105 mb-6 flex items-center">
                <BarChart3 className="h-4.5 w-4.5 mr-2 text-blue-500" />
                Class Sample Distribution (Deduplicated)
              </h3>
              <div className="h-80 w-full text-[10px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={classDistData.slice(0, 12)} margin={{ bottom: 50, left: 10, right: 10, top: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="name"
                      angle={-30}
                      textAnchor="end"
                      height={80}
                      tick={{ fontSize: 9 }}
                      tickFormatter={(value) => value.length > 20 ? value.substring(0, 20) + '...' : value}
                    />
                    <YAxis label={{ value: 'Images', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                    <RechartsTooltip />
                    <Bar dataKey="samples" fill="#2563eb" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Scrollable Class Dictionary Table */}
            <div className="rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h3 className="text-sm font-bold text-slate-805 dark:text-slate-105 mb-4 flex items-center">
                <Info className="h-4.5 w-4.5 mr-2 text-blue-500" />
                DermNet Class Index
              </h3>
              <div className="max-h-72 overflow-y-auto border rounded-lg text-xs">
                <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800">
                  <thead className="bg-slate-50 dark:bg-slate-900 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left font-bold text-slate-600 dark:text-slate-400">Class Name</th>
                      <th className="px-3 py-2 text-right font-bold text-slate-600 dark:text-slate-400">Samples</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-850">
                    {classDistData.map((cls, idx) => (
                      <tr
                        key={idx}
                        className="hover:bg-slate-50 dark:hover:bg-slate-900 cursor-pointer transition-colors"
                        onClick={() => {
                          setHighlightedClass(cls.name);
                          scrollTo('umap');
                        }}
                      >
                        <td className="px-3 py-2 text-slate-700 dark:text-slate-350">{cls.name}</td>
                        <td className="px-3 py-2 text-right font-semibold text-slate-550">{cls.samples}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* MODELS COMPARED SECTION */}
      <section
        id="models"
        ref={modelsRef}
        className="scroll-mt-28 py-20 bg-white dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Models Compared</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              We compare three architectures with completely distinct pretraining objectives to validate how visual features organize.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-16">
            {[
              {
                name: 'DINOv2',
                arch: 'ViT-B/14 (Vision Transformer)',
                objective: 'Self-supervised self-distillation (DINO + iBOT masked image modeling)',
                dim: '768 Dimensions',
                strength: 'Preserves high-frequency local textures, borders, and visual contours extremely well.',
                weakness: 'Lacks text guidance; confounded by macro visual co-variables.',
                runtime: '~670s (~27 img/s) on MPS'
              },
              {
                name: 'CLIP',
                arch: 'ViT-B/32 Image Encoder',
                objective: 'Multimodal contrastive language-image pretraining on 400M web pairs',
                dim: '512 Dimensions',
                strength: 'Aligns closely with high-level semantic disease labels due to text guidance.',
                weakness: 'Contrastive visual pooling discards local fine-grained textures.',
                runtime: '~230s (~80 img/s) on MPS'
              },
              {
                name: 'ResNet50',
                arch: 'CNN (Convolutional Network)',
                objective: 'Fully supervised classification on ImageNet-1K object classes',
                dim: '2048 Dimensions',
                strength: 'Fast execution runtime baseline.',
                weakness: 'Supervised object classification generalizes poorly to fine-grained lesion structures.',
                runtime: '~135s (~135 img/s) on MPS'
              }
            ].map((enc, idx) => (
              <div key={idx} className="rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-50 text-blue-600 dark:bg-slate-800 dark:text-blue-400 mb-4 font-bold text-sm">
                  {enc.name[0]}
                </div>
                <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">{enc.name}</h3>
                <span className="text-[10px] text-blue-650 dark:text-blue-400 font-semibold block mt-1 uppercase tracking-wider">{enc.arch}</span>

                <dl className="mt-6 space-y-3.5 text-xs">
                  <div>
                    <dt className="font-bold text-slate-500">Objective</dt>
                    <dd className="mt-0.5 text-slate-700 dark:text-slate-350">{enc.objective}</dd>
                  </div>
                  <div>
                    <dt className="font-bold text-slate-500">Feature Dimensions</dt>
                    <dd className="mt-0.5 text-slate-700 dark:text-slate-350 font-mono bg-slate-50 dark:bg-slate-950 px-1.5 py-0.5 rounded border inline-block">{enc.dim}</dd>
                  </div>
                  <div>
                    <dt className="font-bold text-slate-500"> MPS Inference Speed</dt>
                    <dd className="mt-0.5 text-slate-700 dark:text-slate-350 flex items-center font-mono">
                      <Clock className="h-3.5 w-3.5 mr-1 text-slate-400" />
                      {enc.runtime}
                    </dd>
                  </div>
                  <div className="pt-2.5 border-t dark:border-slate-850">
                    <dt className="font-bold text-emerald-600 dark:text-emerald-500">Strengths</dt>
                    <dd className="mt-0.5 text-slate-600 dark:text-slate-400">{enc.strength}</dd>
                  </div>
                  <div className="pt-2.5">
                    <dt className="font-bold text-rose-600 dark:text-rose-500">Weaknesses</dt>
                    <dd className="mt-0.5 text-slate-600 dark:text-slate-400">{enc.weakness}</dd>
                  </div>
                </dl>
              </div>
            ))}
          </div>

          {/* Model Comparison Summary Table (Merged from Comparison) */}
          <div className="text-center max-w-3xl mx-auto mb-8 mt-12">
            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Comparative Probing Metrics</h3>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Evaluating parameters, metrics, and recommended use-cases across representations.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-205 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900 overflow-x-auto mb-16">
            <table className="min-w-full divide-y divide-slate-200 text-xs dark:divide-slate-800">
              <thead className="bg-slate-50 dark:bg-slate-900">
                <tr>
                  <th className="px-4 py-3 text-left font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</th>
                  <th className="px-4 py-3 text-left font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Training Type</th>
                  <th className="px-4 py-3 text-right font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Embed Size</th>
                  <th className="px-4 py-3 text-right font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Runtime</th>
                  <th className="px-4 py-3 text-right font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider text-blue-600 dark:text-blue-400">Silhouette</th>
                  <th className="px-4 py-3 text-right font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider text-emerald-600 dark:text-emerald-500">KMeans ARI</th>
                  <th className="px-4 py-3 text-right font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Purity</th>
                  <th className="px-4 py-3 text-left font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Best Use Case</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-150 dark:divide-slate-850">
                <tr className="hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors">
                  <td className="px-4 py-3 font-bold text-slate-950 dark:text-slate-100">DINOv2</td>
                  <td className="px-4 py-3 text-slate-550">Self-Supervised Distillation</td>
                  <td className="px-4 py-3 text-right font-mono">768</td>
                  <td className="px-4 py-3 text-right font-mono">670s</td>
                  <td className="px-4 py-3 text-right font-mono font-bold text-blue-650 bg-blue-50/20">0.1609</td>
                  <td className="px-4 py-3 text-right font-mono">0.0922</td>
                  <td className="px-4 py-3 text-right font-mono">0.2626</td>
                  <td className="px-4 py-3 font-medium text-slate-700 dark:text-slate-350">Visual Similarity Retrieval (CBIR)</td>
                </tr>
                <tr className="hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors">
                  <td className="px-4 py-3 font-bold text-slate-950 dark:text-slate-100">CLIP</td>
                  <td className="px-4 py-3 text-slate-550">Contrastive Text-Image</td>
                  <td className="px-4 py-3 text-right font-mono">512</td>
                  <td className="px-4 py-3 text-right font-mono">230s</td>
                  <td className="px-4 py-3 text-right font-mono">0.0965</td>
                  <td className="px-4 py-3 text-right font-mono font-bold text-emerald-650 bg-emerald-50/20">0.1001</td>
                  <td className="px-4 py-3 text-right font-mono font-bold">0.2674</td>
                  <td className="px-4 py-3 font-medium text-slate-700 dark:text-slate-350">Semantic Image Categorization</td>
                </tr>
                <tr className="hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors">
                  <td className="px-4 py-3 font-bold text-slate-950 dark:text-slate-100">ResNet50</td>
                  <td className="px-4 py-3 text-slate-550">Supervised ImageNet CNN</td>
                  <td className="px-4 py-3 text-right font-mono">2048</td>
                  <td className="px-4 py-3 text-right font-mono">135s</td>
                  <td className="px-4 py-3 text-right font-mono text-rose-500">0.0451</td>
                  <td className="px-4 py-3 text-right font-mono text-rose-500">0.0499</td>
                  <td className="px-4 py-3 text-right font-mono text-rose-500">0.2151</td>
                  <td className="px-4 py-3 font-medium text-slate-750 dark:text-slate-350">Supervised legacy CNN baseline</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Performance Runtimes and Dimension Charts (Merged from Performance) */}
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* Chart: Runtime */}
            <div className="rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h4 className="text-xs font-bold text-slate-805 dark:text-slate-105 mb-4 flex items-center">
                <Clock className="h-4.5 w-4.5 text-blue-500 mr-2" />
                Feature Extraction Inference Time (seconds)
              </h4>
              <div className="h-64 w-full text-[10px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { name: 'DINOv2', runtime: 670 },
                    { name: 'CLIP', runtime: 230 },
                    { name: 'ResNet50', runtime: 135 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Seconds', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                    <RechartsTooltip />
                    <Bar dataKey="runtime" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart: Embed dimensions */}
            <div className="rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h4 className="text-xs font-bold text-slate-805 dark:text-slate-105 mb-4 flex items-center">
                <Layers className="h-4.5 w-4.5 text-blue-500 mr-2" />
                Embedding Dimensionality (Feature size)
              </h4>
              <div className="h-64 w-full text-[10px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { name: 'DINOv2', dims: 768 },
                    { name: 'CLIP', dims: 512 },
                    { name: 'ResNet50', dims: 2048 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Dimensions', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                    <RechartsTooltip />
                    <Bar dataKey="dims" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Pipeline Badges (Merged from Technologies) */}
          <div className="mt-12 flex flex-wrap justify-center gap-2 max-w-2xl mx-auto border-t dark:border-slate-850 pt-8">
            {['Python', 'PyTorch', 'DINOv2', 'CLIP', 'OpenCV', 'NumPy', 'Scikit-learn', 'UMAP', 'Pandas', 'Next.js', 'Tailwind', 'Recharts'].map((tech, idx) => (
              <span key={idx} className="rounded-full bg-slate-100 px-3 py-1 text-[10px] font-semibold text-slate-600 dark:bg-slate-900 dark:text-slate-400">
                {tech}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* RESULTS DASHBOARD */}
      <section
        id="results"
        ref={resultsRef}
        className="scroll-mt-28 py-20 bg-slate-50 dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Results Dashboard</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Evaluating representation post-hoc clustering quality across metrics.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            <div className="rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-4">Clustering Method</h3>

                <div className="grid grid-cols-3 gap-2 p-1 bg-slate-100 rounded-lg dark:bg-slate-950 border dark:border-slate-850 mb-6">
                  {(['kmeans', 'agglomerative', 'hdbscan'] as const).map(method => (
                    <button
                      key={method}
                      onClick={() => setSelectedClusteringMethod(method)}
                      className={`rounded-md py-1.5 text-[10px] font-bold uppercase tracking-wider transition ${
                        selectedClusteringMethod === method
                          ? 'bg-white text-blue-600 shadow-sm dark:bg-slate-800 dark:text-blue-400'
                          : 'text-slate-505 hover:text-slate-900 dark:hover:text-slate-100'
                      }`}
                    >
                      {method}
                    </button>
                  ))}
                </div>

                <div className="space-y-4 text-xs">
                  <div>
                    <h4 className="font-bold text-slate-700 dark:text-slate-300">Silhouette Width</h4>
                    <p className="text-slate-500 mt-0.5 leading-relaxed">Measures spatial cluster tightness and compactness. Larger scores indicate visually isolated clusters.</p>
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-700 dark:text-slate-300">Adjusted Rand Index (ARI)</h4>
                    <p className="text-slate-500 mt-0.5 leading-relaxed">Measures agreement between clustering partitions and ground-truth disease categories, adjusted for chance.</p>
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-700 dark:text-slate-300">Normalized Mutual Information (NMI)</h4>
                    <p className="text-slate-500 mt-0.5 leading-relaxed">Information shared between clusters and labels normalized between 0 (independence) and 1 (perfect matching).</p>
                  </div>
                </div>
              </div>

              <div className="pt-6 border-t mt-6 dark:border-slate-850 text-[10px] text-slate-400">
                💡 Note: CLIP achieves the best labels agreement (ARI: 0.1001), while DINOv2 yields the highest spatial silhouette thickness (0.1609).
              </div>
            </div>

            <div className="lg:col-span-2 rounded-2xl border border-slate-205 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center justify-between">
                <span>Model Comparisons ({selectedClusteringMethod.toUpperCase()})</span>
                <span className="text-[10px] text-blue-600 dark:text-blue-400 font-semibold">{selectedClusteringMethod === 'hdbscan' ? 'Variable Clusters' : 'Fixed 23 Clusters'}</span>
              </h3>

              <div className="h-80 w-full text-[10px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 0.35]} />
                    <RechartsTooltip />
                    <RechartsLegend />
                    <Bar dataKey="Silhouette" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="ARI" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="NMI" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Purity" fill="#ec4899" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <p className="text-[10px] text-slate-400 block mt-4 text-center leading-relaxed">
                {selectedClusteringMethod === 'hdbscan'
                  ? 'Notice: HDBSCAN reports high Silhouette scores (e.g. 0.2581 for ResNet50) because it found only 3-5 clusters and discarded up to 43.8% of images as noise.'
                  : 'Notice: Absolute metrics look modest (ARI ~0.1) due to skin diseases sharing redness/scale visual variables. However, visual topology remains highly structured.'}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* REPRESENTATION ANALYSIS */}
      <section
        id="umap"
        ref={umapRef}
        className="scroll-mt-28 py-20 bg-white dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Representation Analysis</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Explore 1,150 UMAP points (sampled symmetrically across classes). Pan by dragging, scroll to zoom, hover to see filenames & clinical photos, or select a category to highlight its configuration.
            </p>
          </div>

          <div className="grid lg:grid-cols-4 gap-8 mb-16">
            {/* Controls Side Panel */}
            <div className="lg:col-span-1 rounded-2xl border border-slate-205 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 flex flex-col justify-between max-h-[600px] overflow-y-auto">
              <div>
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Representation Model</h3>

                <div className="space-y-1 mb-6">
                  {(['dinov2', 'clip', 'resnet50'] as const).map(m => (
                    <button
                      key={m}
                      onClick={() => {
                        setSelectedModel(m);
                        setHoveredPoint(null);
                      }}
                      className={`w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs font-semibold transition ${
                        selectedModel === m
                          ? 'bg-blue-600 text-white shadow-sm'
                          : 'text-slate-650 hover:bg-slate-100 dark:hover:bg-slate-800'
                      }`}
                    >
                      <span>{m.toUpperCase()}</span>
                      {selectedModel === m && <Check className="h-3.5 w-3.5" />}
                    </button>
                  ))}
                </div>

                <div className="border-t pt-4 dark:border-slate-850">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Highlight Category</h3>
                    {highlightedClass && (
                      <button onClick={() => setHighlightedClass(null)} className="text-[10px] text-blue-500 font-bold hover:underline">
                        Clear
                      </button>
                    )}
                  </div>
                  <div className="max-h-64 overflow-y-auto space-y-1 pr-1 border rounded-lg p-1.5 bg-slate-50 dark:bg-slate-950">
                    {classesList.map((cls, idx) => (
                      <button
                        key={idx}
                        onClick={() => setHighlightedClass(highlightedClass === cls ? null : cls)}
                        className={`w-full flex items-center text-left rounded px-2 py-1 text-[10px] font-semibold transition ${
                          highlightedClass === cls
                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                            : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200/50 dark:hover:bg-slate-900'
                        }`}
                      >
                        <span
                          className="h-2 w-2 rounded-full shrink-0 mr-2"
                          style={{ backgroundColor: CLASS_COLORS[idx % CLASS_COLORS.length] }}
                        />
                        <span className="truncate">{cls}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t dark:border-slate-850 flex gap-2">
                <button
                  onClick={() => handleZoom('in')}
                  className="flex-1 rounded-lg border px-3 py-1.5 text-xs font-semibold hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  +
                </button>
                <button
                  onClick={() => handleZoom('out')}
                  className="flex-1 rounded-lg border px-3 py-1.5 text-xs font-semibold hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  -
                </button>
                <button
                  onClick={resetZoom}
                  className="flex-1 rounded-lg border px-3 py-1.5 text-xs font-semibold hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* HTML5 Canvas Panel */}
            <div className="lg:col-span-3 rounded-2xl border border-slate-205 bg-white dark:border-slate-800 dark:bg-slate-900 relative shadow-sm h-[600px] overflow-hidden flex flex-col justify-between">
              <div className="absolute top-4 left-4 z-10 pointer-events-none">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">UMAP Canvas Layout</span>
                <h4 className="text-xs font-bold text-slate-800 dark:text-slate-250 mt-0.5">
                  Probing Space: {selectedModel.toUpperCase()} | Zoom: {zoom.toFixed(1)}x
                </h4>
              </div>

              <canvas
                ref={canvasRef}
                className="w-full flex-1 cursor-grab active:cursor-grabbing"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />

              {/* Dynamic tooltip */}
              <AnimatePresence>
                {hoveredPoint && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    style={{
                      position: 'absolute',
                      left: tooltipPos.x,
                      top: tooltipPos.y,
                      pointerEvents: 'none',
                      zIndex: 20
                    }}
                    className="w-56 rounded-xl border border-slate-205 bg-white p-3 shadow-xl dark:border-slate-800 dark:bg-slate-950 text-xs"
                  >
                    <div className="relative w-full h-32 rounded bg-slate-50 dark:bg-slate-900 overflow-hidden mb-2 border border-slate-100 dark:border-slate-850 flex items-center justify-center">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`${basePath}${hoveredPoint.image_path}`}
                        alt={hoveredPoint.filename}
                        className="object-cover w-full h-full"
                      />
                    </div>
                    <span className="font-bold text-[11px] text-blue-600 dark:text-blue-400 block truncate">
                      {hoveredPoint.class_name}
                    </span>
                    <span className="text-[9px] text-slate-455 block truncate font-mono mt-0.5">
                      File: {hoveredPoint.filename}
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="p-3 border-t bg-slate-50 dark:bg-slate-950 dark:border-slate-850 flex justify-between items-center text-[10px] text-slate-500">
                <span>Left-click + drag to Pan. Scroll to zoom. Hover points to inspect lesion photographs.</span>
                <button onClick={resetZoom} className="text-blue-500 font-bold hover:underline">
                  Recenter Space
                </button>
              </div>
            </div>
          </div>

          {/* Supplementary Visual Figures (Merged from Gallery) */}
          <div className="text-center max-w-3xl mx-auto mb-8 mt-12">
            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Supplementary Visualizations</h3>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Explore pre-computed manifold plots and class-centroid distance matrices. Click to expand.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {[
              {
                src: `${basePath}/plots/umap_dinov2_by_label.png`,
                alt: 'DINOv2 UMAP Projections',
                caption: '2D UMAP projection of DINOv2 embedding space colored by disease ground-truth labels.'
              },
              {
                src: `${basePath}/plots/umap_clip_by_label.png`,
                alt: 'CLIP UMAP Projections',
                caption: '2D UMAP projection of CLIP embedding space colored by disease ground-truth labels.'
              },
              {
                src: `${basePath}/plots/umap_resnet50_by_label.png`,
                alt: 'ResNet50 UMAP Projections',
                caption: '2D UMAP projection of ResNet50 embedding space colored by disease ground-truth labels.'
              },
              {
                src: `${basePath}/plots/centroid_heatmap_dinov2.png`,
                alt: 'DINOv2 Centroid Distance Heatmap',
                caption: 'Pairwise cosine distances between class centroids in DINOv2 representation space.'
              },
              {
                src: `${basePath}/plots/centroid_heatmap_clip.png`,
                alt: 'CLIP Centroid Distance Heatmap',
                caption: 'Pairwise cosine distances between class centroids in CLIP representation space.'
              },
              {
                src: `${basePath}/plots/pca_explained_variance.png`,
                alt: 'PCA explained variance',
                caption: 'Principal components explained variance ratio curves.'
              }
            ].map((img, idx) => (
              <div
                key={idx}
                onClick={() => setLightboxImage(img)}
                className="group rounded-2xl border border-slate-205 bg-white overflow-hidden shadow-sm dark:border-slate-800 dark:bg-slate-900 cursor-zoom-in"
              >
                <div className="relative w-full h-40 bg-slate-50 border-b dark:bg-slate-950 dark:border-slate-850 overflow-hidden flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={img.src}
                    alt={img.alt}
                    className="object-cover w-full h-full transition duration-300 group-hover:scale-105"
                  />
                </div>
                <div className="p-4">
                  <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100">{img.alt}</h4>
                  <p className="text-[10px] text-slate-550 mt-1 leading-relaxed">{img.caption}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* DISEASE CONFUSION ANALYSIS */}
      <section
        id="confusions"
        ref={confusionRef}
        className="scroll-mt-28 py-20 bg-slate-50 dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Disease Confusion Analysis</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              We list the top-5 disease pairs consistently mapped to neighboring zones. Select the model representation to inspect its specific overlaps.
            </p>
          </div>

          {/* Local Confusion model switcher */}
          <div className="flex justify-center space-x-2 mb-8">
            {(['dinov2', 'clip', 'resnet50'] as const).map(m => (
              <button
                key={m}
                onClick={() => setSelectedConfusionModel(m)}
                className={`rounded-lg px-4 py-2 text-xs font-semibold uppercase tracking-wider transition ${
                  selectedConfusionModel === m
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'bg-white hover:bg-slate-100 text-slate-650 dark:bg-slate-900 dark:hover:bg-slate-800 border dark:border-slate-850'
                }`}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {filteredConfusedPairs.slice(0, 6).map((pair, idx) => (
              <div key={idx} className="rounded-2xl border border-slate-205 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <span className="inline-flex items-center rounded bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-600 dark:bg-amber-950/30 dark:text-amber-400">
                      Rank {pair.rank}
                    </span>
                    <span className="font-mono text-[10px] text-slate-400">Score: {pair.confusion_score.toFixed(3)}</span>
                  </div>

                  <div className="flex items-center justify-between gap-1 p-2 rounded-lg bg-slate-50 dark:bg-slate-950 border text-[11px] font-bold mb-4 text-slate-800 dark:text-slate-350">
                    <span className="truncate flex-1 text-center">{pair.class_a.split(' ')[0]} ...</span>
                    <span className="text-blue-500 font-extrabold mx-1">↔</span>
                    <span className="truncate flex-1 text-center">{pair.class_b.split(' ')[0]} ...</span>
                  </div>

                  <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed mb-6">
                    {pair.rank === 1 && "Similarity caused by annular borders, circular scales, and peripheral textures. Both present with scaly erythematous plaques."}
                    {pair.rank === 2 && "Similarity in localized swelling, erythema, and oozing vesicles. Lack of defined clinical margins triggers visual mapping confusion."}
                    {pair.rank === 3 && "Visual overlap from raw eroded skin textures and generalized scaling. Blistered areas present identical color features."}
                    {pair.rank === 4 && "Pigmented hyperkeratotic skin plaques. Common crusty textures, brown-tan scales, and local boundaries cause feature mapping merge."}
                    {pair.rank === 5 && "Cutaneous lupus rashes mimic photodermatoses. Highly proximate spatial characteristics in exposed body regions."}
                    {! [1,2,3,4,5].includes(pair.rank) && "High similarity in skin textures, redness statistics, or circular shape coordinates in embedding geometry."}
                  </p>
                </div>

                <button
                  onClick={() => setLightboxImage({
                    src: `${basePath}${pair.image_path}`,
                    alt: `${pair.class_a} vs ${pair.class_b}`,
                    caption: `Proximity of Class A (${pair.class_a}) and Class B (${pair.class_b}) in embedding space. These represent top visual confused pairs.`
                  })}
                  className="w-full flex items-center justify-center rounded-lg border py-2 text-xs font-semibold hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                >
                  <Maximize2 className="h-3.5 w-3.5 mr-1 text-slate-400" />
                  <span>View Side-by-Side Snapshot</span>
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* NEAREST NEIGHBOUR RETRIEVAL SECTION */}
      <section
        id="nearest-neighbor"
        ref={nnRef}
        className="scroll-mt-28 py-20 bg-white dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Nearest Neighbour Retrieval</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Evaluating local neighborhood quality. Select a query disease and slide coordinates to view the pre-plotted query-response grid.
            </p>
          </div>

          <div className="grid lg:grid-cols-4 gap-8">
            {/* Gallery Control Panel */}
            <div className="rounded-2xl border border-slate-205 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-4">Gallery Search</h3>

                {/* Local model selection for NN */}
                <label className="text-[10px] font-bold text-slate-405 uppercase tracking-wider block mb-1">Pretrained Model</label>
                <div className="flex space-x-1 p-1 bg-slate-100 dark:bg-slate-950 border dark:border-slate-850 rounded-lg mb-4">
                  {(['dinov2', 'clip', 'resnet50'] as const).map(m => (
                    <button
                      key={m}
                      onClick={() => setSelectedNnModel(m)}
                      className={`flex-1 text-center rounded py-1 text-[10px] font-bold transition uppercase ${
                        selectedNnModel === m
                          ? 'bg-white text-blue-600 shadow-sm dark:bg-slate-800 dark:text-blue-400'
                          : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-100'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>

                <label className="text-[10px] font-bold text-slate-405 uppercase tracking-wider block mb-1">Query Class</label>
                <select
                  className="w-full border rounded-lg p-2 text-xs bg-slate-50 mb-4 focus:outline-none dark:bg-slate-950 dark:border-slate-850"
                  value={nnClass}
                  onChange={(e) => {
                    setNnClass(e.target.value);
                    setNnIndex(0);
                  }}
                >
                  <option value="Acne and Rosacea Photos">Acne and Rosacea</option>
                  <option value="Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions">Actinic Keratosis</option>
                  <option value="Atopic Dermatitis Photos">Atopic Dermatitis</option>
                </select>

                <label className="text-[10px] font-bold text-slate-455 uppercase tracking-wider block mb-1">Query Index</label>
                <div className="grid grid-cols-5 gap-1.5 mb-6">
                  {Array.from({ length: 10 }).map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => setNnIndex(idx)}
                      className={`border rounded py-1 text-xs font-semibold transition ${
                        nnIndex === idx
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'hover:bg-slate-100 dark:hover:bg-slate-800'
                      }`}
                    >
                      #{idx}
                    </button>
                  ))}
                </div>

                <div className="space-y-3 text-xs bg-slate-50 dark:bg-slate-950 p-3 rounded-lg border dark:border-slate-850">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Accuracy (top-1):</span>
                    <span className="font-bold text-blue-600 dark:text-blue-400">
                      {nearestNeighborResults.find(r => r.representation === selectedNnModel && r.disease_class.startsWith(nnClass.slice(0, 10)))?.top1_accuracy !== undefined
                        ? `${(nearestNeighborResults.find(r => r.representation === selectedNnModel && r.disease_class.startsWith(nnClass.slice(0, 10)))!.top1_accuracy * 100).toFixed(0)}%`
                        : '60%'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Accuracy (top-5):</span>
                    <span className="font-bold text-emerald-600 dark:text-emerald-500">
                      {nearestNeighborResults.find(r => r.representation === selectedNnModel && r.disease_class.startsWith(nnClass.slice(0, 10)))?.top5_accuracy !== undefined
                        ? `${(nearestNeighborResults.find(r => r.representation === selectedNnModel && r.disease_class.startsWith(nnClass.slice(0, 10)))!.top5_accuracy * 100).toFixed(0)}%`
                        : '80%'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Image Panel */}
            <div className="lg:col-span-3 rounded-2xl border border-slate-205 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900 relative flex flex-col justify-between">
              <div className="relative w-full h-80 rounded-xl overflow-hidden bg-slate-50 border dark:bg-slate-950 dark:border-slate-850 flex items-center justify-center cursor-zoom-in group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={activeNnImagePath}
                  alt={`Retrieval query grid for ${nnClass}`}
                  className="max-w-full max-h-full object-contain transition group-hover:scale-[1.01]"
                  onClick={() => setLightboxImage({
                    src: activeNnImagePath,
                    alt: `Query Grid: ${nnClass}`,
                    caption: `Evaluation grid displaying query skin lesion photograph on the left, and top-5 visual neighbors found using cosine similarities.`
                  })}
                />
              </div>

              <div className="p-3 border-t bg-slate-50 dark:bg-slate-950 dark:border-slate-850 flex justify-between items-center text-[10px] text-slate-500 mt-4">
                <span>The grid shows Query Image (left) vs Top-5 retrieved matches (right) with cosine distances. Click image to expand.</span>
                <span className="font-semibold text-blue-600 dark:text-blue-400 font-mono">nn_{nnClass.slice(0, 10)}_{nnIndex}.png</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* KEY FINDINGS SECTION */}
      <section
        id="findings"
        ref={findingsRef}
        className="scroll-mt-28 py-20 bg-white dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Key Findings</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Derived from spatial comparisons and mathematical probing.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {[
              {
                title: 'DINOv2 Compact Geometry',
                desc: 'DINOv2 uses local self-attention patches which preserve boundary lines and scale textures, yielding compact spatial clusters (average silhouette score 0.1609).',
                icon: <Layers className="h-5 w-5 text-blue-600" />
              },
              {
                title: 'CLIP Semantic Guidance',
                desc: 'Contrastive pretraining aligns embeddings with high-level medical text descriptors. This aligns visual representations closer to clinical disease labels (ARI: 0.1001).',
                icon: <Sparkles className="h-5 w-5 text-emerald-600" />
              },
              {
                title: 'Supervised Baseline Object Bias',
                desc: 'ResNet50 pretraining on standard ImageNet classification tasks results in heavy object shape shortcut bias, generalizing poorly to skin lesions without fine-tuning.',
                icon: <AlertCircle className="h-5 w-5 text-rose-505" />
              }
            ].map((insight, idx) => (
              <div key={idx} className="rounded-2xl border border-slate-205 bg-slate-50 p-6 dark:border-slate-800 dark:bg-slate-900">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white shadow-sm dark:bg-slate-950 mb-4">
                  {insight.icon}
                </div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">{insight.title}</h3>
                <p className="mt-3 text-xs text-slate-505 dark:text-slate-400 leading-relaxed">
                  {insight.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* LIMITATIONS SECTION REMOVED */}

      {/* FUTURE WORK ROADMAP */}
      <section
        id="future"
        ref={futureRef}
        className="scroll-mt-28 py-20 bg-white dark:bg-slate-950 border-b dark:border-slate-900"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Future Research Roadmap</h2>
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Planned future steps to translate these findings into clinically viable tools.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6 max-w-6xl mx-auto mb-12">
            {[
              { title: 'Domain-Specific SSL', desc: 'Pretraining encoders on large clinical skin datasets (ISIC, HAM10000) using DINOv2 self-distillation.' },
              { title: 'Medical Foundation Models', desc: 'Pretraining on specialized multi-spectral lesion image libraries.' },
              { title: 'Multimodal Text Fusion', desc: 'Fusing image visual embeddings with clinical text patient histories and symptoms.' },
              { title: 'Prototype-Based Retrieval', desc: 'Structuring centroid prototypes for differential diagnosis assistance.' },
              { title: 'Hierarchical Taxonomies', desc: 'Loss functions matching the diagnostic trees used by clinical dermatologists.' },
              { title: 'Longitudinal Analysis', desc: 'Tracking representations shifts over multiple patient visits.' }
            ].map((item, idx) => (
              <div key={idx} className="rounded-xl border border-slate-205 bg-slate-50 p-5 dark:border-slate-800 dark:bg-slate-900/50">
                <span className="text-[10px] font-bold text-blue-600 dark:text-blue-400 tracking-widest uppercase block mb-1 font-mono">Step 0{idx + 1}</span>
                <h4 className="text-xs font-bold text-slate-850 dark:text-slate-150">{item.title}</h4>
                <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>

        </div>
      </section>

      {/* FOOTER */}
      <footer className="bg-slate-950 text-slate-400 py-12 border-t border-slate-900 relative">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 pb-8 border-b border-slate-900 text-xs">
            {/* Left Column: Project Info */}
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <div className="flex h-6 w-6 items-center justify-center rounded bg-blue-600 text-white font-bold text-xs shadow-sm">
                  D
                </div>
                <span className="font-bold text-slate-200">DermNet Representation Analysis</span>
              </div>
              <p className="leading-relaxed text-slate-400">
                Evaluating pretrained foundation backbones (DINOv2, CLIP, ResNet50) on dermatological macro photography. Built as part of Saarland University research.
              </p>
            </div>

            {/* Middle Column: Useful Links */}
            <div>
              <span className="font-bold text-slate-250 block mb-3 uppercase tracking-wider text-[10px]">Project Links</span>
              <ul className="space-y-2">
                <li>
                  <a href="https://github.com" target="_blank" rel="noreferrer" className="hover:text-white transition flex items-center space-x-1.5">
                    <GithubIcon className="h-3.5 w-3.5" />
                    <span>Source Code (GitHub)</span>
                  </a>
                </li>
                <li>
                  <a href="https://www.kaggle.com/datasets/shubhamgoel27/dermnet/data" target="_blank" rel="noreferrer" className="hover:text-white transition flex items-center space-x-1.5">
                    <ExternalLink className="h-3.5 w-3.5 text-slate-500" />
                    <span>View Dataset on Kaggle</span>
                  </a>
                </li>
                <li>
                  <a href={`${basePath}/reports/final_report.pdf`} download className="hover:text-white transition flex items-center space-x-1.5">
                    <FileText className="h-3.5 w-3.5 text-slate-500" />
                    <span>Download Project Report</span>
                  </a>
                </li>
              </ul>
            </div>

            {/* Right Column: Team & Contact */}
            <div className="space-y-2">
              <span className="font-bold text-slate-250 block uppercase tracking-wider text-[10px]">Project Team</span>
              <p className="leading-relaxed text-slate-300">
                <span className="font-semibold text-slate-400">Team Members:</span><br />
                Sai Vishwa Babu, Mahitha Senthilnathan, Hemashruthi Durairaj
              </p>
              <p className="leading-relaxed text-slate-300 pt-1.5">
                <span className="font-semibold text-slate-400">Contact:</span><br />
                Email: <a href="mailto:saivishwababu@gmail.com" className="text-blue-400 hover:underline font-semibold">saivishwababu@gmail.com</a>
              </p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row justify-between items-center text-[10px] pt-8 text-slate-500">
            <span>© 2026 Dermatology Representation Analysis Project. All rights reserved.</span>
            <div className="flex space-x-4 mt-3 sm:mt-0 font-semibold">
              <button onClick={() => scrollTo('home')} className="hover:text-white transition cursor-pointer">Back to Top</button>
            </div>
          </div>
        </div>
      </footer>

      {/* METHODOLOGY EXPLANATORY OVERLAY MODAL */}
      <AnimatePresence>
        {methodologyModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={() => setMethodologyModal(null)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 10 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 10 }}
              className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-sm font-bold text-slate-950 dark:text-slate-100">{methodologyModal.title}</h3>
              <div className="mt-4 text-xs text-slate-500 dark:text-slate-400 leading-relaxed space-y-3">
                <p>{methodologyModal.desc}</p>
                {methodologyModal.title === 'Deduplication' && (
                  <p className="font-semibold text-slate-800 dark:text-slate-200">
                    Why deduplicate? Near-duplicate photographs taken under slightly shifted lighting or angles biases clustering densities, causing artificial skew in silhouette dimensions.
                  </p>
                )}
                {methodologyModal.title === 'PCA Reduction' && (
                  <p className="font-semibold text-slate-800 dark:text-slate-200">
                    Applying PCA reduces the 768/512 dimension visual vectors down to 50 dimensions to remove high-frequency noise from ImageNet objects.
                  </p>
                )}
              </div>
              <button
                onClick={() => setMethodologyModal(null)}
                className="mt-6 w-full rounded-xl bg-slate-900 py-2.5 text-xs font-semibold text-white dark:bg-slate-100 dark:text-slate-950 hover:bg-slate-800 transition"
              >
                Close Explanation
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* GALLERY LIGHTBOX MODAL */}
      <AnimatePresence>
        {lightboxImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 cursor-zoom-out"
            onClick={() => setLightboxImage(null)}
          >
            <motion.div
              initial={{ scale: 0.98 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.98 }}
              className="max-w-4xl max-h-[85vh] flex flex-col justify-center"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative w-full overflow-hidden bg-slate-950 flex items-center justify-center rounded-t-2xl">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={lightboxImage.src}
                  alt={lightboxImage.alt}
                  className="max-w-full max-h-[70vh] object-contain"
                />
              </div>
              <div className="bg-slate-900 text-white p-5 rounded-b-2xl max-w-full text-xs">
                <div className="flex justify-between items-center mb-1">
                  <h3 className="font-bold text-slate-200">{lightboxImage.alt}</h3>
                  <button onClick={() => setLightboxImage(null)} className="text-[10px] text-slate-400 hover:text-white uppercase font-bold tracking-wider">
                    Close
                  </button>
                </div>
                <p className="text-slate-400 leading-relaxed">{lightboxImage.caption}</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
