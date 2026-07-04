import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Dermatological Image Representation Analysis | Master's Research",
  description: "Objective evaluation of pretrained self-supervised visual features (DINOv2, CLIP, ResNet50) for dermatological disease categorization, visual clustering, and content-based retrieval.",
  keywords: ["Dermatology AI", "Self-Supervised Learning", "DINOv2", "CLIP", "ResNet50", "Representation Analysis", "Medical Imaging", "Saarland University"],
  authors: [{ name: "Saarland University Master's Student" }],
  openGraph: {
    title: "Dermatological Image Representation Analysis | DINOv2, CLIP & ResNet50 Probing",
    description: "Evaluating how pretrained visual foundation models encode medical concepts without supervised fine-tuning.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} scroll-smooth`}>
      <body className="font-sans antialiased bg-slate-50 text-slate-900 transition-colors duration-300 dark:bg-slate-950 dark:text-slate-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
