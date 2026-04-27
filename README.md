# EdgeSR: High-Performance CPU Video Super-Resolution

![C++20](https://img.shields.io/badge/C%2B%2B-20%2B-blue.svg)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Optimized-purple.svg)
![CMake](https://img.shields.io/badge/CMake-Build-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Zero-GPU, Edge-Ready AI Video Enhancement.** > A pure C++ pipeline demonstrating industrial-grade inference optimization for standard Intel CPUs.

## 🚀 The Vision

Deploying heavy Deep Learning models usually requires expensive GPU infrastructure, creating a massive barrier to entry for SMBs and edge-device applications. **EdgeSR** bridges this gap.

This repository demonstrates how to architect a highly optimized, bare-metal C++ pipeline that leverages OpenVINO and hardware-specific SIMD (AVX2/AVX-512) instructions to perform real-time (or near real-time) Video Super-Resolution entirely on a standard CPU.

It is built for performance, memory safety, and minimal deployment overhead.

---

## 🧠 Architecture & Algorithmic Optimizations

Processing 3-channel (RGB) high-resolution video through a Convolutional Neural Network on a CPU is computationally prohibitive. This engine achieves its speed by combining signal processing principles with AI inference techniques:

### 1. Biological Color Space Splitting (YCbCr)
The human eye is incredibly sensitive to luminance (sharpness, edges, contrast) but highly forgiving regarding chrominance (color).
Instead of processing all three RGB channels through the heavy AI model:
* The input frame is converted to the **YCbCr** color space.
* The **Y (Luma)** channel is isolated and fed into the Neural Network for high-fidelity structural upscaling.
* The **Cb & Cr (Chroma)** channels are upscaled in parallel using low-cost bicubic interpolation.
* **Result:** A 66% reduction in tensor matrix multiplications with zero perceived loss in visual quality.

### 2. Tiling & Dynamic Batching
The ONNX model expects a strict `[?, 1, 224, 224]` input tensor. To process an arbitrary resolution (e.g., 360p or 720p) video:
* The Luma plane is padded and mathematically sliced into $224 \times 224$ tiles.
* Tiles are packed into a single dynamic batch tensor (e.g., `[45, 1, 224, 224]`).
* OpenVINO processes the entire batch concurrently, maximizing CPU core utilization and cache hits.
* The output tiles ($672 \times 672$) are mapped and stitched back into a cohesive high-resolution frame.

---

## 📊 Benchmarks

*Hardware: Intel Core i5 (X Cores, X GHz)*
*Model: ESPCN (Sub-Pixel CNN) x3 Upscaling*

| Operation | Resolution (In -> Out) | Inference Time (Avg per Frame) |
| :--- | :--- | :--- |
| **Single Tile** | 224x224 -> 672x672 | ~25 ms |
| **Full Frame** | 640x360 -> 1920x1080 | [X] ms / [Y] FPS |

*(Note: Replace [X] and [Y] with your actual metrics after running the full video loop).*

---

## 🛠️ Build Instructions

This project uses `vcpkg` for dependency management and modern `CMake` for cross-platform building.

### Prerequisites
* C++20 compatible compiler (GCC 11+, Clang, or MSVC)
* CMake 3.20+
* [vcpkg](https://github.com/microsoft/vcpkg) installed and bootstrapped

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<eyals99-lang>/EdgeSR-OpenVINO.git
cd EdgeSR-OpenVINO