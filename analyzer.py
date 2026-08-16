"""
analyzer.py
Extracts low-level image-forensics signals using OpenCV + NumPy only.
No external models, no internet calls, no GPU required.
"""

import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image file. It may be corrupted or in an unsupported format.")
    return img


def compute_sharpness(gray):
    """Laplacian variance - a classic focus/sharpness measure.
    Low values => blurry / low quality image."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_edge_density(gray):
    """Fraction of pixels detected as edges by Canny."""
    edges = cv2.Canny(gray, 100, 200)
    density = float(np.sum(edges > 0)) / edges.size
    return density


def compute_texture_stats(gray):
    """Average local standard deviation over small patches.
    Flat / plastic-looking surfaces (renders, mockups) tend to have low texture variance."""
    h, w = gray.shape
    patch = 16
    stds = []
    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            block = gray[y:y + patch, x:x + patch]
            stds.append(block.std())
    if not stds:
        return 0.0, 0.0
    stds = np.array(stds)
    return float(stds.mean()), float(stds.std())


def compute_color_stats(img):
    """Unique-color ratio (posterization / flat-color detector) and channel variation."""
    small = cv2.resize(img, (64, 64))
    pixels = small.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    total_pixels = pixels.shape[0]
    unique_ratio = unique_colors / total_pixels

    b, g, r = cv2.split(img)
    color_std = float(np.mean([b.std(), g.std(), r.std()]))
    return unique_ratio, color_std


def compute_compression_blockiness(gray):
    """Approximates JPEG blockiness by measuring pixel discontinuity
    at 8x8 grid boundaries versus interior pixels. Repeated re-saving
    / heavy re-compression (common with scraped marketplace images)
    increases this value."""
    gray = gray.astype(np.float64)
    h, w = gray.shape
    diffs = []

    for x in range(8, w - 1, 8):
        diffs.append(np.mean(np.abs(gray[:, x] - gray[:, x - 1])))
    for y in range(8, h - 1, 8):
        diffs.append(np.mean(np.abs(gray[y, :] - gray[y - 1, :])))

    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def compute_noise_level(gray):
    """Residual energy between the image and a median-filtered version.
    Very low noise can indicate synthetic/rendered images; very high
    noise can indicate poor-quality re-shoots or heavy upscaling."""
    median = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, median)
    return float(residual.mean())


def analyze_image(path):
    img = load_image(path)

    # Normalize very large images for consistent, fast processing
    h0, w0 = img.shape[:2]
    max_dim = 1200
    if max(h0, w0) > max_dim:
        scale = max_dim / max(h0, w0)
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = compute_sharpness(gray)
    edge_density = compute_edge_density(gray)
    texture_mean, texture_std = compute_texture_stats(gray)
    unique_ratio, color_std = compute_color_stats(img)
    blockiness = compute_compression_blockiness(gray)
    noise = compute_noise_level(gray)

    h, w = gray.shape

    return {
        "sharpness": sharpness,
        "edge_density": edge_density,
        "texture_mean": texture_mean,
        "texture_std": texture_std,
        "unique_ratio": unique_ratio,
        "color_std": color_std,
        "blockiness": blockiness,
        "noise": noise,
        "width": int(w0),
        "height": int(h0),
    }
