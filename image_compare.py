"""
image_compare.py
Compares the product listing image against buyer/reviewer-uploaded photos.
A large mismatch (different color palette, no matching visual features)
is a meaningful counterfeit signal: it can mean the buyer received a
different item than what was advertised.

Everything here is local OpenCV - no external models, no internet.
"""

import cv2
import numpy as np


def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


def normalize(value, low, high, invert=False):
    if high == low:
        return 50.0
    pct = (value - low) / (high - low)
    pct = max(0.0, min(1.0, pct))
    score = pct * 100.0
    if invert:
        score = 100.0 - score
    return score


def _load(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    img = cv2.imread(path_or_array)
    return img


def _pair_similarity(img_a, img_b):
    """Returns a 0-1 similarity score between two images using color
    histogram correlation + ORB keypoint feature matching."""
    size = (256, 256)
    a = cv2.resize(img_a, size)
    b = cv2.resize(img_b, size)

    # ---- Color histogram similarity ----
    hist_a = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_sim = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    hist_sim = max(0.0, float(hist_sim))

    # ---- ORB keypoint feature matching ----
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)

    feature_sim = 0.0
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good = [m for m in matches if m.distance < 60]
        feature_sim = len(good) / max(min(len(kp1), len(kp2)), 1)
        feature_sim = min(feature_sim, 1.0)

    return 0.5 * hist_sim + 0.5 * feature_sim


def compare_product_images(listing_image, buyer_image_paths):
    """listing_image: path or already-loaded ndarray for the main listing photo.
    buyer_image_paths: list of file paths to buyer/reviewer-uploaded photos.
    Returns None if no usable buyer images were provided."""
    listing = _load(listing_image)
    if listing is None or not buyer_image_paths:
        return None

    similarities = []
    for p in buyer_image_paths:
        buyer_img = _load(p)
        if buyer_img is None:
            continue
        similarities.append(_pair_similarity(listing, buyer_img))

    if not similarities:
        return None

    avg_similarity = float(np.mean(similarities))  # 0-1
    # High similarity -> low risk. Below ~0.35 similarity is a strong mismatch signal.
    risk = normalize(avg_similarity, 0.25, 0.75, invert=True)

    return {
        "score": round(clamp(risk)),
        "avg_similarity_pct": round(avg_similarity * 100),
        "buyer_photo_count": len(similarities),
    }
