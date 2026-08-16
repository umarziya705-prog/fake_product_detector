"""
review_analyzer.py
Analyzes listing-level signals the way a shopper actually would:
review text authenticity, rating-pattern sanity, and price anomaly.

Everything here runs locally:
- The text classifier is a tiny TF-IDF + Logistic Regression model
  trained at startup on a small bundled CSV (data/review_training_samples.csv).
  No pretrained weights are downloaded, no internet access required.
- Rating and price checks are plain heuristics.

All three signals are OPTIONAL - if the caller doesn't provide them,
the corresponding analyze_* function returns None and scoring.py simply
excludes that signal from the combined score.
"""

import os
import re
import csv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "review_training_samples.csv")

_model = None
_vectorizer = None

GENERIC_PHRASES = [
    "good product", "nice product", "value for money", "as described",
    "highly recommend", "five stars", "best product ever", "will buy again",
    "super fast delivery", "totally worth it", "must buy", "genuine product",
]


def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


def normalize(value, low, high, invert=False):
    """Map value in [low, high] to a 0-100 risk contribution."""
    if high == low:
        return 50.0
    pct = (value - low) / (high - low)
    pct = max(0.0, min(1.0, pct))
    score = pct * 100.0
    if invert:
        score = 100.0 - score
    return score


def _load_training_data():
    texts, labels = [], []
    with open(TRAINING_DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def _get_model():
    """Lazily trains the tiny local classifier once per process."""
    global _model, _vectorizer
    if _model is None:
        texts, labels = _load_training_data()
        _vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
        X = _vectorizer.fit_transform(texts)
        _model = LogisticRegression(max_iter=1000)
        _model.fit(X, labels)
    return _model, _vectorizer


def split_reviews(raw_text):
    """Splits pasted review text into individual reviews (one per line/blank-line block)."""
    if not raw_text:
        return []
    parts = re.split(r"\n+", raw_text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_reviews_from_csv(file_stream):
    """Reads an uploaded CSV and pulls review text from a likely column
    (review / text / comment / body), case-insensitive."""
    text_stream = (line.decode("utf-8", errors="ignore") for line in file_stream)
    reader = csv.DictReader(text_stream)
    if not reader.fieldnames:
        return []

    candidates = ["review", "review_text", "text", "comment", "body", "content"]
    col = None
    for c in candidates:
        for field in reader.fieldnames:
            if field.strip().lower() == c:
                col = field
                break
        if col:
            break

    if not col:
        return []

    reviews = []
    for row in reader:
        val = (row.get(col) or "").strip()
        if val:
            reviews.append(val)
    return reviews


def analyze_review_text(reviews):
    """reviews: list[str]. Returns a risk score (0-100, higher = more likely
    fake/templated reviews) plus supporting stats, or None if no reviews given."""
    reviews = [r for r in reviews if r and r.strip()]
    if not reviews:
        return None

    model, vectorizer = _get_model()
    X = vectorizer.transform(reviews)
    probs = model.predict_proba(X)[:, 1]  # P(suspicious/templated)
    avg_suspicious = float(np.mean(probs)) * 100

    lowered = [r.lower() for r in reviews]
    generic_hits = sum(1 for r in lowered for g in GENERIC_PHRASES if g in r)
    generic_ratio = generic_hits / max(len(reviews), 1)

    avg_words = float(np.mean([len(r.split()) for r in reviews]))
    # very short reviews (<=4 words) are a classic low-effort/fake-review signal
    short_penalty = clamp(100 - avg_words * 8) if avg_words < 12 else 0

    unique_ratio = len(set(lowered)) / len(reviews)
    duplicate_risk = clamp((1 - unique_ratio) * 100)

    score = clamp(
        0.5 * avg_suspicious
        + 0.2 * (generic_ratio * 100)
        + 0.15 * short_penalty
        + 0.15 * duplicate_risk
    )

    return {
        "score": round(score),
        "review_count": len(reviews),
        "avg_words": round(avg_words, 1),
        "duplicate_pct": round(duplicate_risk),
    }


def analyze_rating_pattern(avg_rating, num_ratings):
    """avg_rating: float 0-5, num_ratings: int. Returns None if either is missing."""
    if avg_rating is None or num_ratings is None:
        return None

    risk = 0
    if avg_rating >= 4.8 and num_ratings < 20:
        risk += 40
    elif avg_rating >= 4.6 and num_ratings < 10:
        risk += 25

    if num_ratings < 5:
        risk += 25
    elif num_ratings < 15:
        risk += 10

    if avg_rating >= 4.95:
        risk += 15

    return {
        "score": round(clamp(risk)),
        "avg_rating": avg_rating,
        "num_ratings": num_ratings,
    }


def analyze_price(price, expected_price):
    """price: what the listing charges. expected_price: typical/market price
    the user believes is normal for this product. Returns None if either
    is missing or invalid."""
    if not price or not expected_price or expected_price <= 0:
        return None

    ratio = price / expected_price

    if ratio >= 0.7:
        risk = 5
    elif ratio >= 0.5:
        risk = 40
    elif ratio >= 0.3:
        risk = 70
    else:
        risk = 90

    return {
        "score": round(clamp(risk)),
        "price": price,
        "expected_price": expected_price,
        "ratio": round(ratio, 2),
    }


def analyze_review_velocity(timestamps):
    """timestamps: list of ISO date strings ('YYYY-MM-DD'). Flags listings
    where a suspiciously large share of reviews landed in a short burst -
    a classic sign of a paid review campaign rather than organic sales.
    Returns None if fewer than 3 dated reviews are available."""
    from datetime import datetime, timedelta

    if not timestamps or len(timestamps) < 3:
        return None

    dates = []
    for t in timestamps:
        try:
            dates.append(datetime.strptime(t[:10], "%Y-%m-%d"))
        except (ValueError, TypeError):
            continue

    if len(dates) < 3:
        return None

    dates.sort()
    span_days = max((dates[-1] - dates[0]).days, 1)

    # Busiest 7-day rolling window
    window = timedelta(days=7)
    max_window_count = 0
    for d in dates:
        count = sum(1 for x in dates if 0 <= (x - d).days <= 7)
        max_window_count = max(max_window_count, count)
    burst_ratio = max_window_count / len(dates)

    reviews_per_day = len(dates) / span_days

    burst_risk = normalize(burst_ratio, 0.3, 0.9)
    density_risk = normalize(reviews_per_day, 0.05, 1.5)

    score = clamp(0.6 * burst_risk + 0.4 * density_risk)

    return {
        "score": round(score),
        "review_count": len(dates),
        "span_days": span_days,
        "burst_pct": round(burst_ratio * 100),
    }


def analyze_seller_tenure(months_active):
    """months_active: how long the seller account has been active on the
    platform. Longer tenure -> lower risk. Returns None if not provided."""
    if months_active is None:
        return None

    risk = normalize(months_active, 0, 24, invert=True)

    return {
        "score": round(clamp(risk)),
        "months_active": months_active,
    }
