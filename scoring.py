"""
scoring.py
Turns raw signals into a 0-100 Counterfeit Risk Score.

Image-forensics signals (sharpness, edges, texture, compression, and a
small local logistic "AI/ML prediction" head) are ALWAYS available, since
an image is required.

Review-text authenticity, rating-pattern, and price-anomaly signals are
OPTIONAL - each is only included if the user supplied that data. Weights
are re-normalized across whichever signals are actually present, so the
image-only flow still produces a complete, honest score.
"""

import math


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


# ---------------------------------------------------------------------
# Image sub-scores
# ---------------------------------------------------------------------

def score_image_quality(features):
    sharpness_risk = normalize(features["sharpness"], 0, 500, invert=True)
    noise_risk = normalize(features["noise"], 0, 15)
    return clamp(0.6 * sharpness_risk + 0.4 * noise_risk)


def score_visual_consistency(features):
    ed = features["edge_density"]
    edge_risk = normalize(abs(ed - 0.10), 0, 0.15)
    color_risk = normalize(features["unique_ratio"], 0.05, 0.6, invert=True)
    return clamp(0.5 * edge_risk + 0.5 * color_risk)


def score_texture(features):
    return clamp(normalize(features["texture_std"], 5, 40, invert=True))


def score_compression(features):
    return clamp(normalize(features["blockiness"], 1, 12))


def score_ai_ml_prediction(sub_scores):
    x = (
        0.25 * sub_scores["image_quality"]
        + 0.30 * sub_scores["visual_consistency"]
        + 0.20 * sub_scores["texture"]
        + 0.25 * sub_scores["compression"]
    )
    z = (x - 50) / 12.0
    logistic = 1 / (1 + math.exp(-z))
    return clamp(logistic * 100)


def compute_image_score(features):
    sub_scores = {
        "image_quality": score_image_quality(features),
        "visual_consistency": score_visual_consistency(features),
        "texture": score_texture(features),
        "compression": score_compression(features),
    }
    sub_scores["ai_ml_prediction"] = score_ai_ml_prediction(sub_scores)

    weights = {
        "image_quality": 0.20,
        "visual_consistency": 0.25,
        "texture": 0.15,
        "compression": 0.15,
        "ai_ml_prediction": 0.25,
    }
    score = sum(sub_scores[k] * weights[k] for k in weights)

    return {
        "score": clamp(score),
        "sub_scores": {k: round(v) for k, v in sub_scores.items()},
    }


# ---------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------

def _image_reasons(sub_scores):
    reasons = []
    if sub_scores["image_quality"] > 60:
        reasons.append(
            "Sharpness and noise levels fall outside the range typically seen in authentic product photography."
        )
    if sub_scores["visual_consistency"] > 60:
        reasons.append(
            "Edge patterns and color distribution show inconsistencies compared to genuine listing photos."
        )
    if sub_scores["texture"] > 60:
        reasons.append(
            "Surface texture looks unusually flat or uniform, which can point to a rendered, mocked-up, or heavily retouched image."
        )
    if sub_scores["compression"] > 60:
        reasons.append(
            "Strong compression artifacts were detected, suggesting the image has been re-saved or re-uploaded multiple times."
        )
    if sub_scores["ai_ml_prediction"] > 60:
        reasons.append(
            "The combined image signal pattern is similar to images the local model associates with higher-risk listings."
        )
    return reasons


def _review_reasons(review_result):
    reasons = []
    if review_result["score"] > 60:
        reasons.append(
            "Pasted reviews read as generic or templated, a pattern common in fake/incentivized reviews."
        )
    if review_result["duplicate_pct"] > 40:
        reasons.append(
            "Several of the pasted reviews are near-duplicates of each other."
        )
    if review_result["avg_words"] < 8:
        reasons.append(
            "Reviews are unusually short on average, with little product-specific detail."
        )
    return reasons


def _rating_reasons(rating_result):
    reasons = []
    if rating_result["score"] > 50:
        reasons.append(
            "The rating ({}\u2605 from {} ratings) shows a pattern - very high average with a low "
            "rating count - often seen on new or manipulated listings.".format(
                rating_result["avg_rating"], rating_result["num_ratings"]
            )
        )
    return reasons


def _price_reasons(price_result):
    reasons = []
    if price_result["score"] > 50:
        reasons.append(
            "Listed price is about {}% of the typical market price you entered, a common red flag "
            "for counterfeit or misrepresented listings.".format(int(price_result["ratio"] * 100))
        )
    return reasons


def _velocity_reasons(velocity_result):
    reasons = []
    if velocity_result["score"] > 55:
        reasons.append(
            "{}% of reviews landed within the same 7-day window across a {}-day history, a pattern "
            "consistent with a paid review campaign rather than organic purchases.".format(
                velocity_result["burst_pct"], velocity_result["span_days"]
            )
        )
    return reasons


def _seller_reasons(seller_result):
    reasons = []
    months = seller_result["months_active"]
    if seller_result["score"] > 55:
        reasons.append(
            "The seller account has only been active for about {} month(s), less tenure than "
            "typically seen from established, trustworthy sellers.".format(months)
        )
    return reasons


def _image_match_reasons(image_match_result):
    reasons = []
    if image_match_result["score"] > 50:
        reasons.append(
            "Buyer-uploaded photos show only about {}% visual similarity to the listing image, "
            "which can mean the product actually received doesn't match what was advertised.".format(
                image_match_result["avg_similarity_pct"]
            )
        )
    return reasons


# ---------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------

def compute_combined_score(
    features,
    review_result=None,
    rating_result=None,
    price_result=None,
    velocity_result=None,
    seller_result=None,
    image_match_result=None,
):
    image_result = compute_image_score(features)

    components = {"image": (image_result["score"], 0.20)}
    signals_used = ["Image Analysis"]

    if review_result:
        components["reviews"] = (review_result["score"], 0.15)
        signals_used.append("Review Text")
    if rating_result:
        components["ratings"] = (rating_result["score"], 0.10)
        signals_used.append("Rating Pattern")
    if price_result:
        components["price"] = (price_result["score"], 0.15)
        signals_used.append("Price Comparison")
    if velocity_result:
        components["velocity"] = (velocity_result["score"], 0.15)
        signals_used.append("Review Timing")
    if seller_result:
        components["seller"] = (seller_result["score"], 0.10)
        signals_used.append("Seller Tenure")
    if image_match_result:
        components["image_match"] = (image_match_result["score"], 0.15)
        signals_used.append("Buyer Photo Match")

    # If only the image was provided, it simply carries 100% of the weight.
    total_weight = sum(w for _, w in components.values())
    final = sum(score * (w / total_weight) for score, w in components.values())
    final = round(clamp(final))

    if final >= 70:
        level = "HIGH"
    elif final >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    explanation = _image_reasons(image_result["sub_scores"])
    if review_result:
        explanation += _review_reasons(review_result)
    if rating_result:
        explanation += _rating_reasons(rating_result)
    if price_result:
        explanation += _price_reasons(price_result)
    if velocity_result:
        explanation += _velocity_reasons(velocity_result)
    if seller_result:
        explanation += _seller_reasons(seller_result)
    if image_match_result:
        explanation += _image_match_reasons(image_match_result)

    if len(signals_used) <= 2:
        explanation.append(
            "Limited listing data was available for this assessment - the score leans heavily on "
            "image analysis alone and should be treated as lower-confidence."
        )

    if not explanation:
        explanation.append("No significant anomalies were detected across the signals analyzed.")

    return {
        "score": final,
        "level": level,
        "sub_scores": image_result["sub_scores"],
        "review_analysis": review_result,
        "rating_analysis": rating_result,
        "price_analysis": price_result,
        "velocity_analysis": velocity_result,
        "seller_analysis": seller_result,
        "image_match_analysis": image_match_result,
        "signals_used": signals_used,
        "explanation": explanation,
    }
