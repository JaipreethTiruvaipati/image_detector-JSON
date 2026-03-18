# =============================================================================
# scoring.py — 3-tier weighted scoring engine with confidence labels
# =============================================================================
from config import WEIGHTS, THRESHOLD_HIGH, THRESHOLD_MEDIUM, THRESHOLD_LOW


def compute_score(row: dict) -> float:
    score = 0.0

    # ── Tier 1: Direct Evidence ───────────────────────────────────────────────
    # Only fire if URL looks like an actual diagram file (has image extension)
    if row.get("diagram_url_count", 0) > 0:
        score += WEIGHTS["image_url_found"]

    # Only fire if image_count > 1 (more than just the base OCR scan)
    if row.get("image_count_meta", 0) > 1:
        score += WEIGHTS["image_count_meta"]

    if row.get("has_dead_images", False):
        score += WEIGHTS["has_dead_images"]

    # ── Tier 2: Strong Textual Evidence ──────────────────────────────────────
    if row.get("explicit_ref", 0) > 0:
        score += WEIGHTS["explicit_ref"]

    if row.get("geometry_entity", 0) > 0:
        score += WEIGHTS["geometry_entity"]

    if row.get("graph_axis_ref", 0) > 0:
        score += WEIGHTS["graph_axis_ref"]

    if row.get("circuit_component", 0) > 0:
        score += WEIGHTS["circuit_component"]

    # ── Tier 3: Contextual Signals ────────────────────────────────────────────
    if row.get("physics_domain", 0) > 1:
        score += WEIGHTS["physics_domain"]

    if row.get("latex_complexity", 0) > 2:
        score += WEIGHTS["latex_complexity"]

    if row.get("ocr_noise", 0.0) > 0.01:
        score += WEIGHTS["ocr_noise"]

    if row.get("symbol_density", 0.0) > 0.02:
        score += WEIGHTS["symbol_density"]

    return score


def assign_confidence(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    elif score >= THRESHOLD_MEDIUM:
        return "MEDIUM"
    elif score >= THRESHOLD_LOW:
        return "LOW"
    else:
        return "NONE"


def build_reason(row: dict) -> str:
    reasons = []
    if row.get("diagram_url_count", 0) > 0:
        reasons.append(f"diagram image URL found ({row['diagram_url_count']} file(s))")
    if row.get("image_count_meta", 0) > 1:
        reasons.append(f"image_count = {row['image_count_meta']} (multiple images)")
    if row.get("has_dead_images", False):
        reasons.append("has_dead_images=True")
    if row.get("explicit_ref", 0) > 0:
        reasons.append("explicit figure/diagram reference in text")
    if row.get("geometry_entity", 0) > 0:
        reasons.append("named geometric entity (e.g. △ABC, ∠PQR)")
    if row.get("graph_axis_ref", 0) > 0:
        reasons.append("graph/axis reference detected")
    if row.get("circuit_component", 0) > 0:
        reasons.append("circuit component reference detected")
    if row.get("physics_domain", 0) > 1:
        reasons.append(f"physics domain keywords ({row['physics_domain']} matches)")
    if row.get("latex_complexity", 0) > 2:
        reasons.append(f"LaTeX complexity score = {row['latex_complexity']}")
    return " | ".join(reasons) if reasons else "weak contextual signals only"
