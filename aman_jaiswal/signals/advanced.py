# =============================================================================
# advanced.py — Tier 3: LaTeX complexity, OCR noise, physics domain, symbol density
# =============================================================================
import re
from config import PHYSICS_KEYWORDS, LATEX_TOKENS, OCR_NOISE_PATTERNS


def compute_latex_complexity(text: str) -> int:
    """Count distinct LaTeX structural tokens present."""
    return sum(1 for t in LATEX_TOKENS if re.search(t, text))


def detect_physics_domain(text: str) -> int:
    """Count physics domain keyword matches."""
    text_lower = text.lower()
    return sum(1 for kw in PHYSICS_KEYWORDS if kw in text_lower)


def compute_ocr_noise(text: str) -> float:
    """Compute OCR noise score — noise patterns suggest lost visual content."""
    noise = sum(len(re.findall(p, text)) for p in OCR_NOISE_PATTERNS)
    return noise / (len(text) + 1)


def compute_symbol_density(text: str) -> float:
    """Compute density of mathematical symbols in text."""
    symbols = re.findall(r'[=<>∑∫∆→←≈πθλμΩ±×÷∂∇]', text)
    return len(symbols) / (len(text) + 1)
