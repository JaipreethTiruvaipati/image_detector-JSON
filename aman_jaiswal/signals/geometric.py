# =============================================================================
# geometric.py — Tier 2: named geometric entity detection
# =============================================================================
import re
from config import GEOMETRY_ENTITY_PATTERNS


def detect_geometry_entities(text: str) -> int:
    """Detect named geometric entities like △ABC, ∠PQR, point P, line AB."""
    return sum(1 for p in GEOMETRY_ENTITY_PATTERNS if re.search(p, text, re.IGNORECASE))
