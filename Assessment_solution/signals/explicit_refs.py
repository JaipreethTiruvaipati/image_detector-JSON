# =============================================================================
# explicit_refs.py — Tier 2: explicit visual reference language detection
# =============================================================================
import re
from config import EXPLICIT_REF_PATTERNS, GRAPH_AXIS_PATTERNS, CIRCUIT_PATTERNS


def detect_explicit_refs(text: str) -> int:
    """Detect explicit figure/diagram reference phrases in text."""
    return sum(1 for p in EXPLICIT_REF_PATTERNS if re.search(p, text, re.IGNORECASE))


def detect_graph_axis_refs(text: str) -> int:
    """Detect graph and axis references."""
    return sum(1 for p in GRAPH_AXIS_PATTERNS if re.search(p, text, re.IGNORECASE))


def detect_circuit_components(text: str) -> int:
    """Detect labeled circuit component references."""
    return sum(1 for p in CIRCUIT_PATTERNS if re.search(p, text, re.IGNORECASE))
