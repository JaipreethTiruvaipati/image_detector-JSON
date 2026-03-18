# =============================================================================
# config.py — Central configuration for all signal weights, thresholds, patterns
# =============================================================================

# ── Scoring Thresholds ────────────────────────────────────────────────────────
THRESHOLD_HIGH       = 6.0   # HIGH confidence: almost certainly had an image
THRESHOLD_MEDIUM     = 3.5   # MEDIUM confidence: likely had an image
THRESHOLD_LOW        = 1.5   # LOW confidence: possibly had an image

# Final output threshold (questions scoring >= this are predicted as image questions)
PREDICTION_THRESHOLD = 3.5

# ── Signal Weights ────────────────────────────────────────────────────────────
WEIGHTS = {
    # Tier 1 — Direct Evidence (near-certain)
    "image_url_found"    : 8.0,   # Actual S3 URL found in data
    "image_count_meta"   : 7.0,   # image_count > 0 in metadata
    "has_dead_images"    : 6.0,   # has_dead_images flag is True

    # Tier 2 — Strong Textual Evidence
    "explicit_ref"       : 5.0,   # "shown in figure", "refer to diagram"
    "geometry_entity"    : 4.0,   # △ABC, ∠PQR, named geometric entities
    "graph_axis_ref"     : 3.5,   # "x-axis", "y-axis", "graph of"
    "circuit_component"  : 3.5,   # labeled circuit components

    # Tier 3 — Contextual Signals
    "physics_domain"     : 2.0,   # force, velocity, circuit keywords
    "latex_complexity"   : 1.5,   # \frac, \int, \begin{array}
    "ocr_noise"          : 1.0,   # broken OCR patterns
    "symbol_density"     : 1.0,   # high math symbol density
}

# ── Explicit Visual Reference Phrases ─────────────────────────────────────────
EXPLICIT_REF_PATTERNS = [
    r'\bshown\s+in\s+(the\s+)?(figure|diagram|graph|circuit|table|image|picture|illustration)\b',
    r'\brefer\s+to\s+(the\s+)?(figure|diagram|graph|circuit|table|image)\b',
    r'\bas\s+shown\b',
    r'\bas\s+depicted\b',
    r'\bfrom\s+the\s+(figure|diagram|graph|circuit|table)\b',
    r'\b(figure|diagram|graph|circuit|table)\s+shows\b',
    r'\bin\s+the\s+(figure|diagram|graph|circuit|table|image)\b',
    r'\bgiven\s+in\s+(the\s+)?(figure|diagram|graph|circuit|table)\b',
    r'\bsee\s+(the\s+)?(figure|diagram|graph|circuit|table|image)\b',
    r'\bfig\s*[\.\:]?\s*\d+',
    r'\bthe\s+figure\s+(above|below|given|shown)\b',
    r'\billustrated\s+in\b',
    r'\bdepicted\s+in\b',
    r'\bshown\s+above\b',
    r'\bshown\s+below\b',
]

# ── Named Geometric Entity Patterns ───────────────────────────────────────────
GEOMETRY_ENTITY_PATTERNS = [
    r'[△▲∆]\s*[A-Z]{3}',                        # △ABC
    r'\bangle\s+[A-Z]{2,3}\b',                  # angle PQR
    r'\btriangle\s+[A-Z]{3}\b',                 # triangle ABC
    r'\bline\s+[A-Z]{2}\b',                     # line AB
    r'\bsegment\s+[A-Z]{2}\b',                  # segment AB
    r'\bpoint\s+[A-Z]\b',                       # point P
    r'\\overrightarrow\s*\{[A-Z]{2}\}',         # \overrightarrow{AB}
    r'\b[A-Z]\s*\(\s*\d+\s*,\s*\d+\s*\)',       # A(3, 4) coordinates
    r'\(\s*\d+\s*,\s*\d+\s*\)',                 # (3, 4)
    r'\bcircle\s+with\s+(centre|center|radius)\b',
    r'\bdiameter\s+[A-Z]{2}\b',
    r'\border\s+[A-Z]{2}\b',
]

# ── Graph and Axis Reference Patterns ─────────────────────────────────────────
GRAPH_AXIS_PATTERNS = [
    r'\bx[\s\-]axis\b',
    r'\by[\s\-]axis\b',
    r'\bz[\s\-]axis\b',
    r'\bgraph\s+of\b',
    r'\bplot\s+(of|shows)\b',
    r'\bcurve\s+(shows|given|above|below)\b',
    r'\bslope\s+of\s+(the\s+)?(graph|line|curve)\b',
    r'\barea\s+under\s+(the\s+)?(graph|curve)\b',
    r'\bstraight\s+line\s+graph\b',
    r'\bvelocity[\s\-]time\s+graph\b',
    r'\bposition[\s\-]time\s+graph\b',
    r'\bdisplacement[\s\-]time\b',
]

# ── Circuit Component Patterns ─────────────────────────────────────────────────
CIRCUIT_PATTERNS = [
    r'\bresistor\s+[A-Z0-9]+\b',
    r'\bbattery\s+(of|with|emf)\b',
    r'\b(emf|resistance)\s+[A-Z]\d?\b',
    r'\bswitch\s+[A-Z]\b',
    r'\bcapacitor\s+[A-Z0-9]+\b',
    r'\bwire\s+[A-Z]{2}\b',
    r'\bammeter\b',
    r'\bvoltmeter\b',
    r'\bjunction\s+[A-Z]\b',
    r'\bnode\s+[A-Z]\b',
]

# ── Physics Domain Keywords ────────────────────────────────────────────────────
PHYSICS_KEYWORDS = [
    "force", "velocity", "acceleration", "momentum", "torque",
    "inclined plane", "friction", "projectile", "tension",
    "magnetic field", "electric field", "current", "voltage",
    "lens", "mirror", "refraction", "reflection", "prism",
    "spring", "pendulum", "pulley", "block", "wedge",
    "circuit", "resistor", "capacitor", "inductor",
    "wave", "amplitude", "frequency", "wavelength",
]

# ── LaTeX Complexity Tokens ────────────────────────────────────────────────────
LATEX_TOKENS = [
    r'\\frac', r'\\int', r'\\sum', r'\\sqrt', r'\\vec',
    r'\\begin\{', r'\\end\{', r'\\overrightarrow', r'\\overleftarrow',
    r'\\hat', r'\\bar', r'\\dot', r'\\ddot',
    r'\\oint', r'\\prod', r'\\lim',
]

# ── OCR Noise Patterns ─────────────────────────────────────────────────────────
OCR_NOISE_PATTERNS = [
    r'[^\x00-\x7F]{3,}',          # 3+ consecutive non-ASCII chars
    r'\b[a-z]\b(\s+\b[a-z]\b){3,}', # many isolated single letters
    r'_{3,}',                      # long underscore runs (blank fields)
    r'\?{2,}',                     # multiple question marks
    r'\.{4,}',                     # long dot runs
]

# ── S3 URL Pattern ─────────────────────────────────────────────────────────────
S3_URL_PATTERN = r'https?://[a-z0-9\-]+\.s3[^\s"\'<>]+'
