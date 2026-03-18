import pandas as pd
import re
from typing import List, Dict

# ==============================
# 1. LOAD DATA
# ==============================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df


# ==============================
# 2. TEXT EXTRACTION
# ==============================
def extract_text_fields(ocr_fields: Dict) -> str:
    texts = []

    if isinstance(ocr_fields, dict):
        for key, val in ocr_fields.items():
            if isinstance(val, dict):
                text = val.get("ocr_text", "")
                latex = val.get("ocr_latex", "")
                texts.append(text)
                texts.append(latex)

    return " ".join(texts).lower()


# ==============================
# 3. SIGNAL FUNCTIONS
# ==============================

# --- (A) Symbol Density ---
def compute_symbol_density(text: str) -> float:
    symbols = re.findall(r'[=<>∑∫∆→←≈πθλμΩ]', text)
    return len(symbols) / (len(text) + 1)


# --- (B) Coordinate / Geometry Pattern ---
def detect_geometry_patterns(text: str) -> int:
    patterns = [
        r'\(\s*\d+\s*,\s*\d+\s*\)',   # (x, y)
        r'line\s+[a-z]{1,2}',
        r'angle\s+[a-z]{2,3}',
        r'triangle\s+[a-z]{3}',
        r'circle',
        r'radius',
        r'diameter'
    ]
    return sum(bool(re.search(p, text)) for p in patterns)


# --- (C) Physics Diagram Indicators ---
def detect_physics_context(text: str) -> int:
    keywords = [
        "force", "velocity", "acceleration",
        "circuit", "current", "voltage",
        "magnetic", "electric field",
        "block", "inclined plane"
    ]
    return sum(kw in text for kw in keywords)


# --- (D) OCR Noise Detection ---
def compute_ocr_noise(text: str) -> float:
    noise_patterns = [
        r'\b[a-z]{1}\b',       # single letters
        r'[^a-z0-9\s]{3,}',   # weird symbols
        r'\s{3,}'             # large spaces
    ]
    noise_score = sum(len(re.findall(p, text)) for p in noise_patterns)
    return noise_score / (len(text) + 1)


# --- (E) Latex Complexity ---
def compute_latex_complexity(text: str) -> int:
    latex_tokens = [
        "\\frac", "\\sum", "\\int", "\\sqrt",
        "\\begin", "\\end", "\\theta"
    ]
    return sum(token in text for token in latex_tokens)


# --- (F) Image Count (Ground hint) ---
def extract_image_count(ocr_fields: Dict) -> int:
    count = 0
    if isinstance(ocr_fields, dict):
        for val in ocr_fields.values():
            if isinstance(val, dict):
                count += val.get("image_count", 0)
    return count


# ==============================
# 4. FEATURE ENGINEERING
# ==============================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["ocr_fields"].apply(extract_text_fields)

    df["text"] = texts
    df["symbol_density"] = texts.apply(compute_symbol_density)
    df["geometry_score"] = texts.apply(detect_geometry_patterns)
    df["physics_score"] = texts.apply(detect_physics_context)
    df["ocr_noise"] = texts.apply(compute_ocr_noise)
    df["latex_complexity"] = texts.apply(compute_latex_complexity)
    df["image_count"] = df["ocr_fields"].apply(extract_image_count)

    return df


# ==============================
# 5. RULE-BASED SCORING ENGINE
# ==============================
def compute_image_likelihood(row) -> float:
    score = 0

    # Strong signals
    if row["image_count"] > 0:
        score += 5

    if row["geometry_score"] > 0:
        score += 3

    if row["symbol_density"] > 0.02:
        score += 2

    # Medium signals
    if row["physics_score"] > 1:
        score += 2

    if row["latex_complexity"] > 2:
        score += 1.5

    # Weak but useful signals
    if row["ocr_noise"] > 0.01:
        score += 1

    return score


# ==============================
# 6. PIPELINE RUNNER
# ==============================
def run_pipeline(json_path: str, threshold: float = 4.0) -> List[str]:

    print("Loading data...")
    df = load_data(json_path)

    print("Building features...")
    df = build_features(df)

    print("Scoring...")
    df["image_score"] = df.apply(compute_image_likelihood, axis=1)

    print("Filtering predictions...")
    predicted = df[df["image_score"] >= threshold]

    # Extract IDs
    ids = predicted["_id"].astype(str).tolist()

    print(f"Detected {len(ids)} likely image-based questions.")

    return ids


# ==============================
# 7. MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_jsonl> <output_txt> [threshold]")
        sys.exit(1)
    json_path = sys.argv[1]
    output_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

    image_question_ids = run_pipeline(json_path,threshold)

    # Save output
    if not output_path:
        output_path = "predicted_image_questions.txt"
        
    with open(output_path, "w") as f:
        for qid in image_question_ids:
            f.write(qid + "\n")

    print(f"Saved output to {output_path}")