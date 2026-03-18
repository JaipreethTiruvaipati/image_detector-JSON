# =============================================================================
# detector.py — Main ImageInQuestionDetector class (OOP pipeline)
# =============================================================================
import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# Make sure Assessment_solution is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import PREDICTION_THRESHOLD
from signals.metadata import extract_image_urls, extract_diagram_url_count, extract_image_count, extract_has_dead_images
from signals.explicit_refs import detect_explicit_refs, detect_graph_axis_refs, detect_circuit_components
from signals.geometric import detect_geometry_entities
from signals.advanced import compute_latex_complexity, detect_physics_domain, compute_ocr_noise, compute_symbol_density
from scoring import compute_score, assign_confidence, build_reason


class ImageInQuestionDetector:

    def __init__(self, threshold: float = PREDICTION_THRESHOLD):
        self.threshold = threshold
        self.df = None
        self.results = None

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    def load(self, jsonl_path: str):
        print(f"[1/4] Loading data from: {jsonl_path}")
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        self.df = pd.DataFrame(records)
        print(f"      Loaded {len(self.df):,} questions.")
        return self

    # ── 2. Extract Text ───────────────────────────────────────────────────────
    def _extract_text(self, ocr_fields: dict) -> str:
        texts = []
        if isinstance(ocr_fields, dict):
            for val in ocr_fields.values():
                if isinstance(val, dict):
                    texts.append(str(val.get("ocr_text", "") or ""))
                    texts.append(str(val.get("ocr_latex", "") or ""))
                    texts.append(str(val.get("html", "") or ""))
        return " ".join(texts)

    def _extract_id(self, raw_id) -> str:
        if isinstance(raw_id, dict):
            return raw_id.get("$oid", str(raw_id))
        return str(raw_id)

    # ── 3. Build Features ─────────────────────────────────────────────────────
    def extract_features(self):
        print("[2/4] Extracting features...")
        tqdm.pandas(desc="      Processing")

        df = self.df
        df["question_id"]      = df["_id"].apply(self._extract_id)
        df["combined_text"]    = df["ocr_fields"].progress_apply(self._extract_text)

        df["url_count"]         = df["ocr_fields"].apply(extract_image_urls)
        df["diagram_url_count"] = df["ocr_fields"].apply(extract_diagram_url_count)
        df["image_count_meta"]  = df["ocr_fields"].apply(extract_image_count)
        df["has_dead_images"]   = df["has_dead_images"].apply(extract_has_dead_images)

        df["explicit_ref"]     = df["combined_text"].apply(detect_explicit_refs)
        df["graph_axis_ref"]   = df["combined_text"].apply(detect_graph_axis_refs)
        df["circuit_component"]= df["combined_text"].apply(detect_circuit_components)
        df["geometry_entity"]  = df["combined_text"].apply(detect_geometry_entities)

        df["latex_complexity"] = df["combined_text"].apply(compute_latex_complexity)
        df["physics_domain"]   = df["combined_text"].apply(detect_physics_domain)
        df["ocr_noise"]        = df["combined_text"].apply(compute_ocr_noise)
        df["symbol_density"]   = df["combined_text"].apply(compute_symbol_density)

        self.df = df
        print("      Feature extraction complete.")
        return self

    # ── 4. Score & Predict ────────────────────────────────────────────────────
    def predict(self):
        print("[3/4] Scoring and predicting...")
        self.df["image_score"]  = self.df.apply(compute_score, axis=1)
        self.df["confidence"]   = self.df["image_score"].apply(assign_confidence)
        self.df["reason"]       = self.df.apply(build_reason, axis=1)
        self.df["has_image_predicted"] = self.df["image_score"] >= self.threshold

        predicted = self.df[self.df["has_image_predicted"]]
        print(f"      Predicted {len(predicted):,} image questions out of {len(self.df):,} total.")
        print(f"      HIGH confidence: {len(predicted[predicted['confidence']=='HIGH']):,}")
        print(f"      MEDIUM confidence: {len(predicted[predicted['confidence']=='MEDIUM']):,}")
        self.results = predicted
        return self

    # ── 5. Save Outputs ───────────────────────────────────────────────────────
    def save(self, output_dir: str):
        print("[4/4] Saving outputs...")
        os.makedirs(output_dir, exist_ok=True)

        # Plain ID list (required deliverable)
        ids_path = os.path.join(output_dir, "predicted_image_questions.txt")
        with open(ids_path, "w") as f:
            for qid in self.results["question_id"].tolist():
                f.write(qid + "\n")
        print(f"      Saved {len(self.results):,} IDs → {ids_path}")

        # Detailed predictions with confidence + reason
        detail_path = os.path.join(output_dir, "detailed_predictions.jsonl")
        detail_cols = ["question_id", "qtype", "image_score", "confidence", "reason"]
        self.results[detail_cols].to_json(detail_path, orient="records", lines=True)
        print(f"      Saved detailed predictions → {detail_path}")

        return self

    # ── Full Pipeline ─────────────────────────────────────────────────────────
    def run(self, jsonl_path: str, output_dir: str):
        return (self
                .load(jsonl_path)
                .extract_features()
                .predict()
                .save(output_dir))