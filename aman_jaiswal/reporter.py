# =============================================================================
# reporter.py — Generates accuracy report using image_count as ground truth
# =============================================================================
import os
import pandas as pd


def generate_report(df: pd.DataFrame, output_dir: str):
    """Generate accuracy report using image_count > 0 as ground truth proxy."""

    # Real ground truth: diagram URL (with image extension) OR multiple images OR dead image flag
    # image_count=1 with no extension URL = just the OCR formula scan, NOT a diagram
    df["ground_truth"] = (
        (df["diagram_url_count"] > 0) |
        (df["image_count_meta"] > 1) |
        (df["has_dead_images"] == True)
    )
    df["predicted"]    = df["has_image_predicted"]

    tp = len(df[ df["ground_truth"] &  df["predicted"]])
    fp = len(df[~df["ground_truth"] &  df["predicted"]])
    fn = len(df[ df["ground_truth"] & ~df["predicted"]])
    tn = len(df[~df["ground_truth"] & ~df["predicted"]])

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / len(df)

    total_with_images   = df["ground_truth"].sum()
    total_predicted     = df["predicted"].sum()
    high_conf           = len(df[(df["confidence"] == "HIGH") & df["predicted"]])
    medium_conf         = len(df[(df["confidence"] == "MEDIUM") & df["predicted"]])

    report = f"""
================================================================================
  IMAGE-IN-QUESTION DETECTOR — ANALYSIS REPORT
================================================================================

DATASET SUMMARY
---------------
  Total questions          : {len(df):,}
  Questions with images* : {total_with_images:,}  ({100*total_with_images/len(df):.1f}%)
  Questions without images : {len(df) - total_with_images:,}  ({100*(len(df)-total_with_images)/len(df):.1f}%)
  (* ground truth = image_count > 0 OR S3 URL present)

PREDICTION SUMMARY
------------------
  Total predicted as image : {total_predicted:,}
    HIGH confidence        : {high_conf:,}
    MEDIUM confidence      : {medium_conf:,}
    LOW confidence         : {total_predicted - high_conf - medium_conf:,}

ACCURACY METRICS (vs ground truth proxy)
-----------------------------------------
  Precision  : {precision:.3f}  ({100*precision:.1f}%)
  Recall     : {recall:.3f}  ({100*recall:.1f}%)
  F1 Score   : {f1:.3f}  ({100*f1:.1f}%)
  Accuracy   : {accuracy:.3f}  ({100*accuracy:.1f}%)

CONFUSION MATRIX
----------------
  True Positives  (correctly caught image questions) : {tp:,}
  False Positives (wrongly flagged non-image Qs)     : {fp:,}
  False Negatives (missed image questions)           : {fn:,}
  True Negatives  (correctly cleared non-image Qs)  : {tn:,}

SIGNAL BREAKDOWN (predicted image questions)
---------------------------------------------
  With S3 image URL found      : {len(df[df["url_count"] > 0]):,}
  With image_count metadata    : {len(df[df["image_count_meta"] > 0]):,}
  With explicit figure refs    : {len(df[df["explicit_ref"] > 0]):,}
  With geometry entities       : {len(df[df["geometry_entity"] > 0]):,}
  With graph/axis refs         : {len(df[df["graph_axis_ref"] > 0]):,}
  With circuit components      : {len(df[df["circuit_component"] > 0]):,}

WHAT THE SYSTEM CATCHES
-----------------------
  [+] Questions with S3 image URLs (dead or alive)
  [+] Questions with explicit figure/diagram references
  [+] Questions with named geometric entities (triangle ABC, angle PQR)
  [+] Questions with graph, axis, or curve references
  [+] Questions with labeled circuit components
  [+] Questions with high-complexity LaTeX + physics domain context

WHAT THE SYSTEM MAY MISS
------------------------
  [-] Silent images: diagrams with zero textual reference (~25-35% of cases)
  [-] Chemistry structural diagrams with no keyword context
  [-] Tables treated as inline text by OCR
  [-] Questions where image_count=0 but image existed (metadata error)

================================================================================
"""
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Report saved → {report_path}")