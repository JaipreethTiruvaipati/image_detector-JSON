# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image-in-Question Detector v2.0 — A heuristic-based detection engine that identifies JEE/NEET exam questions that originally contained diagrams, circuits, or geometric figures before OCR processing. The system uses a weighted, multi-signal algorithmic approach to find structural footprints images leave behind in text and HTML.

## Commands

```bash
# Run detector with subject-calibrated thresholds (default)
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results

# Run with verbose debugging for first 20 questions
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results --verbose

# Run weight tuning diagnostics
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results --tune

# Run with global threshold instead of subject-calibrated
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results --global-threshold

# Run with custom threshold
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results --threshold 0.45

# Run the legacy OCR analysis pipeline (Assessment_solution/)
python3 Assessment_solution/ocr_analysis.py ocr_converted.jsonl output.txt 4.0
```

## Architecture

### Core Engine (`detector_v2.py`)

The detector processes JSONL input containing OCR-extracted fields and scores each question across **9 independent signals**:

| Signal | Weight | Description |
|--------|--------|-------------|
| S1 | 0.35 | Explicit visual references ("figure", "diagram", "as shown") |
| S2 | 0.15 | Spatial/deictic language ("to the left", "marked in", "indicated by") |
| S3 | 0.10 | Orphaned option labels (A/B/C/D with minimal text between) |
| S4 | 0.10 | Semantic incompleteness (dangling sentences like "as shown.") |
| S5 | 0.12 | Subject-specific LaTeX/symbols (circuitikz, chemfig, \\hat{i}) |
| S6 | 0.08 | OCR artifacts/placeholders ([IMAGE], □□□, ???) |
| S7 | 0.03 | Coordinate/value density (Cartesian pairs, vector points) |
| S8 | 0.05 | HTML structural anomalies (empty `<td>`, consecutive `<br>`) |
| S9 | 0.02 | Vector/spatial math (unit vectors, "bounded by", cross products) |

### Scoring Pipeline

1. **Subject Detection**: Infers subject (Physics, Chemistry, Biology, Mathematics) from field value or vocabulary scoring with tiebreaker logic for ambiguous terms (e.g., "nucleus")
2. **Signal Extraction**: Each signal returns a score (0-1) and reasons
3. **Weighted Scoring**: Applies subject-specific multipliers to signal weights
4. **Co-occurrence Amplification**: Boosts score when 3+ signals fire independently
5. **Hard Overrides**: Force-high for definitive patterns (circuitikz, chemfig, placeholders); force-low for long self-contained text
6. **Confidence Tiers**: definite (≥0.80), probable (0.60-0.80), possible (0.50-0.60), unlikely (<0.50)

### Output Files (`/results`)

- `image_questions.jsonl` — Flagged questions with full signal breakdown
- `all_scores.csv` — Complete dataset with all 9 signal scores
- `review_queue.jsonl` — Borderline cases (scores 0.40-0.60) for human review
- `detection_summary.txt` — Diagnostic report with accuracy estimates

### Key Design Decisions

- **Pre-compiled regexes**: All patterns compiled at module load for ~15% speedup on 20k questions
- **Subject multipliers**: S7 (coordinates) gets 1.8x in Math but 1.0x in Biology; S5 (Chem LaTeX) gets 1.5x in Chemistry
- **Negative guard on S1**: Phrases like "NOT shown" or "incorrectly drawn" apply 0.4x penalty to prevent false positives on trick questions
- **Signal co-occurrence is intentional**: Multiple signals firing on the same phrase indicates strong evidence, not redundancy

### File Structure

```
├── detector_v2.py              # Main detection engine (1066 lines)
├── tune_threshold.py           # Threshold tuning utility
├── Assessment_solution/
│   ├── ocr_analysis.py         # Legacy v1 pipeline (pandas-based)
│   ├── image_detector.py       # Legacy v1 detector
│   ├── approach.md             # Approach documentation
│   └── notebook/
│       └── experiment.ipynb    # Feature engineering notebook
└── results/                    # Output directory
```

### Dependencies

```
pandas, tqdm, scikit-learn, numpy, matplotlib
```

Note: `regrex` in requirements.txt is likely a typo for `re` (built-in) or `regex` package.
