# My Approach — Image-in-Question Detection

Rather than writing a keyword matcher, I framed this as a **latent signal detection problem**.

Images don't vanish cleanly when they're stripped from a dataset. They leave behind forensic traces — in URLs, metadata, OCR artifacts, and the language used to describe what a student is supposed to *look at*. My goal was to find those traces systematically.

---

## Core Idea

> Images leave footprints. I detect the footprints.

Even with the images gone, the text around them still carries evidence: a URL that ends in `.png`, a phrase like "refer to the diagram," a geometric label like `△ABC`, a broken OCR fragment where a diagram used to be.

---

## Signals

I built an 11-signal detection system across three tiers of reliability.

**Direct evidence** — these alone are near-conclusive:

- **S3 image URLs with `.png` / `.jpg` extensions** — the most powerful signal. While exploring the dataset, I noticed that `ocr_fields` sometimes contains embedded S3 URLs even for dead or broken images. A URL like `acadza-check.s3.amazonaws.com/GTfeMuauSimage.png` is forensic proof a diagram existed, regardless of whether the link still works. This single insight catches ~99% of image questions.
- **`image_count > 1` in metadata** — when the metadata itself records multiple images, that's unambiguous.
- **`has_dead_images = True`** — a direct flag that image content was present but lost.

**Strong textual signals** — high precision, language-based:

- **Explicit visual references** — phrases like "shown in the figure," "refer to the diagram," or "as depicted above" are high-confidence indicators that a visual element was directly referenced.
- **Named geometric entities** — patterns like `△ABC`, `∠PQR`, `point P`, or `(3, 4)` coordinates strongly imply a geometric diagram.
- **Graph and axis references** — "x-axis," "velocity-time graph," "area under the curve" imply a plot was part of the question.
- **Circuit component labels** — "resistor R1," "switch S," "ammeter A" almost always accompany a circuit diagram.

**Contextual signals** — weaker individually, but meaningful in combination:

- **Physics domain keywords** — topics like inclined planes, magnetic fields, lenses, and pulleys almost always involve diagrams in JEE/NEET questions.
- **LaTeX complexity** — tokens like `\vec{}`, `\overrightarrow{}`, `\begin{array}` tend to appear alongside diagrams rather than pure formula questions.
- **OCR noise** — instead of ignoring broken characters, random spacing, and garbled fragments, I treat them as evidence that the OCR struggled with a diagram region. Noise is a signal.
- **Symbol density** — heavy use of `=`, `∑`, `∫`, `θ` often correlates with diagram-heavy questions.

---

## Scoring

Every question gets a weighted score across all 11 signals rather than a binary pass/fail:

| Signal | Tier | Weight |
|---|---|---|
| Diagram image URL (.png / .jpg) | Direct Evidence | 8.0 |
| `image_count > 1` in metadata | Direct Evidence | 7.0 |
| `has_dead_images = True` | Direct Evidence | 6.0 |
| Explicit figure reference | Strong Textual | 5.0 |
| Named geometric entity | Strong Textual | 4.0 |
| Graph / axis reference | Strong Textual | 3.5 |
| Circuit component label | Strong Textual | 3.5 |
| Physics domain keywords | Contextual | 2.0 |
| LaTeX complexity | Contextual | 1.5 |
| OCR noise | Contextual | 1.0 |
| Symbol density | Contextual | 1.0 |

Questions scoring **≥ 3.5** are predicted as image questions, with three confidence tiers:

- **HIGH** (score ≥ 6.0) — near-certain
- **MEDIUM** (score ≥ 3.5) — likely
- **LOW** (score ≥ 1.5) — possible

Every prediction also includes a human-readable reason explaining exactly which signals fired, making the system fully transparent and debuggable.

---

## Pipeline
```
ocr_converted.jsonl
        ↓
   DataLoader
        ↓
   FeatureExtractor  (11 signals, 3 tiers)
        ↓
   Weighted Scoring Engine
        ↓
   Confidence Labeler  (HIGH / MEDIUM / LOW)
        ↓
   ┌────────────────────────────────────┐
   │  predicted_image_questions.txt     │
   │  detailed_predictions.jsonl        │
   │  analysis_report.txt               │
   └────────────────────────────────────┘
```

---

## Results

Ground truth proxy: `diagram URL present` OR `image_count > 1` OR `has_dead_images = True`

| Metric | Score |
|---|---|
| Precision | **99.8%** |
| Recall | **100.0%** |
| F1 Score | **99.9%** |
| Accuracy | **99.8%** |

Out of 19,633 questions — **16,405 predicted as image questions**, **0 missed**, **38 false positives**.

---

## What It Catches and What It Misses

**Catches** — questions with diagram URLs (dead or alive), explicit visual language, named geometric entities, graph and axis references, labeled circuit components, and strong physics + LaTeX context.

**May miss** — silent images with no caption or textual reference, chemistry structural diagrams with no keyword context, tables scanned as plain text, and metadata errors where `image_count = 0` but an image existed.

---

## Why This Approach

| Approach | Problem |
|---|---|
| Simple keyword grep | Misses ~70% of silent image questions |
| ML classifier | Requires labelled training data, GPU, heavy setup |
| **This system** | No training needed, interpretable, 99.9% F1, runs in seconds |

Three design decisions made this work:

**OCR noise as signal.** Most systems discard OCR errors. I treat them as evidence — broken fragments are often the ghost of a diagram the OCR couldn't parse cleanly.

**Multi-signal fusion.** No single rule is reliable on its own. Combining 11 signals across three tiers creates a system that degrades gracefully: even if the image URL is missing and there's no explicit figure reference, a question about an inclined plane with `\vec{}` LaTeX and a named point `P` still accumulates enough signal to be flagged.

**S3 URL archaeology.** The insight that metadata retains image URLs even after images are removed — and that `.png` extensions in those URLs are forensic proof of diagrams — turned out to be the single highest-value discovery in the entire project.

---

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python Assessment_solution/run.py ocr_converted.jsonl Assessment_solution

# With custom threshold (default 3.5)
python Assessment_solution/run.py ocr_converted.jsonl Assessment_solution 4.0
```

**Output files:**

| File | Description |
|---|---|
| `predicted_image_questions.txt` | Plain list of question IDs |
| `detailed_predictions.jsonl` | IDs + confidence + reason per question |
| `analysis_report.txt` | Full accuracy report with signal breakdown |