# Image-in-Question Detector (v2.0)

This repository contains a heuristic-based detection engine designed to identify JEE/NEET exam questions that originally contained diagrams, circuits, or geometric figures before being processed by OCR. 

Because OCR often strips or mangles images without leaving explicit trace tokens, this detector uses a **weighted, multi-signal algorithmic approach** to find the structural footprints that images leave behind in the text and HTML.

---

## 🧠 Core Philosophy
> **Images leave structural footprints in OCR data. We decode the evidence, not just the keywords.**

Every question in the dataset originated from a scanned PDF. The primary challenge is distinguishing questions with **embedded content diagrams** (circuits, graphs, organic structures) from questions whose only images are **formula render tiles** (math equations rendered as PNGs). 

This script relies on 9 independent "signals". No single signal is perfectly reliable, but when multiple signals trigger on the same question (Signal Co-occurrence), confidence scales geometrically.

---

## ⚙️ How the Engine Works: The 9 Signals

The script (`detector_v2.py`) processes the `ocr_fields` (both raw text and structural HTML) and grades them across 9 pattern domains:

### 1. Explicit Visual References (`S1`)
- **What it looks for**: Phrases like *"shown in the figure"*, *"refer to the given diagram"*, *"match the column"*.
- **Why it matters**: This is the most direct evidence. It has a **Negative Guard** that penalizes the score if it finds *"Which of the following is NOT shown"* (preventing trick questions from flagging false positives).

### 2. Spatial / Deictic language (`S2`)
- **What it looks for**: Directional pointers like *"in the above"*, *"marked in"*, *"to the left"*, *"indicated by"*.
- **Why it matters**: A question telling a student to look "at the left" strongly implies spatial reasoning linked to a figure.

### 3. Orphaned Option Labels (`S3`)
- **What it looks for**: Sequences of multiple choice labels like `(A) \n (B) \n (C)` with less than 15 characters of text between them.
- **Why it matters**: If an OCR engine reads options that were purely images (e.g., 4 organic chem structures), it extracts just the labels, leaving them "orphaned".

### 4. Semantic Incompleteness (`S4`)
- **What it looks for**: Dangling sentences like *"Identify P in the..."* or sentences ending abruptly in *"as shown."*
- **Why it matters**: Detects grammatical cliffs where the sentence was supposed to be completed by a visual element.

### 5. Subject-specific LaTeX / Technical Symbols (`S5`)
- **What it looks for**: Hard LaTeX environments `\begin{circuitikz}`, `\chemfig{}`, `\begin{tikzpicture}`, `ray diagram`, `FBD`.
- **Why it matters**: These specific LaTeX packages are strictly used to render visual diagrams.

### 6. OCR Artifacts and Placeholders (`S6`)
- **What it looks for**: Literal placeholders like `[IMAGE]`, `[FIGURE]`, `(image)`, or dense clusters of replacement characters (`□□□`, `???`).
- **Why it matters**: Direct evidence that an OCR pipeline acknowledged an image block but failed to parse it.

### 7. Coordinate / Value Density (`S7`)
- **What it looks for**: Dense clusters of Cartesian coordinates `(x, y)` or vector points in mathematics.
- **Why it matters**: High coordinate density usually accompanies a geometric plot or shaded region question.

### 8. HTML Structural Anomalies (`S8`) - *[v2.0 Feature]*
- **What it looks for**: Un-stripped raw HTML elements:
  - Empty `<td></td>` tags (Match-the-column grids where images used to sit).
  - Clusters of `<br><br><br>` (A vertical gap where an image was ripped out).
  - High `<math>` tag depth with very few visible characters.
- **Why it matters**: Catches non-text diagram indicators that standard NLP regex will miss.

### 9. Vector / Spatial Math Recognition (`S9`) - *[v2.0 Feature]*
- **What it looks for**: $\hat{i}, \hat{j}, \hat{k}$ unit vectors, "bounded by the curve", "area enclosed", cross-products.
- **Why it matters**: Catches advanced Physics and Calculus diagram questions that never explicitly use the word "figure".

---

## ⚖️ Scoring Methodology and Subject Disambiguation

Because OCR text provides highly variable reliability based on the subject (e.g., a Biology question using the word "cell" implies something entirely different than a Physics question using "cell" - battery), the engine calculates multipliers using **Subject Detection** and applies a carefully tuned weighted scoring system.

### 1. Base Weights
Every signal has a base weight assigned based on its historic precision:
* **`S1` (0.35)**: Explicit references ("shown in the figure") are the strongest definitive proof.
* **`S2` (0.15)**: Spatial language ("to the left") strongly implies a visual layout.
* **`S5` (0.12)**: Subject-specific LaTeX (`\chemfig{}`, `\begin{circuitikz}`) guarantees a diagram environment.
* **`S3` & `S4` (0.10)**: Orphaned labels and semantic cliffs are strong circumstantial evidence.
* **`S6` & `S8` (0.08 & 0.05)**: OCR placeholders and HTML anomalies (`<td></td>`) provide structural backend evidence.
* **`S7` & `S9` (0.03 & 0.02)**: Coordinate density and Vector math act as subtle supporting evidence mathematically.

### 2. Subject Detection & Component Multipliers
The engine first guesses the subject based on vocabulary density (e.g., `circuit`, `momentum` vs `mitosis`, `enzyme`). Tiebreakers are handled contextually (e.g. `nucleus` + `radioactive` = Physics; `nucleus` + `membrane` = Biology).

Once the subject is determined, **Subject Multipliers** are applied to the base weights. This is crucial because a signal's importance changes drastically depending on the subject:
* **Physics & Mathematics**: `S9` (Vector/spatial math) gets a **1.8x** multiplier. `S7` (Coordinate density) gets a **2.2x** multiplier in Math because geometric plots heavily rely on `(x, y)` coordinates, while it remains standard in Biology.
* **Chemistry**: `S5` (LaTeX) gets a massive **2.2x** multiplier because chemical diagram rendering almost exclusively relies on specific LaTeX commands compared to standard text. 
* **Biology**: `S1` and `S2` get high multipliers (**1.8x**) because Biology questions rarely use advanced LaTeX or coordinate grids; they rely heavily on language like "label A in the given diagram". `S8` (HTML anomalies) gets a **2.0x** boost to catch missing anatomical tables/grids.

### 3. Subject-Specific Confidence Thresholds (The Sweet Spot)
After calculating the sum of `(Base Weight * Subject Multiplier)` for all matched signals, it is divided by the maximum possible score for that subject to get a final `score` between `0.0` and `1.0`.

To maximize recall (finding every possible image) without destroying precision (flagging false positives), the detection threshold (the "sweet spot" at which a question is officially flagged) is dynamically adjusted per subject:
* **Physics / Unknown (`0.45`)**: Physics has very strong signals (circuits, vectors), so the threshold remains moderately high.
* **Mathematics (`0.35`)**: Lowered because math diagrams (like small geometry plots) often trigger fewer linguistic signals and rely on quiet structural hints.
* **Chemistry (`0.30`)**: Lowered further to catch organic chemistry reaction schemes which often fail OCR entirely and leave very little text behind.
* **Biology (`0.25`)**: The lowest threshold. Biology questions with images are often visually simple (e.g., a single picture of a cell) and leave almost zero footprints other than "Identify the part marked A". 

### 4. Advanced Guards and Amplifications
* **Negative Guarding**: If `S1` triggers on the word "figure", it first checks for phrases like *"Which of the following is NOT shown in the figure"*. If found, it imposes a massive penalty (multiplier of `0.4x`) to prevent trick questions from flagging as false positives.
* **Co-occurrence Amplification**: If 3 or more independent signals fire (e.g., A spatial pointer + An orphaned label + An empty HTML table cell), it means the evidence is overwhelming. The algorithm artificially boosts the final score by up to `+0.15`, often securing a "Definite" confidence flag (`≥ 0.80`).

---

## 🚀 Execution & Performance Improvements

The v2.0 engine uses **Compiled Regex Globals**, bringing the processing speed of 19,633 rows down to approximately **0.3 seconds**.

### Running the Detector
```bash
# Standard grading (Uses subject-calibrated semantic thresholds)
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results

# Run with verbose debugging for the top 20 questions
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results --verbose

# Output mathematical weight balancing suggestions (for threshold tuning)
python3 detector_v2.py --input ocr_converted.jsonl --output-dir ./results --tune
```

### Understanding Outputs (`/results`)
- `image_questions.jsonl`: The primary output. Contains all flagged questions where the engine confidently determined an image was present.
- `all_scores.csv`: The entire dataset dump, containing raw decimal scores for all 9 individual signal trackers.
- `review_queue.jsonl`: Questions scoring between `0.40 - 0.60`. These are borderline cases suitable for human review or downstream syntactic dependency parsing.
- `detection_summary.txt`: A detailed operational log tracking false positive rates, signal redundancy (how often a signal fired completely alone), and accuracy estimation metrics.
