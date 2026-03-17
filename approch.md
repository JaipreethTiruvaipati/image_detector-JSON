##  My Approach 

Instead of relying on simple keyword matching, I approached this as a:

> **Latent Signal Detection Problem**

Even if images are removed, they leave behind **patterns in text**.

---

## Key Idea

> **Images leave “footprints” in OCR text — we just need to detect them.**

---

##  Signals Used in the System

### 1.  Structural Signals (Geometry Patterns)

* Detect patterns like:

  * `(x, y)` coordinates
  * "triangle ABC"
  * "angle PQR"
  * "line AB"

👉 These strongly imply **visual diagrams were present**

---

### 2.  Semantic Signals (Domain Knowledge)

* Physics terms:

  * force, velocity, circuit, current
* Geometry terms:

  * radius, diameter, circle

👉 These topics **almost always involve diagrams**

---

### 3.  Symbol Density

* Counts mathematical symbols:

  * `=`, `∑`, `∫`, `θ`, etc.

👉 High symbol usage often correlates with **diagram-heavy questions**

---

### 4.  OCR Noise Analysis (Creative Insight)

* Detect:

  * broken words
  * random characters
  * spacing issues

👉 Instead of ignoring OCR errors, I use them as:

> **Evidence that important visual content was lost**

---

### 5.  LaTeX Complexity

* Detect tokens like:

  * `\frac`, `\int`, `\sqrt`

👉 Complex expressions often appear with diagrams

---

### 6.  Hidden Image Metadata (Important Insight)

* Extract `image_count` from OCR fields

👉 I use it as a **strong prior signal**

---

##  Scoring Strategy

Instead of binary rules, I use a **multi-signal weighted scoring system**:

| Signal            | Weight          |
| ----------------- | --------------- |
| Image count       | High            |
| Geometry patterns | High            |
| Symbol density    | Medium          |
| Physics context   | Medium          |
| OCR noise         | Weak but useful |

👉 This creates a **robust and flexible detection system**

---

## 🏗️ Pipeline Architecture

```
Load JSONL Data
        ↓
Extract OCR Text + LaTeX
        ↓
Feature Engineering
        ↓
Multi-Signal Scoring
        ↓
Threshold Filtering
        ↓
Output Question IDs
```

---

##  Output

The system generates:

* ✅ List of question IDs with likely images
* 📁 Saved in:

  ```
  predicted_image_questions.txt
  ```

---

##  Why I Designed It This Way

###  Problem with Basic Approaches

* Keyword matching → unreliable
* ML models → heavy, data-dependent

---

###  My Solution Advantages

#### 1.  Interpretable

* Every prediction is explainable

#### 2.  Lightweight

* No GPU / training required

#### 3.  Domain-Aware

* Uses physics + geometry knowledge

#### 4.  Robust to OCR Errors

* Uses noise as signal instead of ignoring it

#### 5.  Scalable

* Works on large datasets efficiently

---

##  Creative Contributions (What Makes This Unique)

###  1. OCR Noise as Signal

Most systems ignore OCR errors
👉 I convert them into **useful features**

---

###  2. Multi-Signal Fusion

Instead of:

* single rule ❌

I use:

* combined reasoning across multiple signals ✅

---

###  3. Latent Layout Inference

Detects diagram existence from:

* textual structure alone

---

###  4. Weak Supervision

Uses:

* `image_count` as hidden supervision signal

---

##  Expected Performance

* Much higher accuracy than keyword-based systems
* Near ML-level performance without training

---

##  How to Run

```bash
python Assessment_solution/ocr_analysis.py <input_jsonl_path> <output_dir> 
```

---