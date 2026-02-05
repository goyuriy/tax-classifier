# Tax Form Classification Pipeline

## Problem

Classify pages in multi-page PDF tax documents into specific form types (Form 1040, Schedule A, etc.). Must handle OCR noise, mixed layouts, and run on CPU.

## Solution

Hybrid classifier: regex heuristics (fast, high precision) + embeddings (robust to noise). 0.96 accuracy on 152 pages.

**Stack:**
- `pdfplumber` — extracts text and layout
- `all-MiniLM-L6-v2` (384d) — embeddings, CPU-optimized
- `LogisticRegression` — fast, interpretable, works with small data
- `Pydantic` — typed API contracts
- Structured logging — no `print()`

## Why This Approach

### Considered Alternatives

1. **Pure regex/rules:** Fast, but breaks on OCR noise or non-standard layouts. 0 generalization.
2. **LayoutLM/multimodal (text + bounding boxes):** Best accuracy, but requires GPU, ~200ms/page, and needs bounding box annotations. Overkill for MVP.
3. **End-to-end transformer (e.g. BERT classifier):** Slower inference, needs more training data. LogReg on embeddings is 10x faster with same accuracy on this dataset.
4. **TF-IDF + SVM:** Works, but embeddings capture semantic similarity better (e.g. "Sch A" vs "Schedule A").

### Trade-offs Made

| Decision | Why | Cost |
|----------|-----|------|
| **Hybrid (heuristics + ML)** | 80%+ of forms are standard → regex catches them instantly. ML handles edge cases. | Two models to maintain. |
| **CPU-only (no GPU)** | Production deployment cost. Most tax forms are text-heavy; embeddings are fast enough. | Can't use LayoutLM or large models. |
| **LogReg on embeddings** | Interpretable (weights per class), fast inference, works with pages data. | Less expressive than neural classifier. |
| **Train/Test Split + Page-level Validation** | Explicit negative sampling ("background" pages) to reduce false positives. | Requires manually labeled ground truth. |
| **No OCR** | MVP scope. Real OCR adds latency and cost. | Scanned pages return "scanned_document". |

### Metrics: Why F1?

**Problem:** Multiclass classification with imbalanced classes (e.g. more 1040 pages than Schedule E).

**Why not accuracy?** Accuracy = (TP + TN) / total. Dominated by the majority class. A classifier that always predicts "1040f" could get 60% accuracy but miss all schedules.

**Why F1?** F1 = 2 × (precision × recall) / (precision + recall). Balances false positives and false negatives. For document classification, both matter:
- **High precision:** Don't mis-tag forms (causes downstream errors).
- **High recall:** Don't miss forms (causes incomplete processing).

**Macro-averaged F1** (average F1 per class) ensures small classes aren't ignored.

## How It Works

### Two-Tier Classification

1. **Heuristics (Tier 1):** Regex on first 50 chars of page. Catches 80%+ of standard forms instantly.
2. **Semantic (Tier 2):** First 200 chars → embeddings → LogReg. Handles OCR errors, wording variation, and context.

**Why 200 chars?**
Previously 50 chars, but increased to 200 to capture context like "See Schedule A" vs "Schedule A". The model learns to distinguish form headers from references by training on "background" pages.

### Validation Strategy

1. **Train/Test Split:** Standard 80/20 split on pages, stratified by class.
2. **Page-level Metrics:** We evaluate every page, including unlabeled ones.
   - **True Positives:** Correctly identified forms.
   - **True Negatives:** Background pages correctly ignored.
   - **False Positives (Misleading Mentions):** Background pages incorrectly flagged as forms (e.g. instructions mentioning a form).
   - **Wrong Class:** Form A classified as Form B.

Explicitly trained on "background" pages (negative sampling) to minimize misleading mentions.

### Edge Cases

**Unknown forms:** Confidence threshold (default 0.7). Below threshold → flag for manual review or route to LLM.

**Scanned pages:** Text density check. <50 chars + images present → flag as scan. For production OCR: trigger Textract, or open-source models like [docTR (Mindee)](https://github.com/mindee/doctr) or [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)—both fast, accurate, and easy to host locally. Here: log a warning, but **no actual OCR**.

## Results

Validation on test set (152 pages):

### Semantic Only

```text
              precision    recall  f1-score   support

       1040f       0.50      1.00      0.67         4
     f1040s1       0.40      1.00      0.57         2
     f1040s3       0.67      1.00      0.80         4
     f1040sa       1.00      1.00      1.00         1
     f1040sb       1.00      1.00      1.00         2
     f1040sd       1.00      1.00      1.00         3
     f1040se       1.00      1.00      1.00         1
       f8889       1.00      1.00      1.00         1
       f8949       1.00      1.00      1.00         2
       other       1.00      0.93      0.96       132

    accuracy                           0.94       152
   macro avg       0.86      0.99      0.90       152
weighted avg       0.97      0.94      0.95       152

Accuracy: 0.9408
```

### Hybrid Approach (Heuristics + Semantic)

```text
              precision    recall  f1-score   support

       1040f       1.00      1.00      1.00         4
     f1040s1       1.00      1.00      1.00         2
     f1040s2       0.00      0.00      0.00         0
     f1040s3       1.00      1.00      1.00         4
     f1040sa       1.00      1.00      1.00         1
     f1040sb       0.67      1.00      0.80         2
     f1040sd       1.00      1.00      1.00         3
     f1040se       1.00      1.00      1.00         1
       f8889       1.00      1.00      1.00         1
       f8949       1.00      1.00      1.00         2
       other       1.00      0.95      0.98       132

    accuracy                           0.96       152
   macro avg       0.88      0.90      0.89       152
weighted avg       1.00      0.96      0.98       152

Hybrid Accuracy: 0.9605
```

## What's Missing

- **Layout features:** LayoutLMv3 for bounding boxes (needed when forms share text but differ in structure).
- **Active learning:** Human corrections → retrain loop.
- **RAG:** Extract line-item data after classification.

## Project Structure

```
├── api.py              # FastAPI service
├── data/               # input/, output/, target/
├── models/             # semantic_classifier.pkl
├── notebooks/          # exploration, solution
├── src/
│   ├── loader.py       # PDF load, scan detection
│   ├── models.py       # Heuristic, Semantic, Hybrid
│   ├── pipeline.py     # end-to-end
│   ├── train_utils.py  # data prep
│   └── utils.py        # logging
├── main.py             # CLI
├── Makefile
└── requirements.txt
```

## Usage

Prefer **make**; use raw **python** only when needed (e.g. custom args).

### Setup

```bash
make setup
source .venv/bin/activate   # if not using make for run commands
```

Optional: `pip install -r requirements.txt` (or use uv/venv yourself).

### Train

Reads `data/input/*.pdf` and `data/target/*.json`, writes `models/semantic_classifier.pkl`.

```bash
make train
```

Optional: `python main.py --mode train` (e.g. with `--data_dir`, `--model_path`, `--output_dir`).

### Predict

Reads `data/input/*.pdf`, writes `data/output/*.json`.

```bash
make predict
```

Optional: `python main.py --mode predict` with same override options.

### API

```bash
make run-api
```

Serves at `http://0.0.0.0:8000`. Optional: `uvicorn api:app --reload --host 0.0.0.0 --port 8000`.

**Endpoints:**
- `GET /health` → `{"status": "ok", "model_path": "..."}`
- `POST /classify` → upload PDF, get `{"forms": [{"document_type": "1040f", "start_page": 1, "end_page": 2}, ...]}`

```bash
curl -X POST http://localhost:8000/classify -F "file=@data/input/dummy1.pdf"
```

---

## Development Process

### Problem-Solving Approach

1. **Explored the data:** 10 PDFs, mixed layouts (digital + scanned). Forms have consistent headers ("Schedule A", "Form 1040") but body text varies.
2. **Hypothesis:** Header text is high-signal. Don't need to process the entire page for most cases.
3. **Tested heuristics first:** Regex on first 50 chars. Caught 60%+ instantly. Fast fail for unknown forms.
4. **Added semantic fallback:** For pages where heuristics miss (typos, OCR errors). Embeddings + LogReg generalizes better than pure regex.
5. **Validation:** Switched to page-level validation with explicit negative sampling ("background" class) to reduce false positives (misleading mentions).
6. **Refinements:** Increased context window from 50 to 200 chars to capture references ("See Schedule A"). Added scan detection and confidence thresholding.

### AI Assistant Usage

**Tools used:**
- Antigravity for research and brainstorming, also for domain-specific knowledge.
- Cursor with various models, subagents, custom instructions and tools

**How AI helped:**
- Geting domain-specific knowledge and information collection.
- Data preparation and validation instructions.
- Pydantic models and API schemas
- API implementation
- Frontend implementation


**Configuration:**
- `.cursorrules` — defines engineering style, functional patterns, no documentation unless asked.
- subagents for specific tasks like data preparation, API implementation, logging implementation, etc.
- data preparation and validation instructions, 

Note:
all .cursor configuration done with advanced prompting techiques ex. introspection of thoughts prompting, role focused subagents, and advanced tools like deployment tools and instructions.