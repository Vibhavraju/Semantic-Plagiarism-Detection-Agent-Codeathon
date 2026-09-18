<<<<<<< HEAD
# Semantic Plagiarism Detection Agent

Compares two documents by **meaning**, not just matching words — so a student who
paraphrases or restructures a source still gets flagged.

## How it works

1. Both documents are split into sentences (rule-based splitter, no model download needed).
2. Every sentence is embedded with a sentence-transformer (`all-MiniLM-L6-v2`).
3. Cosine similarity is computed between every suspect sentence and every source sentence.
4. **Two-stage retrieval + rerank**: the bi-encoder finds the top-K candidate matches,
   then a cross-encoder (`stsb-distilroberta-base`) reranks those candidates for
   sharper, more accurate scores. Same pattern production semantic search systems use.
5. Each best-matching pair gets **two** scores:
   - **Semantic score** — do they mean the same thing? (cross-encoder reranked)
   - **Lexical score** — do they share the same words? (`difflib` sequence match)
6. High semantic + low lexical = paraphrasing. That gap is the whole point: a
   keyword-matching checker would miss it, this system catches it.
7. Consecutive matched sentences are merged into **sections** (matched passages),
   not reported as a scattered sentence list.
8. Overall similarity % = share of the suspect document's words that fall inside a
   high-risk (exact or paraphrase) match.

## Setup (needs internet once for model download)

```bash
cd backend
pip install -r requirements.txt
```

Models are downloaded and cached on first run (~180MB total for both the bi-encoder
and cross-encoder). After that the app works fully offline.

To pre-download before the demo:
```bash
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/stsb-distilroberta-base')
"
```

To run without the cross-encoder (bi-encoder scores only):
```bash
DISABLE_CROSS_ENCODER=1 uvicorn main:app --port 8000
```

## Run it

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the frontend is served automatically.
API docs: **http://localhost:8000/docs**

## Project structure

```
semantic-plagiarism-detector/
├── backend/
│   ├── main.py           # FastAPI app + full detection pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html        # single-file UI, calls /api/analyze and /api/extract
├── sample_docs/          # ready-made demo pairs
│   ├── source.txt        # urban green spaces essay
│   ├── suspect_paraphrased.txt  # paraphrased version
│   └── suspect_unrelated.txt    # unrelated ML text
└── README.md
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Returns `{"status": "ok", "cross_encoder": true/false}` |
| `POST` | `/api/analyze` | Body: `{"source_text": "...", "suspect_text": "..."}` — returns full report |
| `POST` | `/api/extract` | Multipart file upload (`.pdf`, `.docx`, `.txt`) — returns extracted text |

## Thresholds (tuned 2026-09-15)

| Constant | Value | Meaning |
|----------|-------|---------|
| `PARAPHRASE_THRESHOLD` | 0.60 | Semantic score >= this = paraphrase or close copy |
| `RELATED_THRESHOLD` | 0.48 | Below this = no match emitted at all |
| `NEAR_VERBATIM_SEM` | 0.85 | Semantic >= this + lexical >= 0.60 = near-verbatim |
| `PARAPHRASED_LEX` | 0.55 | Lexical BELOW this + sem >= 0.60 = reworded copy |
| `TOP_K_CANDIDATES` | 8 | Bi-encoder top-K sent to cross-encoder rerank |

## Requirement checklist

- [x] **Document/text processing** — sentence segmentation (`split_sentences`)
- [x] **File upload** — PDF (pdfplumber), DOCX (python-docx), TXT via `/api/extract`
- [x] **Semantic embeddings** — SBERT (`all-MiniLM-L6-v2`)
- [x] **Two-stage retrieval + rerank** — bi-encoder top-K + cross-encoder rerank
- [x] **Similarity calculation** — cosine similarity matrix
- [x] **Paraphrase detection** — semantic-vs-lexical gap (`classify`)
- [x] **Matching section identification** — consecutive-sentence merge (`merge_sections`)
- [x] **Similarity percentage** — `overall_similarity_percent`
- [x] **Plagiarism report** — verdict + matched passages + score ring + gap chips
- [x] **Edge cases** — empty input, identical documents, very short docs, no matches

## Demo script (60-90 seconds)

> Use the sample docs — don't type live.

**[0-5s] Hook — state the problem:**
> "Plagiarism checkers like Turnitin match keywords. If you rewrite a source in
> your own words, they miss it. This tool catches paraphrased plagiarism by
> comparing meaning, not matching words."

**[5-25s] Demo 1 — paraphrased pair (the headline feature):**
1. Click **Load sample source**, then **Load sample (paraphrased)**.
2. Click **Analyze documents**.
3. Point at the **100% score ring** and **"High risk of plagiarism"** verdict.
4. Scroll to a matched passage card. Read the two bars:
   > "Semantic meaning: 77% — these passages say the same thing. Wording overlap:
   > 33% — almost no exact word match. A 44-point gap. That gap is what a normal
   > plagiarism checker would miss."

**[25-40s] Demo 2 — unrelated pair (prove you don't flag everything):**
1. Click **Load sample source**, then click **Load sample (unrelated)**.
2. Click **Analyze documents**.
3. Point at **0% similarity** and **"Low similarity"** verdict.
   > "No false positives. These are both academic texts about different topics,
   > and the system correctly reports no plagiarism."

**[40-55s] Demo 3 — the engine (what's under the hood):**
1. Point at the **"two-stage rerank: ON"** chip.
   > "This uses the same two-stage pattern as production semantic search: a fast
   > bi-encoder finds candidate matches, then a cross-encoder reranks the top
   > candidates for sharp, accurate scores. CPU-only, runs in under 3 seconds."

**[55-70s] Demo 4 — file upload + edge cases:**
1. Point at the **Upload file** buttons.
   > "Supports PDF, DOCX, and TXT uploads — not just pasted text."
2. Mention edge case handling:
   > "Handles empty input, identical documents, and very short texts gracefully."

**[70-85s] Close — why it matters:**
> "The key insight: semantic similarity and lexical overlap measure different
> things. A plagiarism checker that only does keyword matching misses reworded
> copies. By computing both and showing the gap, this tool makes paraphrase
> detection visual and obvious."

**If time permits / asked about scale:**
> "This is a 1-vs-1 comparator. Extending to a corpus of N documents means
> embedding the corpus once and doing nearest-neighbor search — the bi-encoder
> keeps that scalable."

## Notes

- Swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` for higher quality at the cost of
  more latency (larger model, slower on CPU).
- Thresholds are calibrated on the sample docs — adjust `PARAPHRASE_THRESHOLD` etc. in
  `main.py` if your real-world data needs different boundaries.
- The cross-encoder fallback (`DISABLE_CROSS_ENCODER=1`) uses bi-encoder scores directly.
  Scores will be less sharp but the pipeline still works.
=======
# Semantic-plagiarism-detection-agent-codeathonv1
>>>>>>> 9f6c9ad505187bece864fa8be1d380cc6528622d
