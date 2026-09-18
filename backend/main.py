"""
Semantic Plagiarism Detection Agent — backend

Compares two documents for MEANING-based similarity, not just shared
words, so paraphrased or restructured plagiarism still gets caught.

Pipeline: split into sentences -> embed with SBERT (bi-encoder, fast
retrieval) -> cosine similarity matrix -> pick top-K candidate pairs ->
rerank those pairs with a cross-encoder (accurate scoring) -> classify
each match (semantic score vs lexical/wording overlap) -> the semantic/
lexical GAP is the paraphrase signature -> merge consecutive matches
into sections -> score + report. 

Two-stage retrieval + rerank is the same pattern production semantic
search systems use: the bi-encoder keeps it fast (no all-vs-all
cross-encoder scoring), the cross-encoder makes the final scores sharp.
"""

import os
import re
from difflib import SequenceMatcher
from io import BytesIO
from typing import Dict, List

import numpy as np
import spacy
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per uploaded file

app = FastAPI(title="Semantic Plagiarism Detection Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading semantic model (first run downloads ~90MB, then it's cached offline)...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Rule-based sentence splitter. No trained pipeline to download, so this
# still works even if you lose wifi right before the demo.
NLP = spacy.blank("en")
NLP.add_pipe("sentencizer")

# ---------------------------------------------------------------------------
# Thresholds — tuned against the sample docs (2026-09-15)
# ---------------------------------------------------------------------------
# Semantic scores here are the CROSS-ENCODER reranked scores, which are
# calibrated and sharp. Empirical calibration on the sample pair:
#   paraphrased sentences rerank to 0.63–0.93 (all are real paraphrases)
#   unrelated sentences rerank to 0.01–0.11 (huge margin below any of these)
PARAPHRASE_THRESHOLD = 0.60   # sem >= this + low wording = paraphrased
RELATED_THRESHOLD = 0.48      # thematically related but not copy
NEAR_VERBATIM_SEM = 0.85      # both thresholds high = copied with light edits
NEAR_VERBATIM_LEX = 0.60
PARAPHRASED_LEX = 0.55        # wording overlap BELOW this flags reworded copy
TOP_K_CANDIDATES = 8          # cross-encoder reranks only top-8 pairs/sentence

# Optional: set DISABLE_CROSS_ENCODER=1 to skip the reranker (forces the
# laptop-fallback bi-encoder scores). Useful to demo the fallback path.
_CROSS_ENCODER = None
CE_AVAILABLE = os.environ.get("DISABLE_CROSS_ENCODER") != "1"


class AnalyzeRequest(BaseModel):
    source_text: str
    suspect_text: str


# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    doc = NLP(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 3]


def lexical_similarity(a: str, b: str) -> float:
    """Surface-level word/character overlap. Used alongside the semantic
    score: HIGH semantic + LOW lexical is the signature of paraphrasing —
    exactly what keyword-matching plagiarism checkers miss."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_cross_encoder():
    """Lazy-load the reranker. If it can't load (offline, disk corrupted)
    we quietly fall back to the bi-encoder scores — the demo still works."""
    global _CROSS_ENCODER, CE_AVAILABLE
    if not CE_AVAILABLE:
        return None
    if _CROSS_ENCODER is None:
        try:
            print("Loading cross-encoder reranker (distilroberta, ~90MB, cached offline)...")
            _CROSS_ENCODER = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"WARNING: cross-encoder unavailable ({exc}); using bi-encoder scores only.")
            CE_AVAILABLE = False
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


def classify(semantic_score: float, lex_score: float) -> Dict[str, str]:
    """Label a (rewritten|verbatim|related|nothing) pair. The semantic-vs-
    lexical contrast is the classifier: a big gap (high sem, low lex) means
    the meaning was copied but the words were changed — i.e. paraphrase."""
    if semantic_score >= NEAR_VERBATIM_SEM and lex_score >= NEAR_VERBATIM_LEX:
        return {"label": "Near-verbatim copy", "risk": "high"}
    if semantic_score >= PARAPHRASE_THRESHOLD:
        if lex_score < PARAPHRASED_LEX:
            return {"label": "Paraphrased (reworded copy)", "risk": "high"}
        return {"label": "Close copy (light edits)", "risk": "high"}
    if semantic_score >= RELATED_THRESHOLD:
        return {"label": "Related content", "risk": "medium"}
    return {"label": "No match", "risk": "none"}


def merge_sections(matches: List[Dict]) -> List[Dict]:
    """Group consecutive matched sentences into one section, so the report
    reads as matched PASSAGES, not a scattered list of single sentences.
    Sections are classified from their MEAN scores so the label matches the
    numbers displayed next to it (fix: previous version used one sentence's
    label on a whole section)."""
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m["suspect_idx"])
    groups, current = [], [matches[0]]
    for m in matches[1:]:
        if m["suspect_idx"] - current[-1]["suspect_idx"] <= 1:
            current.append(m)
        else:
            groups.append(current)
            current = [m]
    groups.append(current)

    sections = []
    for group in groups:
        sem = float(np.mean([g["semantic_score"] for g in group]))
        lex = float(np.mean([g["lexical_score"] for g in group]))
        cls = classify(sem, lex)
        sections.append({
            "suspect_text": " ".join(g["suspect_sentence"] for g in group),
            "source_text": " ".join(g["source_sentence"] for g in group),
            "semantic_score": round(sem, 3),
            "lexical_score": round(lex, 3),
            "gap": round(sem - lex, 3),
            "label": cls["label"],
            "risk": cls["risk"],
            "sentence_count": len(group),
        })
    return sections


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyze_documents(source_text: str, suspect_text: str) -> Dict:
    source_sents = split_sentences(source_text)
    suspect_sents = split_sentences(suspect_text)

    if not source_sents or not suspect_sents:
        return {"error": "Couldn't find readable sentences in one or both documents."}

    if _is_identical(source_text, suspect_text):
        return _identical_result(len(suspect_sents))

    source_emb = MODEL.encode(source_sents, convert_to_tensor=True)
    suspect_emb = MODEL.encode(suspect_sents, convert_to_tensor=True)
    sim_matrix = util.cos_sim(suspect_emb, source_emb).cpu().numpy()

    cross_encoder = get_cross_encoder()
    rerank_used = cross_encoder is not None

    # Stage 1 (retrieval): top-K candidate source sentences per suspect
    # sentence from the bi-encoder matrix. Keeps reranking O(N·K), not O(N²).
    candidates = []
    for i in range(len(suspect_sents)):
        top = np.argsort(sim_matrix[i])[::-1][:TOP_K_CANDIDATES]
        candidates.append([(i, int(j), sim_matrix[i][j]) for j in top])

    # Stage 2 (rerank): score just those candidate pairs with the
    # cross-encoder and keep each suspect sentence's single best match.
    if rerank_used:
        pairs = [(suspect_sents[i], source_sents[j]) for i, j, _ in sum(candidates, [])]
        ce_scores = cross_encoder.predict(pairs)
        k = 0
        for cands in candidates:
            for rank in range(len(cands)):
                cands[rank] = (cands[rank][0], cands[rank][1], float(ce_scores[k]))
                k += 1

    matches = []
    for cands in candidates:
        cands.sort(key=lambda c: c[2], reverse=True)  # best reranked score first
        i, best_j, best_score = cands[0]

        # Pairs that don't clear the "related" bar aren't matches at all —
        # keep them out so "matched sentences" means what it says.
        if best_score < RELATED_THRESHOLD:
            continue

        src_sent = source_sents[best_j]
        lex = lexical_similarity(suspect_sents[i], src_sent)
        cls = classify(best_score, lex)
        matches.append({
            "suspect_idx": i,
            "suspect_sentence": suspect_sents[i],
            "source_sentence": src_sent,
            "semantic_score": round(best_score, 3),
            "lexical_score": round(lex, 3),
            "label": cls["label"],
            "risk": cls["risk"],
        })

    return _build_report(matches, suspect_sents)


def _is_identical(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def _identical_result(sentence_count: int) -> Dict:
    return {
        "overall_similarity_percent": 100.0,
        "verdict": "Documents are identical — direct copy, not paraphrase",
        "identical_docs": True,
        "total_suspect_sentences": sentence_count,
        "matched_sentences": sentence_count,
        "matched_sections": [],
        "related_percent": 0.0,
        "rerank_used": False,
    }


def _build_report(matches: List[Dict], suspect_sents: List[str]) -> Dict:
    total_words = sum(len(s.split()) for s in suspect_sents)
    high_words = sum(
        len(m["suspect_sentence"].split())
        for m in matches if m["risk"] == "high"
    )
    related_words = sum(
        len(m["suspect_sentence"].split())
        for m in matches if m["risk"] == "medium"
    )
    overall = round((high_words / total_words) * 100, 1) if total_words else 0.0
    related = round((related_words / total_words) * 100, 1) if total_words else 0.0

    if overall >= 50:
        verdict = "High risk of plagiarism"
    elif overall >= 20:
        verdict = "Moderate similarity — review recommended"
    else:
        verdict = "Low similarity"

    return {
        "overall_similarity_percent": overall,
        "verdict": verdict,
        "identical_docs": False,
        "total_suspect_sentences": len(suspect_sents),
        "matched_sentences": len(matches),
        "matched_sections": merge_sections(matches),
        "related_percent": related,
        "rerank_used": CE_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# File extraction (uploads)
# ---------------------------------------------------------------------------
def extract_text_bytes(filename: str, content: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(BytesIO(content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif ext in ("docx",):
            import docx as python_docx
            doc = python_docx.Document(BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        elif ext == "txt":
            text = content.decode("utf-8", errors="replace")
        else:
            raise HTTPException(
                415,
                f"Unsupported file type '.{ext or '?'}'. Use .pdf, .docx or .txt "
                "— or just paste the text.",
            )
    except EOFError as exc:
        raise HTTPException(422, "That file looks empty or corrupted.") from exc

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise HTTPException(
            422,
            "No readable text in that file. It may be scanned images "
            "(not text-based) — paste the text instead.",
        )
    return text


@app.get("/api/health")
def health():
    return {"status": "ok", "cross_encoder": CE_AVAILABLE}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.source_text.strip():
        raise HTTPException(422, "Source document is empty.")
    if not req.suspect_text.strip():
        raise HTTPException(422, "Suspect document is empty.")
    if len(req.source_text.split()) < 4 or len(req.suspect_text.split()) < 4:
        raise HTTPException(
            422, "One of the documents is very short — need at least ~4 words to compare."
        )
    return analyze_documents(req.source_text, req.suspect_text)


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (limit 10 MB).")
    text = extract_text_bytes(file.filename or "document.txt", content)
    return {"filename": file.filename, "text": text}


# Serve the frontend (index.html + assets) for everything that isn't /api/*
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")