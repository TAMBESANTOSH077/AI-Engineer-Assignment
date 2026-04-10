from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from math import log, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _HAS_SKLEARN = True
except ModuleNotFoundError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _HAS_SKLEARN = False


# Shared lightweight stoplist for keyword extraction and the sklearn-free TF-IDF path.
_ENGLISH_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "are",
        "was",
        "were",
        "from",
        "into",
        "your",
        "you",
        "but",
        "not",
        "have",
        "has",
        "had",
        "will",
        "can",
        "could",
        "should",
        "would",
        "about",
        "there",
        "their",
        "then",
        "than",
        "when",
        "what",
        "where",
        "which",
        "while",
        "also",
    }
)


def sliding_window_chunks(text: str, window_chars: int, overlap_chars: int) -> List[str]:
    """
    Split text into fixed-size character windows with a sliding stride.

    Default assignment parameters approximate ~2 pages (~4000 chars) with ~1 page overlap (~2000 chars),
    so each step advances by (window_chars - overlap_chars).

    Returns non-empty stripped substrings; empty/whitespace-only input yields an empty list.
    """
    if window_chars <= 0:
        raise ValueError("window_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= window_chars:
        raise ValueError("overlap_chars must be < window_chars")

    cleaned = text.strip()
    if not cleaned:
        return []

    step = window_chars - overlap_chars
    chunks = [cleaned[i : i + window_chars] for i in range(0, len(cleaned), step)]
    return [c.strip() for c in chunks if c.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())


def _split_sentences(text: str) -> List[str]:
    """Rough sentence split on . ! ? followed by whitespace (no extra deps)."""
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def extractive_summary(text: str, max_chars: int = 280) -> str:
    """
    Pick sentences whose words are most frequent in the chunk (stopwords stripped), then join
    those sentences in original order up to max_chars. Falls back to a short prefix if needed.
    """
    if max_chars <= 0:
        return ""

    snippet = text.strip().replace("\n", " ")
    if len(snippet) <= max_chars:
        return snippet

    sentences = _split_sentences(snippet)
    if len(sentences) <= 1:
        cut = snippet[:max_chars]
        last_space = cut.rfind(" ")
        return (cut[:last_space] if last_space > max_chars // 2 else cut).rstrip() + "…"

    tokens = [w for w in _tokenize(snippet) if w not in _ENGLISH_STOP]
    if not tokens:
        cut = snippet[:max_chars]
        return cut.rstrip() + "…"

    freq: Dict[str, int] = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1

    scores: List[float] = []
    for s in sentences:
        words = [w for w in _tokenize(s) if w not in _ENGLISH_STOP]
        if not words:
            scores.append(0.0)
        else:
            scores.append(sum(freq.get(w, 0) for w in words) / len(words))

    by_score = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))

    chosen: List[int] = []
    for i in by_score:
        trial = chosen + [i]
        ordered = sorted(trial)
        joined = " ".join(sentences[j] for j in ordered)
        if len(joined) <= max_chars:
            chosen = trial

    if not chosen:
        # Best single sentence or prefix
        best_i = max(range(len(sentences)), key=lambda i: (scores[i], -len(sentences[i])))
        one = sentences[best_i]
        if len(one) <= max_chars:
            return one
        cut = one[:max_chars]
        last_space = cut.rfind(" ")
        return (cut[:last_space] if last_space > max_chars // 2 else cut).rstrip() + "…"

    ordered_chosen = sorted(chosen)
    return " ".join(sentences[j] for j in ordered_chosen)


class _SimpleTfidfVectorizer:
    """
    Minimal TF-IDF vectorizer fallback when scikit-learn isn't installed.

    - Uses a small English stoplist.
    - Produces sparse vectors as dict[int, float] keyed by term-id.
    """

    def __init__(self, max_features: int = 50_000) -> None:
        self.max_features = max_features
        self.vocab_: Dict[str, int] = {}
        self.idf_: List[float] = []

    def fit_transform(self, texts: List[str]) -> List[Dict[int, float]]:
        docs_tokens: List[List[str]] = []
        df: Dict[str, int] = {}

        for t in texts:
            toks = [w for w in _tokenize(t) if w not in _ENGLISH_STOP]
            docs_tokens.append(toks)
            for w in set(toks):
                df[w] = df.get(w, 0) + 1

        # Pick top features by document frequency, then alphabetically for stability
        terms = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_features]
        self.vocab_ = {term: i for i, (term, _) in enumerate(terms)}

        n_docs = max(1, len(texts))
        self.idf_ = [0.0] * len(self.vocab_)
        for term, i in self.vocab_.items():
            dfi = df.get(term, 0)
            # smooth idf similar in spirit to sklearn
            self.idf_[i] = log((1 + n_docs) / (1 + dfi)) + 1.0

        return [self._to_tfidf_vec(toks) for toks in docs_tokens]

    def transform(self, texts: List[str]) -> List[Dict[int, float]]:
        return [self._to_tfidf_vec([w for w in _tokenize(t) if w not in _ENGLISH_STOP]) for t in texts]

    def _to_tfidf_vec(self, toks: List[str]) -> Dict[int, float]:
        tf: Dict[int, int] = {}
        for w in toks:
            i = self.vocab_.get(w)
            if i is None:
                continue
            tf[i] = tf.get(i, 0) + 1

        if not tf:
            return {}

        # log-normalized TF
        vec: Dict[int, float] = {}
        for i, c in tf.items():
            vec[i] = (1.0 + log(c)) * self.idf_[i]
        return vec


def _cosine_similarity_sparse(query_vec: Dict[int, float], matrix: List[Dict[int, float]]) -> List[float]:
    def norm(v: Dict[int, float]) -> float:
        return sqrt(sum(x * x for x in v.values()))

    qn = norm(query_vec)
    if qn == 0.0:
        return [0.0 for _ in matrix]

    sims: List[float] = []
    for doc_vec in matrix:
        dn = norm(doc_vec)
        if dn == 0.0:
            sims.append(0.0)
            continue
        # dot over the smaller dict
        if len(query_vec) <= len(doc_vec):
            dot = sum(val * doc_vec.get(i, 0.0) for i, val in query_vec.items())
        else:
            dot = sum(val * query_vec.get(i, 0.0) for i, val in doc_vec.items())
        sims.append(dot / (qn * dn))
    return sims


class PyramidLevel(str, Enum):
    LEVEL_1_RAW = "level_1_raw"
    LEVEL_2_SUMMARY = "level_2_summary"
    LEVEL_3_CATEGORY = "level_3_category"
    LEVEL_4_KEYWORDS = "level_4_keywords"


@dataclass(frozen=True)
class PyramidWindow:
    window_index: int
    raw_text: str
    summary: str
    category: str
    keywords: List[str]

    def layer_text(self, level: PyramidLevel) -> str:
        if level == PyramidLevel.LEVEL_1_RAW:
            return self.raw_text
        if level == PyramidLevel.LEVEL_2_SUMMARY:
            return self.summary
        if level == PyramidLevel.LEVEL_3_CATEGORY:
            return self.category
        if level == PyramidLevel.LEVEL_4_KEYWORDS:
            return " ".join(self.keywords)
        raise ValueError(f"Unknown level: {level}")


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    window_index: int
    level: PyramidLevel
    content: str


class KnowledgePyramid:
    """
    KnowledgePyramid ingests documents into sliding windows and builds a 4-layer pyramid per window:
    - Level 1: Raw text
    - Level 2: Extractive summary (sentence scoring by in-chunk word frequency)
    - Level 3: Rule-based category label
    - Level 4: Keyword extraction

    Retrieval runs TF-IDF cosine similarity over *all layers across all windows* and returns the best hit.
    """

    def __init__(
        self,
        window_chars: int = 4000,
        overlap_chars: int = 2000,
        keyword_top_k: int = 12,
        summary_max_chars: int = 280,
    ) -> None:
        if window_chars <= 0:
            raise ValueError("window_chars must be > 0")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if overlap_chars >= window_chars:
            raise ValueError("overlap_chars must be < window_chars")
        if keyword_top_k <= 0:
            raise ValueError("keyword_top_k must be > 0")
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be > 0")

        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.keyword_top_k = keyword_top_k
        self.summary_max_chars = summary_max_chars

        self._windows: List[PyramidWindow] = []
        self._vectorizer = None
        self._layer_matrix = None  # scipy sparse matrix
        self._layer_index: List[Tuple[int, PyramidLevel]] = []

    @property
    def windows(self) -> Sequence[PyramidWindow]:
        return tuple(self._windows)

    def ingest_text(self, text: str) -> None:
        """
        Process the provided text into 2-page windows (~4000 chars) with 1-page overlap (~2000 chars),
        then build/refresh the pyramid and TF-IDF index.
        """
        self._windows = self._build_windows(
            sliding_window_chunks(text, self.window_chars, self.overlap_chars)
        )
        self._rebuild_index()

    def _build_windows(self, chunks: Iterable[str]) -> List[PyramidWindow]:
        windows: List[PyramidWindow] = []
        for idx, chunk in enumerate(chunks):
            raw = chunk.strip()
            if not raw:
                continue
            windows.append(
                PyramidWindow(
                    window_index=idx,
                    raw_text=raw,
                    summary=extractive_summary(raw, max_chars=self.summary_max_chars),
                    category=self._categorize_rule_based(raw),
                    keywords=self._extract_keywords(raw),
                )
            )
        return windows

    def _categorize_rule_based(self, text: str) -> str:
        """
        Simple rule-based categorizer.
        Extend keyword sets as needed for your domain.
        """
        t = text.lower()

        finance_markers = (
            "finance",
            "revenue",
            "profit",
            "loss",
            "balance sheet",
            "cash flow",
            "investment",
            "stock",
            "tax",
            "invoice",
        )
        tech_markers = (
            "api",
            "database",
            "server",
            "cloud",
            "docker",
            "kubernetes",
            "python",
            "javascript",
            "machine learning",
            "model",
            "training",
            "neural",
            "ai",
        )

        if any(m in t for m in finance_markers):
            return "Finance"
        if any(m in t for m in tech_markers):
            return "Tech"
        return "General"

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Keyword extraction via simple normalized frequency count.
        (TF-IDF is used for retrieval; keeping keyword extraction lightweight makes ingestion fast.)
        """
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
        if not tokens:
            return []

        freq: dict[str, int] = {}
        for tok in tokens:
            if tok in _ENGLISH_STOP:
                continue
            freq[tok] = freq.get(tok, 0) + 1

        return [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[: self.keyword_top_k]]

    def _iter_all_layer_texts(self) -> Iterable[Tuple[int, PyramidLevel, str]]:
        for w in self._windows:
            for level in (
                PyramidLevel.LEVEL_1_RAW,
                PyramidLevel.LEVEL_2_SUMMARY,
                PyramidLevel.LEVEL_3_CATEGORY,
                PyramidLevel.LEVEL_4_KEYWORDS,
            ):
                yield (w.window_index, level, w.layer_text(level))

    def _rebuild_index(self) -> None:
        texts: List[str] = []
        self._layer_index = []

        for window_index, level, layer_text in self._iter_all_layer_texts():
            layer_text = layer_text.strip()
            if not layer_text:
                continue
            texts.append(layer_text)
            self._layer_index.append((window_index, level))

        if not texts:
            self._vectorizer = None
            self._layer_matrix = None
            return

        if _HAS_SKLEARN:
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50_000,
            )
            self._layer_matrix = self._vectorizer.fit_transform(texts)
        else:
            self._vectorizer = _SimpleTfidfVectorizer(max_features=50_000)
            self._layer_matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str) -> Optional[RetrievalHit]:
        """
        Return the most relevant layer across all windows using TF-IDF cosine similarity.
        """
        if self._vectorizer is None or self._layer_matrix is None or not self._layer_index:
            return None

        q = query.strip()
        if not q:
            return None

        q_vec = self._vectorizer.transform([q])

        if _HAS_SKLEARN:
            sims_arr = cosine_similarity(q_vec, self._layer_matrix)[0]
            sims = [float(x) for x in sims_arr]
        else:
            # q_vec is List[Dict[int, float]] in fallback mode
            sims = _cosine_similarity_sparse(q_vec[0], self._layer_matrix)

        best_i = max(range(len(sims)), key=lambda i: sims[i])

        window_index, level = self._layer_index[best_i]
        score = float(sims[best_i])

        window = next((w for w in self._windows if w.window_index == window_index), None)
        if window is None:
            return None

        content = window.layer_text(level)
        return RetrievalHit(score=score, window_index=window_index, level=level, content=content)

    def retrieve_top_k(self, query: str, k: int = 5) -> List[RetrievalHit]:
        """
        Return up to k highest-scoring layers (deduplicated by window_index + level), sorted by score descending.
        """
        if self._vectorizer is None or self._layer_matrix is None or not self._layer_index:
            return []

        q = query.strip()
        if not q or k <= 0:
            return []

        q_vec = self._vectorizer.transform([q])

        if _HAS_SKLEARN:
            sims_arr = cosine_similarity(q_vec, self._layer_matrix)[0]
            sims = [float(x) for x in sims_arr]
        else:
            sims = _cosine_similarity_sparse(q_vec[0], self._layer_matrix)

        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        hits: List[RetrievalHit] = []
        seen: set[Tuple[int, PyramidLevel]] = set()

        for i in order:
            if len(hits) >= k:
                break
            window_index, level = self._layer_index[i]
            key = (window_index, level)
            if key in seen:
                continue
            seen.add(key)
            window = next((w for w in self._windows if w.window_index == window_index), None)
            if window is None:
                continue
            content = window.layer_text(level)
            hits.append(
                RetrievalHit(
                    score=float(sims[i]),
                    window_index=window_index,
                    level=level,
                    content=content,
                )
            )

        return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Pyramid ingestion + TF-IDF retrieval")
    parser.add_argument(
        "--document",
        type=Path,
        default=Path("document.txt"),
        help="UTF-8 text file to ingest (default: document.txt)",
    )
    parser.add_argument("--window-chars", type=int, default=4000, help="Sliding window size in characters")
    parser.add_argument("--overlap-chars", type=int, default=2000, help="Overlap between consecutive windows")
    parser.add_argument("--keyword-top-k", type=int, default=12, help="Top keywords per window (level 4)")
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=280,
        help="Max length for level-2 extractive summary",
    )
    args = parser.parse_args()

    text = args.document.read_text(encoding="utf-8")

    kp = KnowledgePyramid(
        window_chars=args.window_chars,
        overlap_chars=args.overlap_chars,
        keyword_top_k=args.keyword_top_k,
        summary_max_chars=args.summary_max_chars,
    )
    kp.ingest_text(text)

    query = input("Enter query: ")
    hit = kp.retrieve(query)

    if hit is None:
        print("No results (empty index).")
    else:
        print(f"\nBest match: window={hit.window_index}, level={hit.level.value}, score={hit.score:.4f}")
        print("Content:", hit.content)