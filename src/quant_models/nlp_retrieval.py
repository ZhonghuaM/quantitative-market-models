"""Lightweight retrieval tools for source-grounded financial text analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text: str, chunk_words: int = 160, overlap_words: int = 30) -> list[str]:
    """Split text into overlapping word chunks."""

    words = re.findall(r"\S+", text)
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("invalid chunking parameters")
    chunks = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


@dataclass
class RetrievalResult:
    rank: int
    score: float
    text: str


class TfidfRetriever:
    """Small dependency-light retrieval engine for public filings or research notes."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._matrix = None
        self._chunks: list[str] = []

    def fit(self, chunks: list[str]) -> "TfidfRetriever":
        if not chunks:
            raise ValueError("chunks cannot be empty")
        self._chunks = chunks
        self._matrix = self.vectorizer.fit_transform(chunks)
        return self

    def query(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        if self._matrix is None:
            raise RuntimeError("retriever must be fitted before querying")
        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [
            RetrievalResult(rank=i + 1, score=float(scores[idx]), text=self._chunks[idx])
            for i, idx in enumerate(order)
        ]


def source_grounded_brief(question: str, results: list[RetrievalResult]) -> str:
    """Create a concise extractive brief from retrieved chunks."""

    query_terms = {term.lower() for term in re.findall(r"[A-Za-z]{4,}", question)}
    sentences: list[str] = []
    for result in results:
        for sentence in re.split(r"(?<=[.!?])\s+", result.text):
            lower = sentence.lower()
            if any(term in lower for term in query_terms):
                sentences.append(sentence.strip())
                break
    if not sentences:
        sentences = [result.text[:280].strip() for result in results[:2]]
    return " ".join(sentences[:3])


def retrieval_results_frame(question: str, results: list[RetrievalResult]) -> pd.DataFrame:
    """Convert retrieval results into a report-friendly table."""

    return pd.DataFrame(
        [
            {
                "question": question,
                "rank": result.rank,
                "score": result.score,
                "text": result.text,
            }
            for result in results
        ]
    )
