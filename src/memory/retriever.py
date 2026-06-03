from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import AppConfig, get_config
from src.memory.experiment_log import ExperimentLog
from src.memory.knowledge_store import Finding, KnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    kind: str
    id: str
    text: str
    score: float


def _tfidf_similarity(query: str, corpus: list[str], max_features: int = 256) -> np.ndarray:
    """Fallback TF-IDF cosine similarity."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    mat = vectorizer.fit_transform(corpus + [query])
    doc_mat = mat[:-1]
    q_vec = mat[-1]
    return cosine_similarity(q_vec, doc_mat).flatten()


class MemoryRetriever:
    """Retrieval over findings and experiment interpretations.

    Uses GPU-accelerated sentence-transformer embeddings when available,
    falling back to TF-IDF otherwise.
    """

    def __init__(
        self,
        knowledge: KnowledgeStore,
        experiments: ExperimentLog,
        config: AppConfig | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.experiments = experiments
        self.config = config or get_config()
        self._use_gpu = self.config.use_gpu_embeddings
        if self._use_gpu:
            from src.gpu_embeddings import is_gpu_embeddings_available
            if not is_gpu_embeddings_available():
                logger.info("sentence-transformers not available, falling back to TF-IDF")
                self._use_gpu = False

    def _compute_similarities(self, query: str, corpus: list[str]) -> np.ndarray:
        if self._use_gpu:
            from src.gpu_embeddings import gpu_cosine_similarity
            try:
                return gpu_cosine_similarity(
                    query, corpus,
                    model_name=self.config.embedding_model,
                    device=self.config.device,
                )
            except Exception as e:
                logger.warning("GPU embedding failed (%s), falling back to TF-IDF", e)
        return _tfidf_similarity(query, corpus, self.config.embedding_max_features)

    def retrieve_for_hypothesis(self, query: str, k: int = 5) -> list[RetrievalHit]:
        corpus: list[str] = []
        meta: list[tuple[str, str]] = []
        for f in self.knowledge.all_findings():
            text = f.statement
            corpus.append(text)
            meta.append(("finding", f.id))
        for ex in self.experiments.recent(50):
            interp = ex.interpretation.get("summary", "")
            if interp:
                corpus.append(str(interp))
                meta.append(("experiment", ex.id))
        if not corpus:
            return []

        sims = self._compute_similarities(query, corpus)
        order = np.argsort(-sims)[:k]
        hits: list[RetrievalHit] = []
        for i in order:
            idx = int(i)
            kind, eid = meta[idx]
            hits.append(RetrievalHit(kind=kind, id=eid, text=corpus[idx], score=float(sims[idx])))
        return hits

    def similar_past_experiments(self, hypothesis_statement: str, k: int = 3) -> list[RetrievalHit]:
        corpus: list[str] = []
        ids: list[str] = []
        for ex in self.experiments.recent(100):
            plan = ex.plan.get("steps", [])
            summary = " ".join(str(s.get("description", "")) for s in plan)
            text = ex.interpretation.get("summary", "") + " " + summary
            corpus.append(text.strip())
            ids.append(ex.id)
        if not corpus:
            return []

        sims = self._compute_similarities(hypothesis_statement, corpus)
        order = np.argsort(-sims)[:k]
        return [
            RetrievalHit(kind="experiment", id=ids[int(i)], text=corpus[int(i)], score=float(sims[int(i)]))
            for i in order
        ]
