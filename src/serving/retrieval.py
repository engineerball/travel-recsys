"""FAISS-backed ANN retrieval for two-tower inference.

FAISSRetriever    — single-type ANN index (inner product on L2-normalized embeddings)
MultiTypeIndex    — one FAISSRetriever per ItemType + merged query
RetrievalPipeline — end-to-end: user features → top-K items via model + index
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import tensorflow as tf

from src.data.schema import ItemType


# ---------------------------------------------------------------------------
# FAISSRetriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """Thin wrapper around a FAISS flat inner-product index.

    Embeddings must be L2-normalized before insertion. Under that constraint
    inner product equals cosine similarity.

    Args:
        dim: embedding dimension (default 64, matches ``EmbeddingConfig.output_dim``).
        use_gpu: move index to GPU 0 if True (requires faiss-gpu).

    Example::

        retriever = FAISSRetriever()
        retriever.build_index(embeddings, item_ids)
        ids, scores = retriever.query(user_emb, top_k=100)
    """

    def __init__(self, dim: int = 64, use_gpu: bool = False) -> None:
        self.dim = dim
        self.use_gpu = use_gpu
        self._index: Optional[faiss.Index] = None
        self._item_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: np.ndarray,
    ) -> None:
        """Build FAISS index from pre-computed item embeddings.

        Args:
            embeddings: float32 [N, dim] — must be L2-normalized.
            item_ids:   array [N] — original item IDs (any integer/str type);
                        returned by ``query`` to identify results.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings [N, {self.dim}], got {embeddings.shape}"
            )

        self._item_ids = np.asarray(item_ids)

        index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(embeddings)
        self._index = index

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        user_embedding: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ANN search for nearest items to the user embedding.

        Args:
            user_embedding: float32 [D] or [B, D] — L2-normalized.
            top_k: number of results per query.

        Returns:
            item_ids: array [B, top_k] (or [top_k] if single query).
            scores:   float32 [B, top_k] cosine similarities in [-1, 1].
        """
        if self._index is None or self._item_ids is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q = np.asarray(user_embedding, dtype=np.float32)
        squeeze = q.ndim == 1
        if squeeze:
            q = q[None]  # [1, D]

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)  # both [B, k]

        # Map FAISS sequential indices back to original item IDs
        retrieved_ids = self._item_ids[indices]  # [B, k]

        if squeeze:
            return retrieved_ids[0], scores[0]
        return retrieved_ids, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str, prefix: str = "index") -> None:
        """Save index and item ID mapping to *directory*.

        Creates two files:
            ``{directory}/{prefix}.faiss`` — FAISS binary index
            ``{directory}/{prefix}_ids.npy`` — numpy ID array
        """
        if self._index is None or self._item_ids is None:
            raise RuntimeError("Nothing to save — build_index() first.")
        os.makedirs(directory, exist_ok=True)
        cpu_index = (
            faiss.index_gpu_to_cpu(self._index) if self.use_gpu else self._index
        )
        faiss.write_index(cpu_index, os.path.join(directory, f"{prefix}.faiss"))
        np.save(os.path.join(directory, f"{prefix}_ids.npy"), self._item_ids)

    def load(self, directory: str, prefix: str = "index") -> None:
        """Load index and item ID mapping from *directory*."""
        index_path = os.path.join(directory, f"{prefix}.faiss")
        ids_path = os.path.join(directory, f"{prefix}_ids.npy")
        self._index = faiss.read_index(index_path)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
        self._item_ids = np.load(ids_path, allow_pickle=True)

    @property
    def ntotal(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0


# ---------------------------------------------------------------------------
# MultiTypeIndex
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Single retrieved item from the ANN index."""
    item_id: object       # original ID (int or str)
    item_type: ItemType
    score: float


class MultiTypeIndex:
    """Manages one FAISSRetriever per ItemType.

    Supports per-type queries and merged cross-type queries where results
    are interleaved by score (highest first).

    Args:
        dim:     embedding dimension (default 64).
        use_gpu: pass through to each FAISSRetriever.
    """

    def __init__(self, dim: int = 64, use_gpu: bool = False) -> None:
        self.dim = dim
        self.use_gpu = use_gpu
        self._retrievers: Dict[ItemType, FAISSRetriever] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings_dict: Dict[ItemType, np.ndarray],
        ids_dict: Dict[ItemType, np.ndarray],
    ) -> None:
        """Build one index per item type.

        Args:
            embeddings_dict: ``{ItemType: float32 [N_type, dim]}`` — L2-normalized.
            ids_dict:        ``{ItemType: array [N_type]}`` — original item IDs.
        """
        for item_type, embeddings in embeddings_dict.items():
            retriever = FAISSRetriever(dim=self.dim, use_gpu=self.use_gpu)
            retriever.build_index(embeddings, ids_dict[item_type])
            self._retrievers[item_type] = retriever

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_type(
        self,
        user_embedding: np.ndarray,
        item_type: ItemType,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query a single item-type index.

        Returns:
            item_ids: array [top_k]
            scores:   float32 [top_k]
        """
        retriever = self._retrievers.get(item_type)
        if retriever is None:
            raise KeyError(f"No index for item type: {item_type}")
        return retriever.query(user_embedding, top_k=top_k)

    def query(
        self,
        user_embedding: np.ndarray,
        top_k_per_type: int = 50,
        top_k_total: int = 100,
    ) -> List[RetrievalResult]:
        """Query all item-type indexes and return merged top-K results.

        Retrieves ``top_k_per_type`` candidates from each index, merges
        all candidates, and returns the top ``top_k_total`` by score.

        Args:
            user_embedding: float32 [D] — L2-normalized.
            top_k_per_type: candidates per item type before merging.
            top_k_total:    final list length after cross-type re-ranking.

        Returns:
            List of ``RetrievalResult`` sorted by score descending.
        """
        candidates: List[RetrievalResult] = []
        for item_type, retriever in self._retrievers.items():
            ids, scores = retriever.query(user_embedding, top_k=top_k_per_type)
            for item_id, score in zip(ids, scores):
                candidates.append(
                    RetrievalResult(item_id=item_id, item_type=item_type, score=float(score))
                )

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:top_k_total]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save all per-type indexes to *directory*.

        Each type gets its own prefix (e.g. ``attraction.faiss``).
        """
        for item_type, retriever in self._retrievers.items():
            retriever.save(directory, prefix=item_type.name.lower())

    def load(self, directory: str, item_types: Optional[List[ItemType]] = None) -> None:
        """Load per-type indexes from *directory*.

        Args:
            directory:  path written by ``save()``.
            item_types: which types to load; defaults to all four.
        """
        if item_types is None:
            item_types = list(ItemType)
        for item_type in item_types:
            retriever = FAISSRetriever(dim=self.dim, use_gpu=self.use_gpu)
            retriever.load(directory, prefix=item_type.name.lower())
            self._retrievers[item_type] = retriever

    @property
    def index_sizes(self) -> Dict[ItemType, int]:
        """Number of vectors indexed per item type."""
        return {t: r.ntotal for t, r in self._retrievers.items()}


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """End-to-end retrieval: user features → ranked item list.

    Wraps a trained ``TwoTowerModel`` and a ``MultiTypeIndex``.
    Typical usage::

        pipeline = RetrievalPipeline(model, index)
        results = pipeline.retrieve(user_inputs, top_k=100)

    Args:
        model:        trained ``TwoTowerModel`` (weights loaded).
        multi_index:  populated ``MultiTypeIndex``.
    """

    def __init__(self, model, multi_index: MultiTypeIndex) -> None:
        self.model = model
        self.index = multi_index

    def retrieve(
        self,
        user_inputs: dict,
        top_k_per_type: int = 50,
        top_k_total: int = 100,
    ) -> List[RetrievalResult]:
        """Encode user features and fetch top-K items from the ANN index.

        Args:
            user_inputs:    dict matching ``UserFeatureSchema`` with batch dim 1.
            top_k_per_type: candidates retrieved per item type.
            top_k_total:    final ranked list length.

        Returns:
            List of ``RetrievalResult`` sorted by score descending.
        """
        user_emb: tf.Tensor = self.model.get_user_embedding(user_inputs, training=False)
        user_emb_np = user_emb.numpy().squeeze(0)  # [D]
        return self.index.query(user_emb_np, top_k_per_type=top_k_per_type, top_k_total=top_k_total)

    @classmethod
    def build_index_from_model(
        cls,
        model,
        item_features_dict: Dict[ItemType, dict],
        item_ids_dict: Dict[ItemType, np.ndarray],
        batch_size: int = 512,
        dim: int = 64,
        use_gpu: bool = False,
    ) -> "RetrievalPipeline":
        """Encode all items with the model and build a MultiTypeIndex.

        Args:
            model:              trained ``TwoTowerModel``.
            item_features_dict: ``{ItemType: feature_dict}`` where each value is
                                a dict of numpy arrays (same format as training inputs)
                                for ALL items of that type. Arrays have leading N dim.
            item_ids_dict:      ``{ItemType: np.ndarray [N]}`` — original item IDs.
            batch_size:         items per forward pass.
            dim:                embedding dimension.
            use_gpu:            pass to ``MultiTypeIndex``.

        Returns:
            Fully built ``RetrievalPipeline`` ready for inference.
        """
        embeddings_dict: Dict[ItemType, np.ndarray] = {}

        for item_type, features in item_features_dict.items():
            n = next(iter(features.values())).shape[0]
            all_embs: List[np.ndarray] = []

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = {k: v[start:end] for k, v in features.items()}
                batch_tf = {k: tf.constant(v) for k, v in batch.items()}
                emb = model.get_item_embedding(batch_tf, item_type, training=False)
                all_embs.append(emb.numpy())

            embeddings_dict[item_type] = np.concatenate(all_embs, axis=0)

        multi_index = MultiTypeIndex(dim=dim, use_gpu=use_gpu)
        multi_index.build(embeddings_dict, item_ids_dict)

        pipeline = cls(model, multi_index)
        return pipeline
