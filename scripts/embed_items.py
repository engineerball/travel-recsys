"""Pre-compute text embeddings for all item types.

Three backends (select via --backend):

  api    (default) — Google AI text-embedding-004 via REST API.
                     Fast: ~100 texts/call, no local GPU needed.
                     Requires: GOOGLE_API_KEY env var.

  vertex           — Vertex AI text-embedding-004 via SDK.
                     Auth via Application Default Credentials (ADC).
                     Best for Vertex AI Workbench — service account Just Works.
                     Requires: --gcp-project  (and optionally --gcp-location).
                     250 texts/call (higher than api backend).

  local            — sentence-transformers with any HuggingFace model.
                     Requires: HF_TOKEN env var + local GPU recommended.
                     Default model: google/embeddinggemma-300m.

All backends output float32 [N, 256] L2-normalized embeddings.

Outputs (written to --processed-dir, default data/processed/):
    {type}_text_embeds.npy   float32 [N, 256]
    {type}_item_ids.npy      object  [N]

Usage:
    # Vertex AI Workbench (service account auth, no env vars needed)
    uv run python scripts/embed_items.py --backend vertex --gcp-project my-project

    # Local machine (Google API key)
    GOOGLE_API_KEY=... uv run python scripts/embed_items.py

    # Local model (GPU recommended)
    HF_TOKEN=hf_... uv run python scripts/embed_items.py --backend local
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    preprocess_articles,
    preprocess_attractions,
    preprocess_accommodations,
    preprocess_events,
)
from src.data.schema import ItemType


_RAW_SUBDIRS = {
    ItemType.ATTRACTION: "destinations",
    ItemType.ACCOMMODATION: "accommodations",
    ItemType.EVENT: "activities",
    ItemType.ARTICLE: "articles",
}

_ID_COLS = {
    ItemType.ATTRACTION: "place_id",
    ItemType.ACCOMMODATION: "place_id",
    ItemType.EVENT: "event_id",
    ItemType.ARTICLE: "article_id",
}

_PREPROCESSORS = {
    ItemType.ATTRACTION: preprocess_attractions,
    ItemType.ACCOMMODATION: preprocess_accommodations,
    ItemType.EVENT: preprocess_events,
    ItemType.ARTICLE: preprocess_articles,
}


# ---------------------------------------------------------------------------
# Backend: Google AI API
# ---------------------------------------------------------------------------

def embed_texts_api(
    texts: List[str],
    model: str = "models/text-embedding-004",
    truncate_dim: int = 256,
    batch_size: int = 100,      # API limit per call
    task_type: str = "RETRIEVAL_DOCUMENT",
    requests_per_minute: int = 1500,  # free-tier limit
) -> np.ndarray:
    """Embed texts via Google AI text-embedding-004 (MRL, no local GPU needed).

    Batches at 100 texts/call (API max). Respects RPM limit with light throttling.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: uv add google-generativeai")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)

    n = len(texts)
    all_embeddings: List[np.ndarray] = []
    min_delay = 60.0 / requests_per_minute  # seconds between calls

    print(f"  API model: {model}  output_dim={truncate_dim}")
    print(f"  {n} texts → {(n + batch_size - 1) // batch_size} API calls …")

    t0 = time.time()
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]

        result = genai.embed_content(
            model=model,
            content=batch,
            task_type=task_type,
            output_dimensionality=truncate_dim,
        )
        batch_embs = np.array(result["embedding"], dtype=np.float32)

        # L2-normalize (API returns raw vectors)
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        batch_embs = batch_embs / np.maximum(norms, 1e-8)

        all_embeddings.append(batch_embs)

        done = min(start + batch_size, n)
        elapsed = time.time() - t0
        print(f"  [{done}/{n}]  {elapsed:.1f}s", end="\r")

        # Throttle to stay under RPM limit
        time.sleep(min_delay)

    print()
    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Backend: Vertex AI SDK (service account / ADC)
# ---------------------------------------------------------------------------

def embed_texts_vertex(
    texts: List[str],
    project: str,
    location: str = "us-central1",
    model_name: str = "text-embedding-004",
    truncate_dim: int = 256,
    batch_size: int = 250,      # Vertex AI limit per call
) -> np.ndarray:
    """Embed texts via Vertex AI SDK using Application Default Credentials.

    On Vertex AI Workbench the instance service account is used automatically.
    Locally, run `gcloud auth application-default login` first.
    """
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    except ImportError:
        raise ImportError("Run: uv add google-cloud-aiplatform")

    print(f"  Vertex AI project={project} location={location} model={model_name}")
    vertexai.init(project=project, location=location)
    model = TextEmbeddingModel.from_pretrained(model_name)

    n = len(texts)
    all_embeddings: List[np.ndarray] = []

    print(f"  {n} texts → {(n + batch_size - 1) // batch_size} Vertex AI calls …")
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        inputs = [
            TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT")
            for t in batch
        ]
        kwargs = {"output_dimensionality": truncate_dim}
        results = model.get_embeddings(inputs, **kwargs)
        batch_embs = np.array([r.values for r in results], dtype=np.float32)

        # L2-normalize
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        batch_embs = batch_embs / np.maximum(norms, 1e-8)

        all_embeddings.append(batch_embs)
        done = min(start + batch_size, n)
        print(f"  [{done}/{n}]  {time.time() - t0:.1f}s", end="\r")

    print()
    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Backend: local sentence-transformers
# ---------------------------------------------------------------------------

def embed_texts_local(
    texts: List[str],
    model_name: str = "google/embeddinggemma-300m",
    batch_size: int = 128,
    hf_token: str = "",
    truncate_dim: int = 256,
) -> np.ndarray:
    """Embed texts via sentence-transformers. GPU auto-detected."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: uv add sentence-transformers")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}  model: {model_name}  batch_size={batch_size}")

    if device == "cpu":
        print("  WARNING: CPU encoding is slow — consider --backend api or a GPU instance")

    st_model = SentenceTransformer(
        model_name,
        token=hf_token or None,
        truncate_dim=truncate_dim,
        device=device,
    )

    print(f"  Encoding {len(texts)} texts …")
    embeddings = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings for items.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--backend", choices=["api", "vertex", "local"], default="api",
        help="api=Google AI key; vertex=Vertex AI ADC (service account); local=HuggingFace",
    )
    # API backend options
    parser.add_argument("--api-model", default="models/text-embedding-004",
                        help="Google AI embedding model")
    parser.add_argument("--api-batch-size", type=int, default=100,
                        help="Texts per API call (max 100)")
    # Vertex AI backend options
    parser.add_argument("--gcp-project", default=None,
                        help="GCP project ID (required for --backend vertex)")
    parser.add_argument("--gcp-location", default="us-central1",
                        help="Vertex AI region (default: us-central1)")
    parser.add_argument("--vertex-model", default="text-embedding-004",
                        help="Vertex AI embedding model name")
    parser.add_argument("--vertex-batch-size", type=int, default=250,
                        help="Texts per Vertex AI call (max 250)")
    # Local backend options
    parser.add_argument("--local-model", default="google/embeddinggemma-300m",
                        help="HuggingFace model ID for local backend")
    parser.add_argument("--local-batch-size", type=int, default=128,
                        help="Batch size for local encoding (increase for GPU)")
    # Shared
    parser.add_argument("--truncate-dim", type=int, default=256,
                        help="MRL output dimension")
    parser.add_argument(
        "--types", nargs="+",
        choices=["attraction", "accommodation", "event", "article"],
        default=["attraction", "accommodation", "event", "article"],
    )
    args = parser.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)

    type_map = {
        "attraction": ItemType.ATTRACTION,
        "accommodation": ItemType.ACCOMMODATION,
        "event": ItemType.EVENT,
        "article": ItemType.ARTICLE,
    }
    requested = [type_map[t] for t in args.types]

    for item_type in requested:
        type_name = item_type.name.lower()
        print(f"\n=== {type_name.upper()} ===")

        import glob as _glob
        subdir = os.path.join(args.raw_dir, _RAW_SUBDIRS[item_type])
        files = sorted(_glob.glob(f"{subdir}/*.parquet"))
        if not files:
            print(f"  SKIP: no parquet files in {subdir}")
            continue

        raw_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"  Loaded {len(raw_df)} rows")

        prepped = _PREPROCESSORS[item_type](raw_df)
        texts = prepped["text_for_embed"].fillna("").tolist()
        item_ids = prepped[_ID_COLS[item_type]].values
        print(f"  {len(texts)} texts to embed")

        if args.backend == "api":
            embeddings = embed_texts_api(
                texts,
                model=args.api_model,
                truncate_dim=args.truncate_dim,
                batch_size=args.api_batch_size,
            )
        elif args.backend == "vertex":
            if not args.gcp_project:
                parser.error("--gcp-project is required for --backend vertex")
            embeddings = embed_texts_vertex(
                texts,
                project=args.gcp_project,
                location=args.gcp_location,
                model_name=args.vertex_model,
                truncate_dim=args.truncate_dim,
                batch_size=args.vertex_batch_size,
            )
        else:
            embeddings = embed_texts_local(
                texts,
                model_name=args.local_model,
                batch_size=args.local_batch_size,
                hf_token=os.environ.get("HF_TOKEN", ""),
                truncate_dim=args.truncate_dim,
            )

        print(f"  Embeddings: {embeddings.shape}  dtype={embeddings.dtype}")

        emb_path = os.path.join(args.processed_dir, f"{type_name}_text_embeds.npy")
        ids_path = os.path.join(args.processed_dir, f"{type_name}_item_ids.npy")
        np.save(emb_path, embeddings)
        np.save(ids_path, item_ids)
        print(f"  Saved → {emb_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
