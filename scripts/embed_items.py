"""Pre-compute Gemma text embeddings for all item types.

Loads raw item data → applies preprocess_* to extract text_for_embed →
batch-encodes via sentence-transformers (google/embeddinggemma-300m, MRL 256-dim) →
saves float32 [N, 256] embeddings and item ID arrays per type.

Outputs (written to --processed-dir, default data/processed/):
    {type}_text_embeds.npy   float32 [N, 256]
    {type}_item_ids.npy      object  [N]

Usage:
    HF_TOKEN=hf_... uv run python scripts/embed_items.py
    HF_TOKEN=hf_... uv run python scripts/embed_items.py --batch-size 64 --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Allow running as `python scripts/embed_items.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    load_and_preprocess,
    preprocess_articles,
    preprocess_attractions,
    preprocess_accommodations,
    preprocess_events,
)
from src.data.schema import ItemType


# Raw subdirectory name per ItemType
_RAW_SUBDIRS: Dict[ItemType, str] = {
    ItemType.ATTRACTION: "destinations",
    ItemType.ACCOMMODATION: "accommodations",
    ItemType.EVENT: "activities",
    ItemType.ARTICLE: "articles",
}

# ID column in preprocessed DataFrame per ItemType
_ID_COLS: Dict[ItemType, str] = {
    ItemType.ATTRACTION: "place_id",
    ItemType.ACCOMMODATION: "place_id",
    ItemType.EVENT: "event_id",
    ItemType.ARTICLE: "article_id",
}


def _load_raw_df(item_type: ItemType, raw_dir: str) -> pd.DataFrame:
    """Load all parquet files from the type-specific subdirectory."""
    import glob as _glob
    subdir = os.path.join(raw_dir, _RAW_SUBDIRS[item_type])
    files = sorted(_glob.glob(f"{subdir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {subdir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _preprocess(item_type: ItemType, raw_df: pd.DataFrame) -> pd.DataFrame:
    dispatch = {
        ItemType.ATTRACTION: preprocess_attractions,
        ItemType.ACCOMMODATION: preprocess_accommodations,
        ItemType.EVENT: preprocess_events,
        ItemType.ARTICLE: preprocess_articles,
    }
    return dispatch[item_type](raw_df)


def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    hf_token: str,
    truncate_dim: int = 256,
) -> np.ndarray:
    """Batch-encode texts with sentence-transformers, return float32 [N, truncate_dim]."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: uv add sentence-transformers")

    print(f"  Loading model {model_name} …")
    model = SentenceTransformer(
        model_name,
        token=hf_token,
        truncate_dim=truncate_dim,
    )

    print(f"  Encoding {len(texts)} texts (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )
    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute Gemma text embeddings for items.")
    parser.add_argument("--raw-dir", default="data/raw", help="Root raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Output directory")
    parser.add_argument("--model", default="google/embeddinggemma-300m", help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--truncate-dim", type=int, default=256, help="MRL truncation dimension")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["attraction", "accommodation", "event", "article"],
        default=["attraction", "accommodation", "event", "article"],
        help="Item types to embed (default: all)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("WARNING: HF_TOKEN not set — may fail on gated models")

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

        raw_df = _load_raw_df(item_type, args.raw_dir)
        print(f"  Loaded {len(raw_df)} rows from raw data")

        prepped = _preprocess(item_type, raw_df)
        print(f"  Preprocessed → {len(prepped)} rows")

        texts = prepped["text_for_embed"].fillna("").tolist()
        item_ids = prepped[_ID_COLS[item_type]].values

        embeddings = embed_texts(
            texts,
            model_name=args.model,
            batch_size=args.batch_size,
            hf_token=hf_token,
            truncate_dim=args.truncate_dim,
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        emb_path = os.path.join(args.processed_dir, f"{type_name}_text_embeds.npy")
        ids_path = os.path.join(args.processed_dir, f"{type_name}_item_ids.npy")
        np.save(emb_path, embeddings)
        np.save(ids_path, item_ids)
        print(f"  Saved → {emb_path}  {ids_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
