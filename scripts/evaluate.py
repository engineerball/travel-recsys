"""Offline evaluation: Hit@K, Recall@K, NDCG@K for the two-tower retrieval model.

For each held-out val interaction (last ``val_fraction`` of interactions, chronological):
  1. Encode the user with the interaction's context via the user tower.
  2. Query the FAISS index for that item's type with top-K candidates.
  3. Check whether the held-out item appears in the retrieved list.

Metrics:
  Hit@K   — fraction of val interactions where the item is in top-K
  NDCG@K  — normalised discounted CG (rewards higher rank within top-K)
  Both reported overall and per item type.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --top-k 10 20 50 100
    uv run python scripts/evaluate.py --checkpoint checkpoints/epoch_022
"""

from __future__ import annotations

import argparse
import glob as _glob
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    preprocess_articles,
    preprocess_attractions,
    preprocess_accommodations,
    preprocess_events,
)
from src.data.schema import (
    ItemType,
    NUM_ARTICLE_TYPES,
    NUM_ATTRACTION_SUBCATS,
    NUM_ITEM_CATEGORIES,
)
from src.models.two_tower import TwoTowerModel
from src.serving.retrieval import MultiTypeIndex, RetrievalPipeline
from src.training.dataset import (
    MAX_AMENITY_INDICES,
    MAX_CAT_PREF,
    MAX_CATEGORY_INDICES,
    MAX_PROVINCE_PREF,
    MAX_SUBCAT_INDICES,
    MAX_TRAVEL_STYLES,
    MAX_TRAVEL_THEMES,
    _pad_seq,
    _to_float32_stack,
)

import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_SUBDIRS: Dict[ItemType, str] = {
    ItemType.ATTRACTION: "destinations",
    ItemType.ACCOMMODATION: "accommodations",
    ItemType.EVENT: "activities",
    ItemType.ARTICLE: "articles",
}

_ID_COLS: Dict[ItemType, str] = {
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
# Data loading (mirrors train.py)
# ---------------------------------------------------------------------------

def _load_raw(subdir: str) -> pd.DataFrame:
    files = sorted(_glob.glob(f"{subdir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {subdir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _load_item_features(
    item_type: ItemType,
    raw_dir: str,
    processed_dir: Optional[str] = None,
) -> pd.DataFrame:
    subdir = os.path.join(raw_dir, _RAW_SUBDIRS[item_type])
    raw_df = _load_raw(subdir)
    df = _PREPROCESSORS[item_type](raw_df)

    if processed_dir:
        type_name = item_type.name.lower()
        emb_path = os.path.join(processed_dir, f"{type_name}_text_embeds.npy")
        ids_path = os.path.join(processed_dir, f"{type_name}_item_ids.npy")
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            embs = np.load(emb_path)
            ids = np.load(ids_path, allow_pickle=True)
            id_col = _ID_COLS[item_type]
            id_to_row = {str(iid): i for i, iid in enumerate(ids)}
            text_embeds = []
            for iid in df[id_col]:
                row = id_to_row.get(str(iid))
                text_embeds.append(
                    embs[row] if row is not None else np.zeros(256, dtype=np.float32)
                )
            df["text_embed"] = text_embeds

    df = df.set_index(_ID_COLS[item_type])
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# User feature extraction (single interaction row → feature dict)
# ---------------------------------------------------------------------------

def _build_user_features(
    user_row: pd.Series,
    day_of_week: float,
    hour_of_day: float,
) -> Dict[str, np.ndarray]:
    """Convert a user_features_df row + context to a batched (B=1) feature dict."""

    def _pad(seq, max_len: int) -> np.ndarray:
        return _pad_seq(seq, max_len).reshape(1, -1)

    dow = day_of_week
    hod = hour_of_day

    return {
        "age_norm":
            np.array([[float(user_row["age_norm"])]], dtype=np.float32).reshape(1),
        "home_country_id":
            np.array([int(user_row["home_country_id"])], dtype=np.int32),
        "travel_style_indices":
            _pad(user_row.get("travel_style_indices"), MAX_TRAVEL_STYLES),
        "travel_theme_indices":
            _pad(user_row.get("travel_theme_indices"), MAX_TRAVEL_THEMES),
        "category_pref_indices":
            _pad(user_row.get("category_pref_indices"), MAX_CAT_PREF),
        "category_interaction_history":
            np.asarray(
                user_row.get("category_interaction_history",
                             np.zeros(NUM_ITEM_CATEGORIES, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(1, -1),
        "subcat_affinity":
            np.asarray(
                user_row.get("subcat_affinity",
                             np.zeros(NUM_ATTRACTION_SUBCATS, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(1, -1),
        "article_type_affinity":
            np.asarray(
                user_row.get("article_type_affinity",
                             np.zeros(NUM_ARTICLE_TYPES, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(1, -1),
        "province_pref_indices":
            _pad(user_row.get("province_pref_indices", []), MAX_PROVINCE_PREF),
        "context_day_sin":
            np.array([math.sin(2 * math.pi * dow / 7)], dtype=np.float32),
        "context_day_cos":
            np.array([math.cos(2 * math.pi * dow / 7)], dtype=np.float32),
        "context_hour_sin":
            np.array([math.sin(2 * math.pi * hod / 24)], dtype=np.float32),
        "context_hour_cos":
            np.array([math.cos(2 * math.pi * hod / 24)], dtype=np.float32),
    }


def _to_tf(d: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    return {k: tf.constant(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ndcg_at_k(rank: Optional[int], k: int) -> float:
    """NDCG@K for a single relevant item at ``rank`` (1-indexed). 0 if not found."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _hit_at_k(rank: Optional[int], k: int) -> float:
    return 1.0 if (rank is not None and rank <= k) else 0.0


# ---------------------------------------------------------------------------
# Build model + load weights
# ---------------------------------------------------------------------------

def _init_model(
    checkpoint_path: str,
    dropout_rate: float,
    temperature: float,
    sample_batch,
) -> TwoTowerModel:
    """Initialise TwoTowerModel, build weights via one dummy forward pass, load ckpt."""
    model = TwoTowerModel(dropout_rate=dropout_rate, temperature=temperature)

    # Build all sub-layers by running one eval step
    user_feats, item_feats, item_types, _ = sample_batch
    for itype in ItemType:
        mask = tf.equal(item_types, int(itype))
        if not tf.reduce_any(mask):
            continue
        u_b = {k: tf.boolean_mask(v, mask) for k, v in user_feats.items()}
        i_b = {k: tf.boolean_mask(v, mask) for k, v in item_feats.items()}
        model({"user": u_b, "item": i_b}, training=False, item_type=itype)

    model.load_weights(checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-tower retrieval model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--generated-dir", default="data/generated")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--index-dir", default="models/index")
    parser.add_argument("--checkpoint", default="checkpoints/best/weights.weights.h5",
                        help="Path to model weights file")
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20, 50, 100],
                        help="K values for Hit@K and NDCG@K")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-eval", type=int, default=None,
                        help="Cap number of val interactions evaluated (for speed)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})
    dropout_rate = float(model_cfg.get("dropout_rate", 0.3))
    temperature = float(model_cfg.get("temperature", 0.07))

    # --- Load interactions + user features ---
    print("Loading data …")
    interactions = pd.read_parquet(
        os.path.join(args.generated_dir, "interactions.parquet")
    )
    user_features_df = pd.read_parquet(
        os.path.join(args.generated_dir, "user_features.parquet")
    )
    user_features_df = user_features_df.set_index("user_id")

    # Chronological val split — same as StratifiedInteractionDataset
    N = len(interactions)
    n_val = max(4, int(N * args.val_fraction))
    val_df = interactions.iloc[N - n_val:].reset_index(drop=True)

    if args.max_eval:
        val_df = val_df.head(args.max_eval)

    print(f"  Val interactions: {len(val_df)}  (from {N} total)")

    # --- Load item features ---
    print("Loading item features …")
    processed_dir = args.processed_dir if os.path.isdir(args.processed_dir) else None
    item_features: Dict[ItemType, pd.DataFrame] = {}
    for itype in ItemType:
        item_features[itype] = _load_item_features(itype, args.raw_dir, processed_dir)
        print(f"  {itype.name}: {len(item_features[itype])} items")

    # --- Load FAISS index ---
    print(f"\nLoading FAISS index from {args.index_dir} …")
    multi_index = MultiTypeIndex(dim=64)
    multi_index.load(args.index_dir)
    print(f"  Index sizes: { {t.name: n for t, n in multi_index.index_sizes.items()} }")

    # --- Build model + load weights ---
    print(f"\nLoading model weights from {args.checkpoint} …")

    # Need a real sample batch to trigger Keras weight initialisation
    from src.training.dataset import StratifiedInteractionDataset
    _dummy_ds = StratifiedInteractionDataset(
        interactions_df=interactions.iloc[:max(512, N - n_val)],
        user_features_df=user_features_df,
        item_features_dict=item_features,
    )
    _sample_batch = next(iter(_dummy_ds.build(batch_size=64, split="val")))

    model = _init_model(args.checkpoint, dropout_rate, temperature, _sample_batch)
    print("  Weights loaded.")

    # --- Eval loop ---
    print(f"\nEvaluating {len(val_df)} interactions …")

    K_values = sorted(args.top_k)
    max_k = max(K_values)

    # Accumulators: overall and per-type
    hit_sums: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    ndcg_sums: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)

    ts = pd.to_datetime(val_df["timestamp"])

    for i, (_, row) in enumerate(val_df.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(val_df)} …")

        uid = row["user_id"]
        item_id = str(row["item_id"])
        item_type = ItemType(int(row["item_type"]))

        if uid not in user_features_df.index:
            continue

        user_row = user_features_df.loc[uid]
        dow = float(ts.iloc[i].dayofweek)
        hod = float(ts.iloc[i].hour)

        user_feats_np = _build_user_features(user_row, dow, hod)
        user_feats_tf = _to_tf(user_feats_np)

        user_emb = model.get_user_embedding(user_feats_tf, training=False)
        user_emb_np = user_emb.numpy().squeeze(0)  # [64]

        # Query only the matching item-type index
        retrieved_ids, _ = multi_index.query_type(user_emb_np, item_type, top_k=max_k)
        retrieved_ids_str = [str(rid) for rid in retrieved_ids]

        # Find rank (1-indexed)
        rank: Optional[int] = None
        for pos, rid in enumerate(retrieved_ids_str, start=1):
            if rid == item_id:
                rank = pos
                break

        type_key = item_type.name.lower()
        for key in ("overall", type_key):
            counts[key] += 1
            for k in K_values:
                hit_sums[key][k] += _hit_at_k(rank, k)
                ndcg_sums[key][k] += _ndcg_at_k(rank, k)

    # --- Report ---
    print(f"\n{'─' * 60}")
    print(f"{'Metric':<20}  " + "  ".join(f"@{k:<5}" for k in K_values))
    print(f"{'─' * 60}")

    report_keys = ["overall"] + [itype.name.lower() for itype in ItemType]
    for key in report_keys:
        n = counts[key]
        if n == 0:
            continue
        label = f"{key} (n={n})"
        for metric, sums in [("Hit", hit_sums), ("NDCG", ndcg_sums)]:
            vals = "  ".join(
                f"{sums[key][k] / n:.4f}" for k in K_values
            )
            print(f"{metric:<6} {label:<30}  {vals}")
        print()

    print(f"{'─' * 60}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
