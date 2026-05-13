"""Local prediction demo: load model and show recommendations for a real user.

Shows:
  - User profile (country, travel style, travel theme)
  - Their interaction history (signal-weighted, most engaged items first)
  - Top-K recommendations from the model
  - Match marker if a recommendation also appears in full interaction history

Usage:
    uv run python scripts/predict.py
    uv run python scripts/predict.py --user-id U0042
    uv run python scripts/predict.py --top-k 20 --item-type attraction
"""

from __future__ import annotations

import argparse
import glob as _glob
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from src.data.schema import (
    HOME_COUNTRIES,
    TRAVEL_STYLES,
    TRAVEL_THEMES,
    ItemType,
    NUM_ARTICLE_TYPES,
    NUM_ATTRACTION_SUBCATS,
    NUM_ITEM_CATEGORIES,
)
from src.models.two_tower import TwoTowerModel
from src.serving.retrieval import MultiTypeIndex
from src.training.dataset import (
    MAX_CAT_PREF,
    MAX_PROVINCE_PREF,
    MAX_TRAVEL_STYLES,
    MAX_TRAVEL_THEMES,
    _pad_seq,
)

# ---------------------------------------------------------------------------
# Article type display names
# ---------------------------------------------------------------------------
_ARTICLE_TYPE_NAMES: Dict[int, str] = {
    1: "Destination", 2: "Province", 3: "See & Do", 4: "Event & Festival",
    5: "Stay", 6: "Food", 7: "Shop", 8: "Trip (Itinerary)",
    9: "Essential", 10: "Activity", 11: "Campaign", 12: "News & Update",
}

_SIGNAL_WEIGHT = {"view": 1.0, "click": 3.0}


# ---------------------------------------------------------------------------
# Name lookup builder
# ---------------------------------------------------------------------------

def _load_raw(subdir: str) -> pd.DataFrame:
    files = sorted(_glob.glob(f"{subdir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {subdir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_name_lookup(raw_dir: str) -> Dict[ItemType, Dict[str, str]]:
    """Return {ItemType: {item_id_str: display_name}}."""
    lookup: Dict[ItemType, Dict[str, str]] = {}

    # Attraction
    df = _load_raw(os.path.join(raw_dir, "destinations"))
    lookup[ItemType.ATTRACTION] = {
        str(r["placeId"]): str(r.get("name", r["placeId"]))
        for _, r in df.iterrows()
    }

    # Accommodation
    df = _load_raw(os.path.join(raw_dir, "accommodations"))
    lookup[ItemType.ACCOMMODATION] = {
        str(r["placeId"]): str(r.get("name", r["placeId"]))
        for _, r in df.iterrows()
    }

    # Event
    df = _load_raw(os.path.join(raw_dir, "activities"))
    lookup[ItemType.EVENT] = {
        str(r["eventId"]): str(r.get("name", r["eventId"]))
        for _, r in df.iterrows()
    }

    # Article
    df = _load_raw(os.path.join(raw_dir, "articles"))
    lookup[ItemType.ARTICLE] = {
        str(r["id"]): str(r.get("title", r["id"]))
        for _, r in df.iterrows()
    }

    return lookup


# ---------------------------------------------------------------------------
# User feature extraction (same as evaluate.py)
# ---------------------------------------------------------------------------

def build_user_features(user_row: pd.Series, dow: float = 2.0, hod: float = 14.0) -> Dict[str, tf.Tensor]:
    def _pad(seq, max_len: int) -> np.ndarray:
        return _pad_seq(seq, max_len).reshape(1, -1)

    feats = {
        "age_norm":
            np.array([float(user_row["age_norm"])], dtype=np.float32),
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
        "context_day_sin": np.array([math.sin(2 * math.pi * dow / 7)], dtype=np.float32),
        "context_day_cos": np.array([math.cos(2 * math.pi * dow / 7)], dtype=np.float32),
        "context_hour_sin": np.array([math.sin(2 * math.pi * hod / 24)], dtype=np.float32),
        "context_hour_cos": np.array([math.cos(2 * math.pi * hod / 24)], dtype=np.float32),
    }
    return {k: tf.constant(v) for k, v in feats.items()}


# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------

def init_model(checkpoint: str, dropout_rate: float, temperature: float,
               user_features_df: pd.DataFrame, item_features_dict,
               interactions: pd.DataFrame) -> TwoTowerModel:
    from src.data.preprocessing import (
        preprocess_articles, preprocess_attractions,
        preprocess_accommodations, preprocess_events,
    )
    from src.training.dataset import StratifiedInteractionDataset

    ds = StratifiedInteractionDataset(
        interactions_df=interactions.head(512),
        user_features_df=user_features_df,
        item_features_dict=item_features_dict,
    )
    sample_batch = next(iter(ds.build(batch_size=64, split="val")))

    model = TwoTowerModel(dropout_rate=dropout_rate, temperature=temperature)
    user_feats, item_feats, item_types, _ = sample_batch
    for itype in ItemType:
        mask = tf.equal(item_types, int(itype))
        if not tf.reduce_any(mask):
            continue
        u_b = {k: tf.boolean_mask(v, mask) for k, v in user_feats.items()}
        i_b = {k: tf.boolean_mask(v, mask) for k, v in item_feats.items()}
        model({"user": u_b, "item": i_b}, training=False, item_type=itype)

    model.load_weights(checkpoint)
    return model


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _decode_indices(indices, vocab: List[str]) -> List[str]:
    return [vocab[i] for i in indices if 0 <= i < len(vocab)]


def print_user_profile(uid: str, user_row: pd.Series) -> None:
    country_id = int(user_row.get("home_country_id", 0))
    country = HOME_COUNTRIES[country_id] if country_id < len(HOME_COUNTRIES) else "?"

    style_indices = user_row.get("travel_style_indices", [])
    styles = _decode_indices([i for i in style_indices if i >= 0], TRAVEL_STYLES)

    theme_indices = user_row.get("travel_theme_indices", [])
    themes = _decode_indices([i for i in theme_indices if i >= 0], TRAVEL_THEMES)

    print(f"\n{'═' * 60}")
    print(f"  USER: {uid}")
    print(f"{'═' * 60}")
    print(f"  Country    : {country}")
    print(f"  Age (norm) : {user_row.get('age_norm', '?'):.2f}")
    print(f"  Styles     : {', '.join(styles) or '-'}")
    print(f"  Themes     : {', '.join(themes) or '-'}")


def print_history(user_interactions: pd.DataFrame,
                  name_lookup: Dict[ItemType, Dict[str, str]],
                  top_n: int = 15) -> None:
    df = user_interactions.copy()
    df["weight"] = df["signal"].map(_SIGNAL_WEIGHT).fillna(1.0)
    df = df.sort_values("weight", ascending=False).head(top_n)

    print(f"\n  ── Interaction History (top {top_n}, by signal weight) ──")
    print(f"  {'Signal':<10} {'Type':<14} {'Item name'}")
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        itype = ItemType(int(row["item_type"]))
        item_id = str(row["item_id"])
        name = name_lookup[itype].get(item_id, item_id)[:45]
        print(f"  {row['signal']:<10} {itype.name.lower():<14} {name}")


def print_recommendations(results, name_lookup: Dict[ItemType, Dict[str, str]],
                          history_ids: set, item_type_filter: Optional[str]) -> None:
    type_label = f" [{item_type_filter.upper()}]" if item_type_filter else " [ALL TYPES]"
    print(f"\n  ── Model Recommendations{type_label} ──")
    print(f"  {'Rank':<5} {'Score':<7} {'Type':<14} {'Match':<6} {'Item name'}")
    print(f"  {'-'*65}")

    for rank, r in enumerate(results, start=1):
        itype = r.item_type
        item_id = str(r.item_id)
        name = name_lookup[itype].get(item_id, item_id)[:40]
        match = "✓" if item_id in history_ids else ""
        print(f"  {rank:<5} {r.score:<7.4f} {itype.name.lower():<14} {match:<6} {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run local prediction for a user.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--generated-dir", default="data/generated")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--index-dir", default="models/index")
    parser.add_argument("--checkpoint", default="checkpoints/best/weights.weights.h5")
    parser.add_argument("--user-id", default=None, help="User ID to query (default: most-interacted)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--item-type", default=None,
                        choices=["attraction", "accommodation", "event", "article"],
                        help="Filter results to one item type (default: all types)")
    parser.add_argument("--dow", type=float, default=2.0, help="Day of week context (0=Mon)")
    parser.add_argument("--hod", type=float, default=14.0, help="Hour of day context (0-23)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})

    # --- Load data ---
    print("Loading data …")
    interactions = pd.read_parquet(os.path.join(args.generated_dir, "interactions.parquet"))
    user_features_df = pd.read_parquet(os.path.join(args.generated_dir, "user_features.parquet"))
    user_features_df = user_features_df.set_index("user_id")

    # --- Build name lookup ---
    print("Building item name lookup …")
    name_lookup = build_name_lookup(args.raw_dir)

    # --- Pick user ---
    if args.user_id:
        uid = args.user_id
        if uid not in user_features_df.index:
            print(f"ERROR: user {uid} not in user_features. Available sample:")
            counts = interactions["user_id"].value_counts().head(5)
            for u, n in counts.items():
                print(f"  {u}: {n} interactions")
            sys.exit(1)
    else:
        uid = interactions["user_id"].value_counts().idxmax()
        print(f"Auto-selected most-interacted user: {uid}")

    user_row = user_features_df.loc[uid]
    user_interactions = interactions[interactions["user_id"] == uid]
    history_ids = set(user_interactions["item_id"].astype(str))

    # --- Load item features (for model build) ---
    from src.data.preprocessing import (
        preprocess_articles, preprocess_attractions,
        preprocess_accommodations, preprocess_events,
    )
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
    processed_dir = args.processed_dir if os.path.isdir(args.processed_dir) else None
    item_features: Dict[ItemType, pd.DataFrame] = {}
    for itype in ItemType:
        subdir = os.path.join(args.raw_dir, _RAW_SUBDIRS[itype])
        raw_df = _load_raw(subdir)
        df = _PREPROCESSORS[itype](raw_df)
        if processed_dir:
            type_name = itype.name.lower()
            emb_path = os.path.join(processed_dir, f"{type_name}_text_embeds.npy")
            ids_path = os.path.join(processed_dir, f"{type_name}_item_ids.npy")
            if os.path.exists(emb_path) and os.path.exists(ids_path):
                embs = np.load(emb_path)
                ids = np.load(ids_path, allow_pickle=True)
                id_to_row = {str(iid): i for i, iid in enumerate(ids)}
                id_col = _ID_COLS[itype]
                df["text_embed"] = [
                    embs[id_to_row[str(iid)]] if str(iid) in id_to_row
                    else np.zeros(256, dtype=np.float32)
                    for iid in df[id_col]
                ]
        df = df.set_index(_ID_COLS[itype])
        df.index = df.index.astype(str)
        df = df[~df.index.duplicated(keep="first")]
        item_features[itype] = df

    # --- Load model ---
    print(f"Loading model from {args.checkpoint} …")
    model = init_model(
        args.checkpoint,
        dropout_rate=float(model_cfg.get("dropout_rate", 0.3)),
        temperature=float(model_cfg.get("temperature", 0.07)),
        user_features_df=user_features_df,
        item_features_dict=item_features,
        interactions=interactions,
    )

    # --- Load FAISS index ---
    print(f"Loading FAISS index from {args.index_dir} …")
    multi_index = MultiTypeIndex(dim=64)
    multi_index.load(args.index_dir)

    # --- Encode user ---
    user_feats_tf = build_user_features(user_row, dow=args.dow, hod=args.hod)
    user_emb = model.get_user_embedding(user_feats_tf, training=False)
    user_emb_np = user_emb.numpy().squeeze(0)

    # --- Retrieve ---
    if args.item_type:
        itype_map = {
            "attraction": ItemType.ATTRACTION, "accommodation": ItemType.ACCOMMODATION,
            "event": ItemType.EVENT, "article": ItemType.ARTICLE,
        }
        target_type = itype_map[args.item_type]
        retrieved_ids, scores = multi_index.query_type(user_emb_np, target_type, top_k=args.top_k)
        from src.serving.retrieval import RetrievalResult
        results = [
            RetrievalResult(item_id=iid, item_type=target_type, score=float(s))
            for iid, s in zip(retrieved_ids, scores)
        ]
    else:
        results = multi_index.query(
            user_emb_np,
            top_k_per_type=args.top_k,
            top_k_total=args.top_k,
        )

    # --- Display ---
    print_user_profile(uid, user_row)
    print_history(user_interactions, name_lookup, top_n=15)
    print_recommendations(results, name_lookup, history_ids, args.item_type)

    matched = sum(1 for r in results if str(r.item_id) in history_ids)
    print(f"\n  Match rate: {matched}/{len(results)} recommendations appear in full history")
    print(f"  Context: day={int(args.dow)} (0=Mon), hour={int(args.hod)}:00\n")


if __name__ == "__main__":
    main()
