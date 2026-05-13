"""Generate synthetic user interaction data and compute user behavior features.

Pipeline:
  1. Load config.yaml (behavior.personas, behavior.num_users, etc.)
  2. Load raw item DataFrames (4 types)
  3. Load + preprocess user profiles
  4. Run InteractionGenerator → interaction table
  5. Compute behavior features from interaction history
  6. Join behavior features onto user profiles
  7. Save outputs:
       data/generated/interactions.parquet   — user-item interaction log
       data/generated/user_features.parquet  — user profiles + behavior features

Usage:
    uv run python scripts/generate_behavior.py
    uv run python scripts/generate_behavior.py --n-users 5000 --seed 99
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.behavior_generator import (
    augment_user_profiles,
    BehaviorDataset,
    InteractionGenerator,
    compute_behavior_features,
    load_personas_from_config,
)
from src.data.preprocessing import preprocess_users
from src.data.schema import ItemType


_RAW_SUBDIRS: Dict[ItemType, str] = {
    ItemType.ATTRACTION: "destinations",
    ItemType.ACCOMMODATION: "accommodations",
    ItemType.EVENT: "activities",
    ItemType.ARTICLE: "articles",
}


def _load_raw(subdir: str) -> pd.DataFrame:
    import glob as _glob
    files = sorted(_glob.glob(f"{subdir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {subdir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic interaction data.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--n-users", type=int, default=None,
                        help="Max users to use (capped at available rows in user_profiles)")
    parser.add_argument("--augment-users", type=int, default=None,
                        help="Augment user profiles to this total count via bootstrap perturbation")
    parser.add_argument("--ipp-min", type=int, default=None,
                        help="Min interactions per user (overrides config)")
    parser.add_argument("--ipp-max", type=int, default=None,
                        help="Max interactions per user (overrides config). "
                             "Increase this when n-users is capped, e.g. --ipp-max 200")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    behavior_cfg = config.get("behavior", {})
    n_users = args.n_users or behavior_cfg.get("num_users", 1000)
    ipp_min = args.ipp_min or int(behavior_cfg.get("interactions_per_user_min", 5))
    ipp_max = args.ipp_max or int(behavior_cfg.get("interactions_per_user_max", 50))
    pop_power = float(behavior_cfg.get("popularity_power", 0.7))

    personas = load_personas_from_config(config)
    print(f"Loaded {len(personas)} personas: {[p.name for p in personas]}")

    # --- Load raw data ---
    print("\nLoading raw data …")
    raw_dfs: Dict[ItemType, pd.DataFrame] = {}
    for itype, subdir_name in _RAW_SUBDIRS.items():
        subdir = os.path.join(args.raw_dir, subdir_name)
        raw_dfs[itype] = _load_raw(subdir)
        print(f"  {itype.name}: {len(raw_dfs[itype])} rows")

    user_raw = _load_raw(os.path.join(args.raw_dir, "user_profiles"))
    print(f"  Users: {len(user_raw)} rows")

    # Cap to available rows
    if n_users and n_users < len(user_raw):
        user_raw = user_raw.head(n_users).reset_index(drop=True)

    # Augment user profiles if requested
    if args.augment_users and args.augment_users > len(user_raw):
        user_raw = augment_user_profiles(user_raw, args.augment_users, seed=args.seed)
        print(f"  Augmented to {len(user_raw)} users")

    print(f"  Using {len(user_raw)} users (ipp={ipp_min}–{ipp_max})")

    # --- Generate interactions ---
    print(f"\nGenerating interactions (ipp={ipp_min}–{ipp_max}, power={pop_power}) …")
    generator = InteractionGenerator(
        users_df=user_raw,
        raw_item_dfs=raw_dfs,
        personas=personas,
        popularity_power=pop_power,
        interactions_per_user=(ipp_min, ipp_max),
        seed=args.seed,
    )
    interactions = generator.generate()
    print(f"  Generated {len(interactions)} interactions across "
          f"{interactions['user_id'].nunique()} users")
    signal_counts = interactions["signal"].value_counts().to_dict()
    print(f"  Signal distribution: {signal_counts}")

    # --- Compute behavior features ---
    print("\nComputing user behavior features …")
    from src.data.preprocessing import preprocess_articles, preprocess_attractions, preprocess_accommodations, preprocess_events
    import glob as _glob

    attr_files = sorted(_glob.glob(os.path.join(args.raw_dir, "destinations", "*.parquet")))
    evt_files = sorted(_glob.glob(os.path.join(args.raw_dir, "activities", "*.parquet")))
    accom_files = sorted(_glob.glob(os.path.join(args.raw_dir, "accommodations", "*.parquet")))
    art_files = sorted(_glob.glob(os.path.join(args.raw_dir, "articles", "*.parquet")))
    attr_raw = pd.concat([pd.read_parquet(f) for f in attr_files], ignore_index=True)
    evt_raw = pd.concat([pd.read_parquet(f) for f in evt_files], ignore_index=True)
    accom_raw = pd.concat([pd.read_parquet(f) for f in accom_files], ignore_index=True) if accom_files else None
    art_raw = pd.concat([pd.read_parquet(f) for f in art_files], ignore_index=True) if art_files else None

    attraction_prepped = preprocess_attractions(attr_raw)
    event_prepped = preprocess_events(evt_raw)
    accom_prepped = preprocess_accommodations(accom_raw) if accom_raw is not None else None
    article_prepped = preprocess_articles(art_raw) if art_raw is not None else None

    behavior_features = compute_behavior_features(
        interactions, attraction_prepped, event_prepped, accom_prepped, article_prepped
    )
    print(f"  Behavior features for {len(behavior_features)} users")

    # --- Build enriched user features ---
    print("\nBuilding enriched user features …")
    user_profile_features = preprocess_users(user_raw)
    user_profile_features = user_profile_features.set_index("user_id")

    # Join behavior features; users with no interactions get zero-fill
    from src.data.schema import NUM_ARTICLE_TYPES, NUM_ATTRACTION_SUBCATS, NUM_ITEM_CATEGORIES, NUM_PROVINCES
    zero_cat = np.zeros(NUM_ITEM_CATEGORIES, dtype=np.float32)
    zero_sub = np.zeros(NUM_ATTRACTION_SUBCATS, dtype=np.float32)
    zero_prov = np.zeros(NUM_PROVINCES, dtype=np.float32)
    zero_art = np.zeros(NUM_ARTICLE_TYPES, dtype=np.float32)

    user_profile_features["category_pref_indices"] = [
        behavior_features.loc[uid, "category_pref_indices"]
        if uid in behavior_features.index else []
        for uid in user_profile_features.index
    ]
    user_profile_features["category_interaction_history"] = [
        behavior_features.loc[uid, "category_interaction_history"]
        if uid in behavior_features.index else zero_cat.copy()
        for uid in user_profile_features.index
    ]
    user_profile_features["subcat_affinity"] = [
        behavior_features.loc[uid, "subcat_affinity"]
        if uid in behavior_features.index else zero_sub.copy()
        for uid in user_profile_features.index
    ]
    user_profile_features["province_affinity"] = [
        behavior_features.loc[uid, "province_affinity"]
        if uid in behavior_features.index else zero_prov.copy()
        for uid in user_profile_features.index
    ]
    user_profile_features["province_pref_indices"] = [
        behavior_features.loc[uid, "province_pref_indices"]
        if uid in behavior_features.index else []
        for uid in user_profile_features.index
    ]
    user_profile_features["article_type_affinity"] = [
        behavior_features.loc[uid, "article_type_affinity"]
        if uid in behavior_features.index else zero_art.copy()
        for uid in user_profile_features.index
    ]

    # --- Save outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    interactions_path = os.path.join(args.output_dir, "interactions.parquet")
    users_path = os.path.join(args.output_dir, "user_features.parquet")

    BehaviorDataset(interactions).save(interactions_path)
    user_profile_features.reset_index().to_parquet(users_path, index=False)

    print("\nSaved:")
    print(f"  {interactions_path}  ({len(interactions)} rows)")
    print(f"  {users_path}  ({len(user_profile_features)} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
