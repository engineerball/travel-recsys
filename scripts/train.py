"""Training entry point for the two-tower retrieval model.

Pipeline:
  1. Load config.yaml
  2. Load processed data (interactions + user features from generate_behavior.py)
  3. Load item features (preprocess_* + join text embeddings from embed_items.py)
  4. Build tf.data train/val datasets (StratifiedInteractionDataset)
  5. Initialize TwoTowerModel + TwoTowerTrainer
  6. Train (joint loss across all 4 item types)
  7. Save towers to --output-dir/towers/
  8. Build FAISS index from item embeddings → save to --output-dir/index/

Requires:
  - data/generated/interactions.parquet    (from generate_behavior.py)
  - data/generated/user_features.parquet   (from generate_behavior.py)
  - data/raw/{destinations,accommodations,activities,articles}/*.parquet
  - data/processed/{type}_text_embeds.npy  (from embed_items.py, optional)
  - data/processed/{type}_item_ids.npy     (from embed_items.py, optional)

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 30 --batch-size 256 --output-dir models
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    preprocess_articles,
    preprocess_attractions,
    preprocess_accommodations,
    preprocess_events,
)
from src.data.schema import ItemType
from src.models.two_tower import TwoTowerModel
from src.serving.retrieval import RetrievalPipeline
from src.training.dataset import StratifiedInteractionDataset
from src.training.trainer import TwoTowerTrainer


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
    """Preprocess item DataFrame and optionally join pre-computed text embeddings."""
    subdir = os.path.join(raw_dir, _RAW_SUBDIRS[item_type])
    raw_df = _load_raw(subdir)
    df = _PREPROCESSORS[item_type](raw_df)

    if processed_dir:
        type_name = item_type.name.lower()
        emb_path = os.path.join(processed_dir, f"{type_name}_text_embeds.npy")
        ids_path = os.path.join(processed_dir, f"{type_name}_item_ids.npy")
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            embs = np.load(emb_path)          # [N, 256]
            ids = np.load(ids_path, allow_pickle=True)
            id_col = _ID_COLS[item_type]
            id_to_row = {str(iid): i for i, iid in enumerate(ids)}
            text_embeds = []
            for iid in df[id_col]:
                row = id_to_row.get(str(iid))
                text_embeds.append(embs[row] if row is not None else np.zeros(256, dtype=np.float32))
            df["text_embed"] = text_embeds
            print(f"    text_embed joined from {emb_path}")
        else:
            print(f"    WARNING: no text embeds found at {emb_path} — using zeros")

    # Set item_id as index; deduplicate in case multiple parquet files overlap
    df = df.set_index(_ID_COLS[item_type])
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]
    return df


def _item_features_for_index(
    item_type: ItemType,
    df: pd.DataFrame,
) -> tuple[dict, np.ndarray]:
    """Convert preprocessed DataFrame to numpy dict + ID array for index building.

    Returns:
        features_dict: {str: np.ndarray} with leading N dimension
        item_ids: np.ndarray [N] of original IDs
    """
    from src.training.dataset import (
        MAX_AMENITY_INDICES,
        MAX_CATEGORY_INDICES,
        MAX_SUBCAT_INDICES,
        _pad_column,
        _to_float32_stack,
    )
    TEXT_DIM = 256
    N = len(df)
    item_ids = df.index.values.astype(str)

    base: dict = {
        "item_type_id": np.full(N, int(item_type), dtype=np.int32),
        "text_embed": (
            _to_float32_stack(df["text_embed"], TEXT_DIM)
            if "text_embed" in df.columns
            else np.zeros((N, TEXT_DIM), dtype=np.float32)
        ),
    }

    if item_type == ItemType.ATTRACTION:
        base.update({
            "sub_category_indices": _pad_column(df["sub_category_indices"], MAX_SUBCAT_INDICES),
            "province_id": df["province_id"].values.astype(np.int32),
            "days_open_vector": np.stack(df["days_open_vector"].values).astype(np.float32),
            "is_free": df["is_free"].values.astype(np.float32),
            "log_view_count": df["log_view_count"].values.astype(np.float32),
        })
    elif item_type == ItemType.ACCOMMODATION:
        base.update({
            "amenity_indices": _pad_column(df["amenity_indices"], MAX_AMENITY_INDICES),
            "province_id": df["province_id"].values.astype(np.int32),
            "price_tier_id": df["price_tier_id"].values.astype(np.int32),
            "is_price_missing": df["is_price_missing"].values.astype(np.float32),
            "star_rating_norm": df["star_rating_norm"].values.astype(np.float32),
            "log_view_count": df["log_view_count"].values.astype(np.float32),
        })
    elif item_type == ItemType.EVENT:
        base.update({
            "category_indices": _pad_column(df["category_indices"], MAX_CATEGORY_INDICES),
            "province_id": df["province_id"].values.astype(np.int32),
            "duration_days_norm": df["duration_days_norm"].values.astype(np.float32),
            "month_sin": df["month_sin"].values.astype(np.float32),
            "month_cos": df["month_cos"].values.astype(np.float32),
        })
    elif item_type == ItemType.ARTICLE:
        base.update({
            "article_type_id": df["article_type_id"].values.astype(np.int32),
            "pub_month_sin": df["pub_month_sin"].values.astype(np.float32),
            "pub_month_cos": df["pub_month_cos"].values.astype(np.float32),
        })

    return base, item_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--generated-dir", default="data/generated")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory with pre-computed text embeddings (optional)")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for towers + FAISS index")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience epochs (default from config)")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip FAISS index building after training")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    epochs = args.epochs or int(train_cfg.get("epochs", 20))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 512))
    steps_per_epoch = args.steps_per_epoch or int(train_cfg.get("steps_per_epoch", 50))
    patience = args.patience or int(train_cfg.get("patience", 10))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    temperature = float(model_cfg.get("temperature", 0.07))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.1))

    # --- Load interactions + user features ---
    print("Loading generated data …")
    interactions_path = os.path.join(args.generated_dir, "interactions.parquet")
    users_path = os.path.join(args.generated_dir, "user_features.parquet")

    if not os.path.exists(interactions_path):
        raise FileNotFoundError(
            f"{interactions_path} not found — run scripts/generate_behavior.py first"
        )

    interactions = pd.read_parquet(interactions_path)
    user_features_df = pd.read_parquet(users_path)
    user_features_df = user_features_df.set_index("user_id")

    print(f"  Interactions: {len(interactions)}")
    print(f"  Users:        {len(user_features_df)}")

    # --- Load item features ---
    print("\nLoading item features …")
    item_features: Dict[ItemType, pd.DataFrame] = {}
    processed_dir = args.processed_dir if os.path.isdir(args.processed_dir) else None
    if processed_dir is None:
        print("  (no processed dir — text embeddings will be zeros)")

    for itype in ItemType:
        print(f"  {itype.name} …")
        item_features[itype] = _load_item_features(itype, args.raw_dir, processed_dir)
        print(f"    {len(item_features[itype])} items")

    # --- Build tf.data datasets ---
    print("\nBuilding tf.data datasets …")
    dataset = StratifiedInteractionDataset(
        interactions_df=interactions,
        user_features_df=user_features_df,
        item_features_dict=item_features,
    )
    train_ds = dataset.build(batch_size=batch_size, split="train")
    val_ds = dataset.build(batch_size=batch_size, split="val")
    print(f"  batch_size={batch_size}, epochs={epochs}, steps_per_epoch={steps_per_epoch}")

    # --- Initialize model ---
    print("\nInitializing TwoTowerModel …")
    model = TwoTowerModel(dropout_rate=dropout_rate, temperature=temperature)

    trainer_config = {
        "learning_rate": lr,
        "temperature": temperature,
    }

    trainer = TwoTowerTrainer(
        model=model,
        config=trainer_config,
        checkpoint_dir=args.checkpoint_dir,
    )

    # --- Train ---
    print(f"\nTraining for {epochs} epochs …\n")
    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        patience=patience,
    )

    # --- Save towers ---
    towers_dir = os.path.join(args.output_dir, "towers")
    print(f"\nSaving towers to {towers_dir} …")
    trainer.save_towers(towers_dir)

    # --- Build FAISS index ---
    if not args.skip_index:
        print("\nBuilding FAISS index …")
        item_features_for_index: Dict[ItemType, dict] = {}
        item_ids_for_index: Dict[ItemType, np.ndarray] = {}
        for itype, df in item_features.items():
            feat_dict, ids = _item_features_for_index(itype, df)
            item_features_for_index[itype] = feat_dict
            item_ids_for_index[itype] = ids
            print(f"  {itype.name}: {len(ids)} items")

        pipeline = RetrievalPipeline.build_index_from_model(
            model=model,
            item_features_dict=item_features_for_index,
            item_ids_dict=item_ids_for_index,
            batch_size=512,
        )

        index_dir = os.path.join(args.output_dir, "index")
        print(f"Saving index to {index_dir} …")
        pipeline.index.save(index_dir)
        print(f"  Index sizes: {pipeline.index.index_sizes}")

    print("\nTraining complete.")
    final_train = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val = history["val_loss"][-1] if history.get("val_loss") else float("nan")
    print(f"  Final train_loss={final_train:.4f}  val_loss={final_val:.4f}")


if __name__ == "__main__":
    main()
