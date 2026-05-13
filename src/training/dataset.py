"""tf.data pipeline for two-tower training.

StratifiedInteractionDataset  — pre-materialises features, yields stratified batches
stratified_batch_sampler       — standalone generator over interaction DataFrame
"""

from __future__ import annotations

import math
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.schema import (
    ItemType,
    NUM_ARTICLE_TYPES,
    NUM_ATTRACTION_SUBCATS,
    NUM_ITEM_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Padding constants for variable-length multi-hot index features
# ---------------------------------------------------------------------------

MAX_TRAVEL_STYLES: int = 5
MAX_TRAVEL_THEMES: int = 15
MAX_CAT_PREF: int = 10        # top category preference indices per user
MAX_PROVINCE_PREF: int = 3    # top province indices per user
MAX_SUBCAT_INDICES: int = 10  # attraction sub-categories
MAX_CATEGORY_INDICES: int = 3  # event categories
MAX_AMENITY_INDICES: int = 24  # accommodation amenities (full vocab fits)

# Default signal → float weight mapping (matches config.yaml)
DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {"view": 1.0, "click": 3.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_seq(seq, max_len: int, pad_val: int = -1) -> np.ndarray:
    """Pad/truncate a sequence to fixed length with pad_val."""
    arr = np.full(max_len, pad_val, dtype=np.int32)
    if seq is not None:
        seq = list(seq)
        n = min(len(seq), max_len)
        if n > 0:
            arr[:n] = seq[:n]
    return arr


def _pad_column(series: pd.Series, max_len: int) -> np.ndarray:
    """Convert a Series of lists to a padded 2D int32 array [N, max_len]."""
    return np.array([_pad_seq(x, max_len) for x in series], dtype=np.int32)


def _to_float32_stack(series: pd.Series, dim: int) -> np.ndarray:
    """Stack a Series of array-like into float32 [N, dim], using zeros for None."""
    rows = []
    for x in series:
        if x is None:
            rows.append(np.zeros(dim, dtype=np.float32))
        else:
            rows.append(np.asarray(x, dtype=np.float32))
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Standalone stratified batch sampler
# ---------------------------------------------------------------------------

def stratified_batch_sampler(
    df: pd.DataFrame,
    batch_size: int,
    item_types: Optional[List[str]] = None,
    type_col: str = "item_type",
    seed: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Yield stratified mini-batches from an interaction DataFrame.

    Each batch contains ``batch_size // len(item_types)`` rows from every type.
    Loops infinitely; caller is responsible for stopping after desired steps.

    Args:
        df: interaction DataFrame with at least a column named *type_col*.
        batch_size: total rows per batch.
        item_types: list of type values to stratify over; defaults to all unique
            values in *type_col*.
        type_col: column name containing item type.
        seed: numpy random seed.

    Yields:
        pd.DataFrame with ``batch_size`` rows (fewer if a type has few rows).
    """
    if item_types is None:
        item_types = sorted(df[type_col].unique().tolist())

    rng = np.random.default_rng(seed)
    n_per_type = max(1, batch_size // len(item_types))
    type_dfs = {t: df[df[type_col] == t].reset_index(drop=True) for t in item_types}

    while True:
        parts: List[pd.DataFrame] = []
        for t in item_types:
            pool = type_dfs[t]
            if len(pool) == 0:
                continue
            n = min(n_per_type, len(pool))
            idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
            parts.append(pool.iloc[idx])

        if not parts:
            return

        batch = pd.concat(parts, ignore_index=True)
        # Shuffle rows so types are interleaved
        batch = batch.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
        yield batch


# ---------------------------------------------------------------------------
# StratifiedInteractionDataset
# ---------------------------------------------------------------------------

class StratifiedInteractionDataset:
    """Pre-materialises all interaction features into numpy arrays, then wraps
    them in a stratified ``tf.data.Dataset``.

    Every batch contains ``batch_size // 4`` samples from each ItemType, enabling
    cross-type in-batch negatives for ``mixed_negative_loss``.

    The dataset yields 4-tuples:
        (user_features: dict, item_features: dict, item_type: int32 [B], signal_weights: float32 [B])

    ``item_features`` is the union of all four type schemas.  Each tower only
    accesses the keys relevant to its type; extra keys are silently ignored.

    Args:
        interactions_df: columns ``[user_id, item_id, item_type (int), signal, timestamp]``.
        user_features_df: indexed by ``user_id``; must contain all user feature columns.
        item_features_dict: ``{ItemType: DataFrame}`` indexed by ``item_id``; must contain
            type-specific feature columns *and* a ``text_embed`` column (float32 [256])
            if pre-computed; otherwise ``text_embed`` defaults to zeros.
        signal_weights: mapping from signal name to float weight.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_dict: Dict[ItemType, pd.DataFrame],
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._sig_w = signal_weights or DEFAULT_SIGNAL_WEIGHTS
        self._arrays = self._materialise(interactions_df, user_features_df, item_features_dict)
        type_arr = self._arrays["item_type"]
        self._type_indices: Dict[ItemType, np.ndarray] = {
            itype: np.where(type_arr == int(itype))[0] for itype in ItemType
        }

    # ------------------------------------------------------------------
    # Feature materialisation
    # ------------------------------------------------------------------

    def _materialise(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_dict: Dict[ItemType, pd.DataFrame],
    ) -> Dict[str, object]:
        """Map every interaction row to flat numpy arrays."""
        idf = interactions_df.reset_index(drop=True)
        N = len(idf)

        # ------ User features (vectorised lookup) ------
        uid_to_pos = {uid: i for i, uid in enumerate(user_features_df.index)}
        u_pos = [uid_to_pos[uid] for uid in idf["user_id"]]
        udf = user_features_df.iloc[u_pos].reset_index(drop=True)

        ts = pd.to_datetime(idf["timestamp"])
        dow = ts.dt.dayofweek.values.astype(float)   # 0–6
        hod = ts.dt.hour.values.astype(float)         # 0–23

        u_arrays: Dict[str, np.ndarray] = {
            "age_norm": udf["age_norm"].values.astype(np.float32),
            "home_country_id": udf["home_country_id"].values.astype(np.int32),
            "travel_style_indices": _pad_column(udf["travel_style_indices"], MAX_TRAVEL_STYLES),
            "travel_theme_indices": _pad_column(udf["travel_theme_indices"], MAX_TRAVEL_THEMES),
            "context_day_sin": np.sin(2 * math.pi * dow / 7).astype(np.float32),
            "context_day_cos": np.cos(2 * math.pi * dow / 7).astype(np.float32),
            "context_hour_sin": np.sin(2 * math.pi * hod / 24).astype(np.float32),
            "context_hour_cos": np.cos(2 * math.pi * hod / 24).astype(np.float32),
            "category_pref_indices": _pad_column(udf["category_pref_indices"], MAX_CAT_PREF),
            "category_interaction_history": _to_float32_stack(
                udf["category_interaction_history"], NUM_ITEM_CATEGORIES
            ),
            "subcat_affinity": _to_float32_stack(udf["subcat_affinity"], NUM_ATTRACTION_SUBCATS),
            "article_type_affinity": _to_float32_stack(udf["article_type_affinity"], NUM_ARTICLE_TYPES),
            "province_pref_indices": _pad_column(udf["province_pref_indices"], MAX_PROVINCE_PREF),
        }

        # ------ Item features (union of all type schemas) ------
        TEXT_DIM = 256

        item_type_ids = idf["item_type"].values.astype(np.int32)
        item_ids = idf["item_id"].values

        # Build per-type row lookup (item_id → integer row position)
        item_row_lookup: Dict[ItemType, Dict] = {
            itype: {iid: i for i, iid in enumerate(df.index)}
            for itype, df in item_features_dict.items()
        }

        # Initialise all arrays with safe defaults (-1 padding for indices, 0 elsewhere)
        iv: Dict[str, np.ndarray] = {
            "item_type_id": np.zeros(N, dtype=np.int32),
            "text_embed": np.zeros((N, TEXT_DIM), dtype=np.float32),
            "province_id": np.zeros(N, dtype=np.int32),
            "region_id": np.zeros(N, dtype=np.int32),
            "log_view_count": np.zeros(N, dtype=np.float32),
            # attraction
            "sub_category_indices": np.full((N, MAX_SUBCAT_INDICES), -1, dtype=np.int32),
            "days_open_vector": np.zeros((N, 7), dtype=np.float32),
            "is_free": np.zeros(N, dtype=np.float32),
            # accommodation
            "amenity_indices": np.full((N, MAX_AMENITY_INDICES), -1, dtype=np.int32),
            "price_tier_id": np.zeros(N, dtype=np.int32),
            "is_price_missing": np.zeros(N, dtype=np.float32),
            "star_rating_norm": np.zeros(N, dtype=np.float32),
            # event
            "category_indices": np.full((N, MAX_CATEGORY_INDICES), -1, dtype=np.int32),
            "duration_days_norm": np.zeros(N, dtype=np.float32),
            "month_sin": np.zeros(N, dtype=np.float32),
            "month_cos": np.zeros(N, dtype=np.float32),
            # article
            "article_type_id": np.zeros(N, dtype=np.int32),
            "pub_month_sin": np.zeros(N, dtype=np.float32),
            "pub_month_cos": np.zeros(N, dtype=np.float32),
        }

        for itype in ItemType:
            type_mask = item_type_ids == int(itype)
            if not np.any(type_mask):
                continue

            tidx = np.where(type_mask)[0]           # global interaction indices for this type
            df = item_features_dict[itype]
            lookup = item_row_lookup[itype]
            row_pos = [lookup.get(item_ids[i], 0) for i in tidx]
            tdf = df.iloc[row_pos].reset_index(drop=True)

            iv["item_type_id"][tidx] = int(itype)

            if "text_embed" in df.columns:
                iv["text_embed"][tidx] = _to_float32_stack(tdf["text_embed"], TEXT_DIM)

            if "province_id" in df.columns:
                iv["province_id"][tidx] = tdf["province_id"].values.astype(np.int32)

            if "region_id" in df.columns:
                iv["region_id"][tidx] = tdf["region_id"].values.astype(np.int32)

            if "log_view_count" in df.columns:
                iv["log_view_count"][tidx] = tdf["log_view_count"].values.astype(np.float32)

            if itype == ItemType.ATTRACTION:
                iv["sub_category_indices"][tidx] = _pad_column(
                    tdf["sub_category_indices"], MAX_SUBCAT_INDICES
                )
                iv["days_open_vector"][tidx] = np.stack(tdf["days_open_vector"].values)
                iv["is_free"][tidx] = tdf["is_free"].values.astype(np.float32)

            elif itype == ItemType.ACCOMMODATION:
                iv["amenity_indices"][tidx] = _pad_column(
                    tdf["amenity_indices"], MAX_AMENITY_INDICES
                )
                iv["price_tier_id"][tidx] = tdf["price_tier_id"].values.astype(np.int32)
                iv["is_price_missing"][tidx] = tdf["is_price_missing"].values.astype(np.float32)
                iv["star_rating_norm"][tidx] = tdf["star_rating_norm"].values.astype(np.float32)

            elif itype == ItemType.EVENT:
                iv["category_indices"][tidx] = _pad_column(
                    tdf["category_indices"], MAX_CATEGORY_INDICES
                )
                iv["duration_days_norm"][tidx] = tdf["duration_days_norm"].values.astype(np.float32)
                iv["month_sin"][tidx] = tdf["month_sin"].values.astype(np.float32)
                iv["month_cos"][tidx] = tdf["month_cos"].values.astype(np.float32)

            elif itype == ItemType.ARTICLE:
                iv["article_type_id"][tidx] = tdf["article_type_id"].values.astype(np.int32)
                iv["pub_month_sin"][tidx] = tdf["pub_month_sin"].values.astype(np.float32)
                iv["pub_month_cos"][tidx] = tdf["pub_month_cos"].values.astype(np.float32)

        # Signal weights
        sw = np.array(
            [self._sig_w.get(str(s), 1.0) for s in idf["signal"]], dtype=np.float32
        )

        return {"user": u_arrays, "item": iv, "item_type": item_type_ids, "signal_weights": sw}

    # ------------------------------------------------------------------
    # tf.data builder
    # ------------------------------------------------------------------

    def build(
        self,
        batch_size: int = 512,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> tf.data.Dataset:
        """Return a ``tf.data.Dataset`` of stratified batches.

        Uses tf.constant + tf.gather (no Python generator) so GPU never starves.
        Training dataset is infinite; call ``.take(steps_per_epoch)`` in trainer.
        Validation dataset performs a single full pass.

        Args:
            batch_size: total samples per batch (divided equally across item types).
            split: ``"train"`` or ``"val"``.
            val_fraction: fraction of interactions reserved for validation.
            seed: random seed for shuffle.

        Returns:
            ``tf.data.Dataset`` yielding
            ``(user_features, item_features, item_type [B], signal_weights [B])``.
        """
        N = len(self._arrays["item_type"])
        n_val = max(4, int(N * val_fraction))
        is_train = split == "train"

        active = np.arange(0, N - n_val) if is_train else np.arange(N - n_val, N)
        active_types = self._arrays["item_type"][active]

        # Convert all arrays to tf.constant once — no Python overhead per batch
        u_t = {k: tf.constant(v) for k, v in self._arrays["user"].items()}
        iv_t = {k: tf.constant(v) for k, v in self._arrays["item"].items()}
        type_t = tf.constant(self._arrays["item_type"], dtype=tf.int32)
        sw_t = tf.constant(self._arrays["signal_weights"], dtype=tf.float32)

        def _gather(idx: tf.Tensor):
            return (
                {k: tf.gather(v, idx) for k, v in u_t.items()},
                {k: tf.gather(v, idx) for k, v in iv_t.items()},
                tf.gather(type_t, idx),
                tf.gather(sw_t, idx),
            )

        if is_train:
            indices_per_type = max(1, batch_size // len(ItemType))
            type_index_ds: List[tf.data.Dataset] = []
            for itype in ItemType:
                pool = active[active_types == int(itype)]
                if len(pool) == 0:
                    continue
                ds_idx = tf.data.Dataset.from_tensor_slices(
                    tf.constant(pool, dtype=tf.int64)
                )
                ds_idx = (
                    ds_idx
                    .shuffle(len(pool), seed=seed, reshuffle_each_iteration=True)
                    .repeat()
                    .batch(indices_per_type, drop_remainder=True)
                )
                type_index_ds.append(ds_idx)

            def _merge_and_gather(*type_batches):
                idx = tf.random.shuffle(tf.concat(type_batches, axis=0))
                return _gather(idx)

            ds = tf.data.Dataset.zip(tuple(type_index_ds))
            ds = ds.map(_merge_and_gather, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices(
                tf.constant(active, dtype=tf.int64)
            )
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE)

        return ds.prefetch(tf.data.AUTOTUNE)
