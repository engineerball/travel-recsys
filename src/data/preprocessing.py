"""Preprocessing transformers and per-dataset feature extractors.

Each preprocess_* function takes a raw DataFrame and returns a feature DataFrame
with columns matching the corresponding *FeatureSchema in schema.py.
A separate text_for_embed column is included for Gemma embedding.

text_embed (float32 [256]) is NOT computed here — it is pre-computed by
scripts/embed_items.py and joined in afterward.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import (
    AMENITY_CODE_TO_IDX,
    AMENITY_CODES,
    ARTICLE_TYPE_TO_IDX,
    ATTRACTION_SUBCAT_TO_IDX,
    EVENT_CATEGORY_TO_IDX,
    HOME_COUNTRIES,
    HOME_COUNTRY_TO_IDX,
    PRICE_TIER_THRESHOLDS,
    PROVINCE_ID_TO_IDX,
    PROVINCE_IDX_TO_REGION_IDX,
    THAI_DAY_TO_DOW,
    TRAVEL_STYLES,
    TRAVEL_STYLE_TO_IDX,
    TRAVEL_THEMES,
    TRAVEL_THEME_TO_IDX,
    ItemType,
)


# ---------------------------------------------------------------------------
# Z-score normalizer
# ---------------------------------------------------------------------------

class ZScoreNormalizer:
    """Fit once on training split, reuse for val/test/inference."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = max(std, 1e-8)

    def fit(self, values: pd.Series) -> "ZScoreNormalizer":
        vals = values.dropna().astype(float)
        self.mean = float(vals.mean())
        self.std = max(float(vals.std()), 1e-8)
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        return ((values.astype(float) - self.mean) / self.std).astype("float32")

    def fit_transform(self, values: pd.Series) -> pd.Series:
        return self.fit(values).transform(values)


# Pre-fit normalizers from full-dataset statistics (avoid leakage at schema level).
# These are re-fit in trainer.py on the training split; values here are fallback defaults.
AGE_NORMALIZER = ZScoreNormalizer(mean=34.1, std=11.03)
STAR_RATING_NORMALIZER = ZScoreNormalizer(mean=3.86, std=0.77)
DURATION_DAYS_NORMALIZER = ZScoreNormalizer(mean=8.91, std=34.71)
# Articles span ~2020–2026; reference = 2026-06-01 → mean age ~2 yrs, std ~1.5 yrs
PUB_RECENCY_NORMALIZER = ZScoreNormalizer(mean=730.0, std=548.0)
_PUB_REFERENCE_DATE = pd.Timestamp("2026-06-01")


# ---------------------------------------------------------------------------
# MultiHotEncoder
# ---------------------------------------------------------------------------

class MultiHotEncoder:
    """Encode a list of categorical values to a list of sequential indices.

    Usage:
        enc = MultiHotEncoder(TRAVEL_STYLES)
        enc.encode(["คู่", "เพื่อน"])          # → [0, 3]
        enc.encode_series(df["travel_style"])  # → pd.Series of lists
    """

    def __init__(self, vocab: List[Any], unknown_idx: int = 0) -> None:
        self.vocab = list(vocab)
        self.vocab_to_idx: Dict[Any, int] = {v: i for i, v in enumerate(vocab)}
        self.unknown_idx = unknown_idx

    def encode(self, values: List[Any]) -> List[int]:
        """Map a list of values to their indices; unknown values → unknown_idx."""
        return [self.vocab_to_idx.get(v, self.unknown_idx) for v in values]

    def encode_unique(self, values: List[Any]) -> List[int]:
        """Same as encode but deduplicates while preserving order."""
        seen: set = set()
        out: List[int] = []
        for v in values:
            idx = self.vocab_to_idx.get(v, self.unknown_idx)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    def encode_series(self, series: pd.Series) -> pd.Series:
        """Apply encode to every element of a Series of lists."""
        return series.apply(
            lambda v: self.encode(v) if isinstance(v, list) else []
        )

    def vocab_size(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# CyclicalEncoder
# ---------------------------------------------------------------------------

class CyclicalEncoder:
    """Encode a periodic integer/float as (sin, cos) pair.

    Usage:
        enc = CyclicalEncoder(period=7)
        enc.encode(3)                    # → (sin, cos) for day 3
        enc.encode_series(df["month"])   # → (sin_series, cos_series)
    """

    def __init__(self, period: float) -> None:
        self.period = period

    def encode(self, value: float) -> Tuple[float, float]:
        angle = 2.0 * math.pi * value / self.period
        return math.sin(angle), math.cos(angle)

    def encode_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Return (sin_series, cos_series) both as float32."""
        angles = series.astype(float) * (2.0 * math.pi / self.period)
        return (
            np.sin(angles).astype("float32"),
            np.cos(angles).astype("float32"),
        )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json(s) -> dict:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return {}
    if isinstance(s, dict):
        return s
    return {}


def _parse_json_array(arr) -> List[dict]:
    """Parse numpy array of JSON strings OR a JSON string of an array."""
    if arr is None or (isinstance(arr, float) and math.isnan(arr)):
        return []
    if isinstance(arr, str):
        try:
            result = json.loads(arr)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    if hasattr(arr, "__iter__"):
        out = []
        for item in arr:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                try:
                    out.append(json.loads(item))
                except (json.JSONDecodeError, ValueError):
                    pass
        return out
    return []


def _safe_str(val) -> str:
    """Convert to str, treating None/NaN/pandas NA as empty string."""
    if val is None:
        return ""
    try:
        if (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


# ---------------------------------------------------------------------------
# Cyclical encoding
# ---------------------------------------------------------------------------

def _cyclic_sin_cos(value: float, period: float):
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


# ---------------------------------------------------------------------------
# User preprocessing
# ---------------------------------------------------------------------------

_STYLE_ENCODER = MultiHotEncoder(TRAVEL_STYLES)
_THEME_ENCODER = MultiHotEncoder(TRAVEL_THEMES)
_DAY_ENCODER = CyclicalEncoder(period=7)
_HOUR_ENCODER = CyclicalEncoder(period=24)
_MONTH_ENCODER = CyclicalEncoder(period=12)


def preprocess_users(
    df: pd.DataFrame,
    age_normalizer: Optional[ZScoreNormalizer] = None,
) -> pd.DataFrame:
    """Convert raw user_profiles DataFrame to model-ready profile features.

    Produces profile features only. Behavior features (category_pref_indices,
    category_interaction_history, subcat_affinity) are added later by
    behavior_generator.py once interaction history is available.

    Args:
        df: raw user_profiles parquet DataFrame.
        age_normalizer: fitted ZScoreNormalizer for age. Uses AGE_NORMALIZER if None.

    Returns:
        DataFrame with user_id + UserFeatureSchema.profile_feature_names columns.
    """
    if age_normalizer is None:
        age_normalizer = AGE_NORMALIZER

    out = pd.DataFrame()
    out["user_id"] = df["user_id"]

    out["age_norm"] = age_normalizer.transform(df["age"])

    out["home_country_id"] = (
        df["home_country"]
        .apply(lambda c: HOME_COUNTRY_TO_IDX.get(str(c), HOME_COUNTRY_TO_IDX["Other"]))
        .astype("int32")
    )

    def _parse_list(s) -> List:
        try:
            return ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, list) else [s])
        except (ValueError, SyntaxError):
            return [s]

    out["travel_style_indices"] = df["travel_style"].apply(
        lambda s: _STYLE_ENCODER.encode_unique(_parse_list(s))
    )
    out["travel_theme_indices"] = df["travel_theme"].apply(
        lambda s: _THEME_ENCODER.encode_unique(_parse_list(s))
    )

    # Cyclical context from preferred_time_start_at
    def _safe_ts_attr(series: pd.Series, attr: str, default: float) -> pd.Series:
        def _get(ts):
            try:
                return float(getattr(pd.Timestamp(ts), attr))
            except Exception:
                return default
        return series.apply(_get)

    dow = _safe_ts_attr(df["preferred_time_start_at"], "dayofweek", 0.0)
    hour = _safe_ts_attr(df["preferred_time_start_at"], "hour", 0.0)

    out["context_day_sin"], out["context_day_cos"] = _DAY_ENCODER.encode_series(dow)
    out["context_hour_sin"], out["context_hour_cos"] = _HOUR_ENCODER.encode_series(hour)

    return out


# ---------------------------------------------------------------------------
# Attraction (destinations) preprocessing
# ---------------------------------------------------------------------------

def _province_id_to_region_idx(province_id_series: pd.Series) -> pd.Series:
    """Map raw province_id → sequential province_idx → region_idx (0-4)."""
    return (
        province_id_series
        .apply(lambda x: PROVINCE_ID_TO_IDX.get(int(x), 0) if pd.notna(x) else 0)
        .apply(lambda idx: PROVINCE_IDX_TO_REGION_IDX[idx])
        .astype("int32")
    )


def _parse_opening_hours(arr) -> List[float]:
    """openingHours ndarray of JSON strings → 7-dim binary float list."""
    vec = [0.0] * 7
    for item in _parse_json_array(arr):
        day_name = item.get("day", "")
        idx = THAI_DAY_TO_DOW.get(day_name)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _parse_is_free(info_str) -> float:
    info = _parse_json(info_str)
    fee = info.get("fee", {})
    if not fee:
        return 1.0  # assume free when fee data absent
    thai = fee.get("thaiAdult", 0) or 0
    foreign = fee.get("foreignerAdult", 0) or 0
    return 1.0 if (thai == 0 and foreign == 0) else 0.0


def _parse_subcat_indices(sub_cats_str) -> List[int]:
    items = _parse_json_array(sub_cats_str)
    idxs = []
    for item in items:
        sid = item.get("subCategoryId")
        if sid is not None and sid in ATTRACTION_SUBCAT_TO_IDX:
            idxs.append(ATTRACTION_SUBCAT_TO_IDX[sid])
    return idxs


def preprocess_attractions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw destinations DataFrame to AttractionTower features.

    Returns DataFrame with: place_id, item_type_id, sub_category_indices,
    province_id, days_open_vector, is_free, log_view_count, text_for_embed.
    """
    out = pd.DataFrame()
    out["place_id"] = df["placeId"]
    out["item_type_id"] = np.int32(ItemType.ATTRACTION)

    out["sub_category_indices"] = df["sub_categories"].apply(_parse_subcat_indices)

    out["province_id"] = (
        df["province_id"]
        .apply(lambda x: PROVINCE_ID_TO_IDX.get(int(x), 0) if pd.notna(x) else 0)
        .astype("int32")
    )
    out["region_id"] = _province_id_to_region_idx(df["province_id"])

    out["days_open_vector"] = df["openingHours"].apply(_parse_opening_hours)

    out["is_free"] = df["information"].apply(_parse_is_free).astype("float32")

    out["log_view_count"] = np.log1p(df["hitScore"].fillna(0).astype(float)).astype("float32")

    out["text_for_embed"] = df.apply(_attraction_text, axis=1)

    return out


def _attraction_text(row) -> str:
    name = _safe_str(row.get("name"))
    info = _parse_json(row.get("information", ""))
    intro = _strip_html(_safe_str(info.get("introduction")))
    detail = _strip_html(_safe_str(info.get("detail")))
    body = detail if len(detail) > len(intro) else intro
    return f"{name}. {body}".strip()


# ---------------------------------------------------------------------------
# Accommodation preprocessing
# ---------------------------------------------------------------------------

def _price_to_tier(price) -> int:
    """Bin a price (THB) into tier 0–5. 0 = missing/unknown."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return 0
    if p <= 0:
        return 0
    for tier, threshold in enumerate(PRICE_TIER_THRESHOLDS, start=1):
        if p < threshold:
            return tier
    return len(PRICE_TIER_THRESHOLDS) + 1  # max tier = 5


def _parse_amenity_indices(facilities_arr, services_arr) -> List[int]:
    codes: set[str] = set()
    for arr in (facilities_arr, services_arr):
        for item in _parse_json_array(arr):
            code = item.get("code", "")
            if code in AMENITY_CODE_TO_IDX:
                codes.add(code)
    return [AMENITY_CODE_TO_IDX[c] for c in sorted(codes)]


def preprocess_accommodations(
    df: pd.DataFrame,
    star_normalizer: Optional[ZScoreNormalizer] = None,
) -> pd.DataFrame:
    """Convert raw accommodations DataFrame to AccommodationTower features."""
    if star_normalizer is None:
        star_normalizer = STAR_RATING_NORMALIZER

    out = pd.DataFrame()
    out["place_id"] = df["placeId"]
    out["item_type_id"] = np.int32(ItemType.ACCOMMODATION)

    out["amenity_indices"] = [
        _parse_amenity_indices(f, s)
        for f, s in zip(df["facilities"], df["services"])
    ]

    out["province_id"] = (
        df["province_id"]
        .apply(lambda x: PROVINCE_ID_TO_IDX.get(int(x), 0) if pd.notna(x) else 0)
        .astype("int32")
    )
    out["region_id"] = _province_id_to_region_idx(df["province_id"])

    prices = df["minPrice"].apply(lambda x: float(x) if x is not None else 0.0)
    out["price_tier_id"] = prices.apply(_price_to_tier).astype("int32")
    out["is_price_missing"] = (prices <= 0).astype("float32")

    # hotelStar lives inside information JSON
    def _extract_star(info_str) -> Optional[float]:
        info = _parse_json(info_str)
        val = info.get("hotelStar")
        return float(val) if val is not None else None

    raw_stars = df["information"].apply(_extract_star)
    # fill null with mean before normalizing so the Z-score is centered at 0
    star_filled = raw_stars.fillna(STAR_RATING_NORMALIZER.mean)
    out["star_rating_norm"] = star_normalizer.transform(star_filled).astype("float32")

    out["log_view_count"] = np.log1p(df["hitScore"].fillna(0).astype(float)).astype("float32")

    out["text_for_embed"] = df.apply(_accommodation_text, axis=1)

    return out


def _accommodation_text(row) -> str:
    name = _safe_str(row.get("name"))
    info = _parse_json(row.get("information", ""))
    intro = _strip_html(_safe_str(info.get("introduction")))
    detail = _strip_html(_safe_str(info.get("detail")))
    body = detail if len(detail) > len(intro) else intro
    return f"{name}. {body}".strip()


# ---------------------------------------------------------------------------
# Event (activities) preprocessing
# ---------------------------------------------------------------------------

def _parse_event_category_indices(categories_arr) -> List[int]:
    idxs = []
    for item in _parse_json_array(categories_arr):
        cid = item.get("categoryId")
        if cid is not None and cid in EVENT_CATEGORY_TO_IDX:
            idxs.append(EVENT_CATEGORY_TO_IDX[cid])
    return idxs or [EVENT_CATEGORY_TO_IDX.get(99, 0)]  # fallback to "other"


def preprocess_events(
    df: pd.DataFrame,
    duration_normalizer: Optional[ZScoreNormalizer] = None,
) -> pd.DataFrame:
    """Convert raw activities DataFrame to EventTower features."""
    if duration_normalizer is None:
        duration_normalizer = DURATION_DAYS_NORMALIZER

    out = pd.DataFrame()
    out["event_id"] = df["eventId"]
    out["item_type_id"] = np.int32(ItemType.EVENT)

    out["category_indices"] = df["categories"].apply(_parse_event_category_indices)

    out["province_id"] = (
        df["province_id"]
        .apply(lambda x: PROVINCE_ID_TO_IDX.get(int(x), 0) if pd.notna(x) else 0)
        .astype("int32")
    )
    out["region_id"] = _province_id_to_region_idx(df["province_id"])

    # Duration: clip negatives to 0 before normalizing
    duration_raw = (df["endDate"] - df["startDate"]).dt.days.clip(lower=0).fillna(0)
    out["duration_days_norm"] = duration_normalizer.transform(duration_raw).astype("float32")

    start_month = df["startDate"].dt.month.fillna(1).astype(float)
    out["month_sin"], out["month_cos"] = _MONTH_ENCODER.encode_series(start_month)

    out["text_for_embed"] = df.apply(_event_text, axis=1)

    return out


def _event_text(row) -> str:
    name = _safe_str(row.get("name"))
    info = _parse_json(row.get("information", ""))
    intro = _strip_html(_safe_str(info.get("introduction")))
    html_detail = _strip_html(_safe_str(info.get("htmlDetail")))
    body = html_detail if len(html_detail) > len(intro) else intro
    return f"{name}. {body}".strip()


# ---------------------------------------------------------------------------
# Article preprocessing
# ---------------------------------------------------------------------------

def preprocess_articles(
    df: pd.DataFrame,
    recency_normalizer: Optional[ZScoreNormalizer] = None,
) -> pd.DataFrame:
    """Convert raw articles DataFrame to ArticleTower features."""
    if recency_normalizer is None:
        recency_normalizer = PUB_RECENCY_NORMALIZER

    out = pd.DataFrame()
    out["article_id"] = df["id"]
    out["item_type_id"] = np.int32(ItemType.ARTICLE)

    def _type_idx(tid) -> int:
        try:
            return ARTICLE_TYPE_TO_IDX.get(int(tid), 0)
        except (TypeError, ValueError):
            return 0

    out["article_type_id"] = df["typeId"].apply(_type_idx).astype("int32")

    pub_month = df["datetimeCreated"].dt.month.fillna(1).astype(float)
    out["pub_month_sin"], out["pub_month_cos"] = _MONTH_ENCODER.encode_series(pub_month)

    # Language: Thai (th) vs other (en).  Binary float.
    out["is_thai"] = (df["language"].fillna("") == "th").astype("float32")

    # Recency: days between publish date and reference date, Z-score normalized.
    created = pd.to_datetime(df["datetimeCreated"], errors="coerce")
    age_days = (_PUB_REFERENCE_DATE - created).dt.days.fillna(recency_normalizer.mean)
    out["pub_recency_norm"] = recency_normalizer.transform(age_days).astype("float32")

    out["text_for_embed"] = df.apply(_article_text, axis=1)

    return out


def _article_text(row) -> str:
    title = _safe_str(row.get("title"))
    intro = _strip_html(_safe_str(row.get("introduction")))
    meta = _strip_html(_safe_str(row.get("metaDataDescription")))
    body = intro if intro else meta
    return f"{title}. {body}".strip()


# ---------------------------------------------------------------------------
# Convenience: load + preprocess full dataset from parquet directory
# ---------------------------------------------------------------------------

def load_and_preprocess(item_type: ItemType, raw_dir: str) -> pd.DataFrame:
    """Load all parquet files from raw_dir and return preprocessed features.

    Args:
        item_type: which item type to load.
        raw_dir: path to directory containing .parquet files.

    Returns:
        Feature DataFrame ready for embedding join and model training.
    """
    import glob as _glob

    files = sorted(_glob.glob(f"{raw_dir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    dispatch = {
        ItemType.ATTRACTION: preprocess_attractions,
        ItemType.ACCOMMODATION: preprocess_accommodations,
        ItemType.EVENT: preprocess_events,
        ItemType.ARTICLE: preprocess_articles,
    }

    return dispatch[item_type](df)
