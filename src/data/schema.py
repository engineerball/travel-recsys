"""Feature schema definitions for all towers.

Contracts between preprocessing and model layers.
Only 4 item types exist in raw data: Attraction, Accommodation, Event, Article.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Item types
# ---------------------------------------------------------------------------

class ItemType(enum.IntEnum):
    ATTRACTION = 0
    ACCOMMODATION = 1
    EVENT = 2
    ARTICLE = 3


NUM_ITEM_TYPES = len(ItemType)


# ---------------------------------------------------------------------------
# Province vocabulary (77 Thai provinces, raw ID → sequential index)
# ---------------------------------------------------------------------------

# Raw province IDs as they appear in the parquet data
PROVINCE_IDS: List[int] = [
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117,
    218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
    231, 232, 233, 234, 235, 236, 237, 238, 239,
    343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    463, 464, 465, 466,
    571, 572, 573, 574, 575, 576, 577, 578, 579, 580,
    581, 582, 583, 584, 585, 586, 587, 588, 589, 590,
]

PROVINCE_ID_TO_IDX: Dict[int, int] = {pid: idx for idx, pid in enumerate(PROVINCE_IDS)}
NUM_PROVINCES = len(PROVINCE_IDS)  # 77; index 0–76, unknown → 0


# ---------------------------------------------------------------------------
# Region vocabulary (5 Thai regions derived from TAT province ID prefix)
# ---------------------------------------------------------------------------

# Regions in sequential index order
REGIONS: List[str] = ["North", "Central", "South", "East", "Northeast"]
NUM_REGIONS = len(REGIONS)  # 5; index 0=N, 1=C, 2=S, 3=E, 4=NE

# Sequential province index (0–76) → region index (0–4)
# TAT prefix: 1xx→North(17), 2xx→Central(22), 3xx→South(14), 4xx→East(4), 5xx→NE(20)
PROVINCE_IDX_TO_REGION_IDX: List[int] = (
    [0] * 17 +   # idx 0-16  → North
    [1] * 22 +   # idx 17-38 → Central
    [2] * 14 +   # idx 39-52 → South
    [3] * 4  +   # idx 53-56 → East
    [4] * 20     # idx 57-76 → Northeast
)


# ---------------------------------------------------------------------------
# Attraction sub-category vocabulary (58 types)
# ---------------------------------------------------------------------------

ATTRACTION_SUBCAT_IDS: List[int] = [
    6, 8, 10, 12, 14, 16, 18, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
    41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73,
    75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105,
    107, 109, 111, 113, 115, 117, 120, 122,
]

ATTRACTION_SUBCAT_TO_IDX: Dict[int, int] = {
    sid: idx for idx, sid in enumerate(ATTRACTION_SUBCAT_IDS)
}
NUM_ATTRACTION_SUBCATS = len(ATTRACTION_SUBCAT_IDS)  # 58


# ---------------------------------------------------------------------------
# Event (activity) category vocabulary
# ---------------------------------------------------------------------------

EVENT_CATEGORY_IDS: List[int] = [2, 3, 6, 7, 9, 52, 53, 54, 55, 56, 99]

EVENT_CATEGORY_TO_IDX: Dict[int, int] = {
    cid: idx for idx, cid in enumerate(EVENT_CATEGORY_IDS)
}
NUM_EVENT_CATEGORIES = len(EVENT_CATEGORY_IDS)  # 11


# ---------------------------------------------------------------------------
# Accommodation amenity vocabulary
# Codes from facilities + services arrays. Union of both arrays.
# ---------------------------------------------------------------------------

AMENITY_CODES: List[str] = [
    "PK",  # Parking
    "TS",  # 24hr service
    "RE",  # Restaurant
    "SW",  # Swimming pool
    "FN",  # Fitness
    "BA",  # Bar/Lounge/Pub
    "SP",  # Spa
    "BF",  # Breakfast
    "WI",  # WiFi
    "CF",  # Coffee shop
    "SM",  # Smoking area
    "JA",  # Jacuzzi
    "WA",  # Wheelchair accessible
    "SB",  # Shuttle bus
    "LD",  # Laundry
    "DP",  # Disabled parking
    "AT",  # ATM
    "RR",  # Restroom
    "RS",  # Room service
    "TR",  # Transfer service
    "WC",  # Wheelchair
    "DR",  # Disabled restroom
    "RT",  # Other transport
    "PA",  # Pets allowed
]

AMENITY_CODE_TO_IDX: Dict[str, int] = {
    code: idx for idx, code in enumerate(AMENITY_CODES)
}
NUM_AMENITY_CODES = len(AMENITY_CODES)  # 24


# ---------------------------------------------------------------------------
# User feature vocabularies
# ---------------------------------------------------------------------------

TRAVEL_STYLES: List[str] = ["คู่", "ครอบครัว", "คนเดียว", "เพื่อน", "ผู้สูงอายุ"]

TRAVEL_STYLE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(TRAVEL_STYLES)}
NUM_TRAVEL_STYLES = len(TRAVEL_STYLES)  # 5

TRAVEL_THEMES: List[str] = [
    "สายคาเฟ่/ถ่ายรูป",
    "สายชิลล์/ไนต์ไลฟ์",
    "สายแอดเวนเจอร์/กิจกรรม",
    "สายอาร์ต/แกลเลอรี่",
    "สายสุขภาพ/สปา/น้ำพุร้อน",
    "สายชีวิตในเมือง/ช้อปปิ้ง",
    "สายทะเล",
    "สายฟาร์ม/เกษตร/โฮมสเตย์",
    "สายแคมป์ปิ้ง",
    "สายบุญ/ประวัติศาสตร์",
    "สายกิน/ตะลุยชิม",
    "สายเทศกาล/อีเวนต์",
    "สายครอบครัว/สวนสนุก",
    "สายป่าเขา/น้ำตก",
    "สายชุมชน/วัฒนธรรม",
]

TRAVEL_THEME_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(TRAVEL_THEMES)}
NUM_TRAVEL_THEMES = len(TRAVEL_THEMES)  # 15

# Top home countries seen in data + catch-all
HOME_COUNTRIES: List[str] = [
    "Thailand", "China", "United States", "United Kingdom",
    "South Korea", "Japan", "Other",
]

HOME_COUNTRY_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(HOME_COUNTRIES)}
NUM_HOME_COUNTRIES = len(HOME_COUNTRIES)  # 7


# ---------------------------------------------------------------------------
# Article type vocabulary (typeId from data)
# ---------------------------------------------------------------------------

ARTICLE_TYPE_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

ARTICLE_TYPE_TO_IDX: Dict[int, int] = {tid: idx for idx, tid in enumerate(ARTICLE_TYPE_IDS)}
NUM_ARTICLE_TYPES = len(ARTICLE_TYPE_IDS)  # 12


# ---------------------------------------------------------------------------
# Unified item-category space for user preference features
# Covers attraction sub-categories (58) + event categories (11).
# User behavior features (category_pref_indices, category_interaction_history,
# subcat_affinity) are populated by behavior_generator.py, not raw profiles.
# ---------------------------------------------------------------------------

# Attraction sub-categories occupy indices 0..57; event categories 58..68.
ITEM_CATEGORY_VOCAB: List[str] = (
    [f"attr_{sid}" for sid in ATTRACTION_SUBCAT_IDS]   # 0–57
    + [f"evt_{cid}" for cid in EVENT_CATEGORY_IDS]     # 58–68
)

ITEM_CATEGORY_TO_IDX: Dict[str, int] = {
    c: i for i, c in enumerate(ITEM_CATEGORY_VOCAB)
}
# Convenience: look up by raw IDs
ATTRACTION_SUBCAT_CATEGORY_IDX: Dict[int, int] = {
    sid: ITEM_CATEGORY_TO_IDX[f"attr_{sid}"] for sid in ATTRACTION_SUBCAT_IDS
}
EVENT_CATEGORY_IDX: Dict[int, int] = {
    cid: ITEM_CATEGORY_TO_IDX[f"evt_{cid}"] for cid in EVENT_CATEGORY_IDS
}
NUM_ITEM_CATEGORIES = len(ITEM_CATEGORY_VOCAB)  # 69


# ---------------------------------------------------------------------------
# Price tier thresholds (THB, derived from accommodation minPrice distribution)
# Tier 0 = unknown/missing, tiers 1–5 = budget→luxury
# ---------------------------------------------------------------------------

PRICE_TIER_THRESHOLDS: List[float] = [500.0, 1500.0, 3000.0, 6000.0]  # 4 breakpoints → 5 tiers

NUM_PRICE_TIERS = 6  # 0=missing, 1=budget(<500), 2=economy, 3=standard, 4=premium, 5=luxury(>6000)


# ---------------------------------------------------------------------------
# Thai day-of-week mapping for openingHours parsing
# ---------------------------------------------------------------------------

THAI_DAY_TO_DOW: Dict[str, int] = {
    "วันจันทร์": 0,    # Monday
    "วันอังคาร": 1,    # Tuesday
    "วันพุธ": 2,       # Wednesday
    "วันพฤหัสบดี": 3,  # Thursday
    "วันศุกร์": 4,     # Friday
    "วันเสาร์": 5,     # Saturday
    "วันอาทิตย์": 6,   # Sunday
    # English fallback
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


# ---------------------------------------------------------------------------
# Embedding / model dimensions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """Shared embedding space and per-feature dimensions."""
    output_dim: int = 64                   # unified item/user embedding space
    text_input_dim: int = 256              # Gemma MRL truncation
    province_emb_dim: int = 16
    home_country_emb_dim: int = 8
    travel_style_emb_dim: int = 8
    travel_theme_emb_dim: int = 8
    attraction_subcat_emb_dim: int = 16
    event_category_emb_dim: int = 8
    amenity_emb_dim: int = 8
    price_tier_emb_dim: int = 8
    item_type_emb_dim: int = 8
    article_type_emb_dim: int = 8


EMB_CONFIG = EmbeddingConfig()


# ---------------------------------------------------------------------------
# Feature schema dataclasses
# Describe inputs each tower expects after preprocessing.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserFeatureSchema:
    """Features fed into UserTower.

    Profile features (age, country, style, theme, context) come from raw user_profiles.
    Behavior features (category_pref_indices, category_interaction_history, subcat_affinity)
    are computed by behavior_generator.py from synthetic interaction history.
    """
    # --- profile features (from raw user_profiles) ---
    age_norm: str = "age_norm"                                    # float32 [B]
    home_country_id: str = "home_country_id"                     # int32  [B]
    travel_style_indices: str = "travel_style_indices"           # int32  [B, ≤5]
    travel_theme_indices: str = "travel_theme_indices"           # int32  [B, ≤15]
    context_day_sin: str = "context_day_sin"                     # float32 [B]
    context_day_cos: str = "context_day_cos"                     # float32 [B]
    context_hour_sin: str = "context_hour_sin"                   # float32 [B]
    context_hour_cos: str = "context_hour_cos"                   # float32 [B]
    # --- behavior features (from behavior_generator.py) ---
    category_pref_indices: str = "category_pref_indices"         # int32  [B, ≤NUM_ITEM_CATEGORIES]
    category_interaction_history: str = "category_interaction_history"  # float32 [B, NUM_ITEM_CATEGORIES]
    subcat_affinity: str = "subcat_affinity"                     # float32 [B, NUM_ATTRACTION_SUBCATS]

    profile_feature_names: List[str] = field(default_factory=lambda: [
        "age_norm",
        "home_country_id",
        "travel_style_indices",
        "travel_theme_indices",
        "context_day_sin",
        "context_day_cos",
        "context_hour_sin",
        "context_hour_cos",
    ])

    behavior_feature_names: List[str] = field(default_factory=lambda: [
        "category_pref_indices",
        "category_interaction_history",
        "subcat_affinity",
    ])

    feature_names: List[str] = field(default_factory=lambda: [
        "age_norm",
        "home_country_id",
        "travel_style_indices",
        "travel_theme_indices",
        "context_day_sin",
        "context_day_cos",
        "context_hour_sin",
        "context_hour_cos",
        "category_pref_indices",
        "category_interaction_history",
        "subcat_affinity",
    ])


@dataclass(frozen=True)
class AttractionFeatureSchema:
    """Features for AttractionTower (destinations dataset)."""
    item_type_id: str = "item_type_id"                # int32 = ItemType.ATTRACTION
    text_embed: str = "text_embed"                    # float32 [B, 256]
    sub_category_indices: str = "sub_category_indices"  # int32 [B, ≤10] multi-hot indices
    province_id: str = "province_id"                  # int32 [B]
    days_open_vector: str = "days_open_vector"        # float32 [B, 7] binary
    is_free: str = "is_free"                          # float32 [B]
    log_view_count: str = "log_view_count"            # float32 [B]  log1p(hitScore)

    feature_names: List[str] = field(default_factory=lambda: [
        "item_type_id",
        "text_embed",
        "sub_category_indices",
        "province_id",
        "days_open_vector",
        "is_free",
        "log_view_count",
    ])


@dataclass(frozen=True)
class AccommodationFeatureSchema:
    """Features for AccommodationTower (accommodations dataset)."""
    item_type_id: str = "item_type_id"                # int32 = ItemType.ACCOMMODATION
    text_embed: str = "text_embed"                    # float32 [B, 256]
    amenity_indices: str = "amenity_indices"          # int32 [B, ≤24] multi-hot indices
    province_id: str = "province_id"                  # int32 [B]
    price_tier_id: str = "price_tier_id"              # int32 [B]  0=missing, 1–5=tiers
    is_price_missing: str = "is_price_missing"        # float32 [B]
    star_rating_norm: str = "star_rating_norm"        # float32 [B]  Z-score, 0 if null
    log_view_count: str = "log_view_count"            # float32 [B]  log1p(hitScore)

    feature_names: List[str] = field(default_factory=lambda: [
        "item_type_id",
        "text_embed",
        "amenity_indices",
        "province_id",
        "price_tier_id",
        "is_price_missing",
        "star_rating_norm",
        "log_view_count",
    ])


@dataclass(frozen=True)
class EventFeatureSchema:
    """Features for EventTower (activities dataset)."""
    item_type_id: str = "item_type_id"                # int32 = ItemType.EVENT
    text_embed: str = "text_embed"                    # float32 [B, 256]
    category_indices: str = "category_indices"        # int32 [B, ≤3] multi-hot indices
    province_id: str = "province_id"                  # int32 [B]
    duration_days_norm: str = "duration_days_norm"    # float32 [B]  Z-score
    month_sin: str = "month_sin"                      # float32 [B]
    month_cos: str = "month_cos"                      # float32 [B]

    feature_names: List[str] = field(default_factory=lambda: [
        "item_type_id",
        "text_embed",
        "category_indices",
        "province_id",
        "duration_days_norm",
        "month_sin",
        "month_cos",
    ])


@dataclass(frozen=True)
class ArticleFeatureSchema:
    """Features for ArticleTower (articles dataset)."""
    item_type_id: str = "item_type_id"                # int32 = ItemType.ARTICLE
    text_embed: str = "text_embed"                    # float32 [B, 256]
    article_type_id: str = "article_type_id"          # int32 [B]
    pub_month_sin: str = "pub_month_sin"              # float32 [B]
    pub_month_cos: str = "pub_month_cos"              # float32 [B]
    is_thai: str = "is_thai"                          # float32 [B]  1=Thai, 0=English
    pub_recency_norm: str = "pub_recency_norm"        # float32 [B]  Z-score days since publish

    feature_names: List[str] = field(default_factory=lambda: [
        "item_type_id",
        "text_embed",
        "article_type_id",
        "pub_month_sin",
        "pub_month_cos",
        "is_thai",
        "pub_recency_norm",
    ])


# ---------------------------------------------------------------------------
# VocabRegistry — unified lookup for all categorical vocabularies
# ---------------------------------------------------------------------------

class VocabRegistry:
    """Central registry mapping raw categorical values to sequential indices.

    Provides a single `encode(field, value)` interface used by preprocessing
    and model config code so vocab sizes stay in sync automatically.
    """

    _VOCABS: Dict[str, Dict[Any, int]] = {
        "province_id": PROVINCE_ID_TO_IDX,
        "attraction_subcat_id": ATTRACTION_SUBCAT_TO_IDX,
        "event_category_id": EVENT_CATEGORY_TO_IDX,
        "amenity_code": AMENITY_CODE_TO_IDX,
        "travel_style": TRAVEL_STYLE_TO_IDX,
        "travel_theme": TRAVEL_THEME_TO_IDX,
        "home_country": HOME_COUNTRY_TO_IDX,
        "article_type_id": ARTICLE_TYPE_TO_IDX,
        "item_category": ITEM_CATEGORY_TO_IDX,
    }

    _SIZES: Dict[str, int] = {
        "province_id": NUM_PROVINCES,
        "attraction_subcat_id": NUM_ATTRACTION_SUBCATS,
        "event_category_id": NUM_EVENT_CATEGORIES,
        "amenity_code": NUM_AMENITY_CODES,
        "travel_style": NUM_TRAVEL_STYLES,
        "travel_theme": NUM_TRAVEL_THEMES,
        "home_country": NUM_HOME_COUNTRIES,
        "article_type_id": NUM_ARTICLE_TYPES,
        "item_category": NUM_ITEM_CATEGORIES,
        "price_tier": NUM_PRICE_TIERS,
        "item_type": NUM_ITEM_TYPES,
    }

    @classmethod
    def encode(cls, field: str, value: Any, unknown: int = 0) -> int:
        """Return the index for value in the given field's vocabulary."""
        vocab = cls._VOCABS.get(field)
        if vocab is None:
            raise KeyError(f"Unknown vocab field: '{field}'")
        return vocab.get(value, unknown)

    @classmethod
    def size(cls, field: str) -> int:
        """Return vocabulary size for field (for Embedding layer initialisation)."""
        if field not in cls._SIZES:
            raise KeyError(f"Unknown vocab field: '{field}'")
        return cls._SIZES[field]

    @classmethod
    def fields(cls) -> List[str]:
        return list(cls._SIZES.keys())


# ---------------------------------------------------------------------------
# Convenience: schema instances and lookup by ItemType
# ---------------------------------------------------------------------------

USER_SCHEMA = UserFeatureSchema()
ATTRACTION_SCHEMA = AttractionFeatureSchema()
ACCOMMODATION_SCHEMA = AccommodationFeatureSchema()
EVENT_SCHEMA = EventFeatureSchema()
ARTICLE_SCHEMA = ArticleFeatureSchema()

ITEM_SCHEMAS = {
    ItemType.ATTRACTION: ATTRACTION_SCHEMA,
    ItemType.ACCOMMODATION: ACCOMMODATION_SCHEMA,
    ItemType.EVENT: EVENT_SCHEMA,
    ItemType.ARTICLE: ARTICLE_SCHEMA,
}
