"""Synthetic interaction generator using persona-based simulation.

Generates (user, item, signal, timestamp) interactions and computes user
behavior features (category_pref_indices, category_interaction_history,
subcat_affinity) from interaction history.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import (
    ItemType,
    NUM_ARTICLE_TYPES,
    NUM_ATTRACTION_SUBCATS,
    NUM_ITEM_CATEGORIES,
    NUM_PROVINCES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGNAL_WEIGHTS: Dict[str, float] = {"view": 1.0, "click": 3.0}
_SESSION_START = pd.Timestamp("2024-01-01")
_SESSION_END = pd.Timestamp("2025-06-30")
_EARTH_RADIUS_KM = 6371.0

# item_type_weights: {ItemType int: sampling probability}
_PERSONA_TYPE_WEIGHTS: Dict[str, Dict[int, float]] = {
    "backpacker":     {0: 0.50, 1: 0.10, 2: 0.30, 3: 0.10},
    "family":         {0: 0.40, 1: 0.20, 2: 0.25, 3: 0.15},
    "culture_seeker": {0: 0.50, 1: 0.10, 2: 0.20, 3: 0.20},
    "foodie":         {0: 0.15, 1: 0.10, 2: 0.45, 3: 0.30},
    "luxury":         {0: 0.25, 1: 0.45, 2: 0.20, 3: 0.10},
}

# Article type preferences per persona.
# Keys are typeId values (1-12) matching the raw `type.id` field.
# Missing typeId → weight 1.0 (neutral). typeId=12 (News & Update) is
# downweighted across all personas — it has no preference signal.
_PERSONA_ARTICLE_TYPE_PREFS: Dict[str, Dict[int, float]] = {
    "backpacker":     {3: 4.0, 8: 4.0, 10: 3.0, 9: 2.5, 1: 2.0, 4: 1.5, 12: 0.2},
    "family":         {4: 4.0, 3: 3.0, 8: 3.0, 2: 2.5, 1: 2.0, 10: 2.0, 12: 0.2},
    "culture_seeker": {1: 4.0, 2: 4.0, 3: 3.0, 4: 3.0, 9: 2.5, 8: 2.0, 12: 0.1},
    "foodie":         {6: 5.0, 8: 3.0, 9: 2.5, 3: 2.0, 10: 1.5, 12: 0.2},
    "luxury":         {5: 4.0, 6: 4.0, 8: 3.0, 9: 3.0, 1: 2.0, 12: 0.1},
}

# travel_theme → additive article-type weight bias (typeId keys, same as _PERSONA_ARTICLE_TYPE_PREFS)
# Applied on top of persona prefs so user-level theme signals compound.
_THEME_ARTICLE_TYPE_BIAS: Dict[str, Dict[int, float]] = {
    "สายกิน/ตะลุยชิม":            {6: 4.0, 8: 2.0, 9: 2.0},
    "สายบุญ/ประวัติศาสตร์":        {1: 4.0, 2: 4.0, 4: 3.0},
    "สายอาร์ต/แกลเลอรี่":          {2: 4.0, 1: 3.0, 4: 3.0},
    "สายแอดเวนเจอร์/กิจกรรม":     {3: 4.0, 8: 3.0, 10: 3.0},
    "สายทะเล":                     {3: 3.0, 8: 3.0, 10: 2.0},
    "สายแคมป์ปิ้ง":                {3: 4.0, 8: 3.0, 10: 3.0},
    "สายป่าเขา/น้ำตก":             {3: 3.0, 8: 3.0, 10: 2.0},
    "สายสุขภาพ/สปา/น้ำพุร้อน":    {5: 4.0, 9: 3.0, 8: 2.0},
    "สายชิลล์/ไนต์ไลฟ์":          {5: 3.0, 9: 3.0, 6: 2.0},
    "สายครอบครัว/สวนสนุก":         {4: 4.0, 3: 3.0, 2: 2.0},
    "สายชีวิตในเมือง/ช้อปปิ้ง":   {6: 3.0, 8: 3.0, 9: 2.0, 5: 2.0},
    "สายคาเฟ่/ถ่ายรูป":           {6: 3.0, 8: 3.0, 9: 2.0},
    "สายฟาร์ม/เกษตร/โฮมสเตย์":   {3: 2.0, 8: 2.0, 4: 2.0},
    "สายเทศกาล/อีเวนต์":           {7: 4.0, 3: 2.0, 8: 2.0},
    "สายชุมชน/วัฒนธรรม":           {1: 3.0, 2: 3.0, 4: 3.0},
}

# home_country → (thai_article_weight, non_thai_article_weight)
# Thai users prefer Thai-language content; Western users prefer English content.
_LANGUAGE_COUNTRY_PREF: Dict[str, tuple] = {
    "Thailand":       (3.0, 1.0),
    "China":          (0.5, 2.0),
    "South Korea":    (0.5, 2.0),
    "Japan":          (0.5, 2.0),
    "United States":  (0.3, 3.0),
    "United Kingdom": (0.3, 3.0),
    "Other":          (1.0, 1.5),
}

# travel_theme → additive persona bias (unnormalized)
# Themes from schema.py TRAVEL_THEMES; missing theme → no bias
_THEME_PERSONA_BIAS: Dict[str, Dict[str, float]] = {
    "สายคาเฟ่/ถ่ายรูป":           {"foodie": 1.0, "luxury": 1.0},
    "สายชิลล์/ไนต์ไลฟ์":          {"luxury": 2.0},
    "สายแอดเวนเจอร์/กิจกรรม":     {"backpacker": 3.0},
    "สายอาร์ต/แกลเลอรี่":          {"culture_seeker": 2.0},
    "สายสุขภาพ/สปา/น้ำพุร้อน":    {"luxury": 2.0},
    "สายชีวิตในเมือง/ช้อปปิ้ง":   {"luxury": 1.0, "foodie": 1.0},
    "สายทะเล":                     {"backpacker": 2.0},
    "สายฟาร์ม/เกษตร/โฮมสเตย์":   {"backpacker": 1.0, "culture_seeker": 1.0},
    "สายแคมป์ปิ้ง":                {"backpacker": 2.0},
    "สายบุญ/ประวัติศาสตร์":        {"culture_seeker": 3.0},
    "สายกิน/ตะลุยชิม":             {"foodie": 3.0},
    "สายเทศกาล/อีเวนต์":           {"culture_seeker": 1.0, "backpacker": 1.0},
    "สายครอบครัว/สวนสนุก":         {"family": 3.0},
    "สายป่าเขา/น้ำตก":             {"backpacker": 2.0},
    "สายชุมชน/วัฒนธรรม":           {"culture_seeker": 2.0},
}

# travel_style → additive persona bias
_STYLE_PERSONA_BIAS: Dict[str, Dict[str, float]] = {
    "คู่":       {"luxury": 1.0},
    "ครอบครัว":  {"family": 3.0},
    "คนเดียว":   {"backpacker": 1.0, "culture_seeker": 1.0},
    "เพื่อน":    {"backpacker": 1.0, "foodie": 1.0},
    "ผู้สูงอายุ": {"culture_seeker": 1.0, "family": 1.0},
}

# signal_probs: conditional click probability per interaction
_PERSONA_SIGNAL_PROBS: Dict[str, Dict[str, float]] = {
    "backpacker":     {"click": 0.14},
    "family":         {"click": 0.28},
    "culture_seeker": {"click": 0.37},
    "foodie":         {"click": 0.40},
    "luxury":         {"click": 0.37},
}


# ---------------------------------------------------------------------------
# PersonaConfig
# ---------------------------------------------------------------------------

@dataclass
class PersonaConfig:
    """Behavioral parameters for one persona type."""

    name: str
    weight: float
    preferred_themes: List[str]
    price_sensitivity: float       # 0=indifferent, 1=strongly prefer free/cheap
    travel_radius_km: float
    item_type_weights: Dict[int, float] = field(default_factory=dict)
    signal_probs: Dict[str, float] = field(default_factory=dict)


def _parse_list(val) -> List:
    """Parse a value that may be a list, stringified list, or scalar."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, list) else [val]
        except (ValueError, SyntaxError):
            return [val]
    return []


def load_personas_from_config(config: dict) -> List[PersonaConfig]:
    """Parse behavior.personas from config dict and return PersonaConfig list."""
    personas = []
    for p in config.get("behavior", {}).get("personas", []):
        name = p["name"]
        personas.append(PersonaConfig(
            name=name,
            weight=float(p.get("weight", 0.2)),
            preferred_themes=p.get("preferred_themes", []),
            price_sensitivity=float(p.get("price_sensitivity", 0.5)),
            travel_radius_km=float(p.get("travel_radius_km", 200)),
            item_type_weights=_PERSONA_TYPE_WEIGHTS.get(
                name, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
            ),
            signal_probs=_PERSONA_SIGNAL_PROBS.get(
                name, {"click": 0.20}
            ),
        ))
    return personas


# ---------------------------------------------------------------------------
# PopularityModel
# ---------------------------------------------------------------------------

class PopularityModel:
    """Power-law item sampling based on ranked popularity scores.

    Items ranked by descending score; sampling probability ∝ rank^(-exponent).
    """

    def __init__(
        self,
        scores: np.ndarray,
        exponent: float = 0.7,
        min_prob: float = 1e-6,
    ) -> None:
        n = len(scores)
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        raw = np.maximum(ranks ** (-exponent), min_prob)
        self._probs = raw / raw.sum()

    def sample(
        self,
        n: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample n indices using power-law probs restricted to eligible items.

        Falls back to uniform over eligible items if all masked probs are zero.
        """
        if rng is None:
            rng = np.random.default_rng()
        probs = self._probs.copy()
        if mask is not None:
            probs[~mask] = 0.0
            total = probs.sum()
            if total == 0:
                # All eligible items masked out — fall back to unmasked power-law probs
                probs = self._probs.copy()
            else:
                probs /= total
        return rng.choice(len(probs), size=n, p=probs, replace=True)


# ---------------------------------------------------------------------------
# GeographicFilter
# ---------------------------------------------------------------------------

def _haversine_batch(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance (km) from one point to many."""
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class GeographicFilter:
    """Computes and caches per-item distances from province centroids."""

    _DEFAULT_LAT, _DEFAULT_LON = 13.75, 100.52  # Bangkok fallback

    def __init__(
        self,
        item_dfs: Dict[ItemType, pd.DataFrame],
        province_centroids: Dict[int, Tuple[float, float]],
    ) -> None:
        self._item_dfs = item_dfs
        self._province_centroids = province_centroids
        self._cache: Dict[Tuple[int, ItemType], np.ndarray] = {}

    def get_mask(
        self,
        province_id: int,
        item_type: ItemType,
        radius_km: float,
    ) -> np.ndarray:
        """Boolean mask: items within radius_km of the province centroid."""
        df = self._item_dfs[item_type]
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return np.ones(len(df), dtype=bool)

        key = (province_id, item_type)
        if key not in self._cache:
            lat0, lon0 = self._province_centroids.get(
                province_id, (self._DEFAULT_LAT, self._DEFAULT_LON)
            )
            lats = df["latitude"].fillna(self._DEFAULT_LAT).to_numpy(dtype=float)
            lons = df["longitude"].fillna(self._DEFAULT_LON).to_numpy(dtype=float)
            self._cache[key] = _haversine_batch(lat0, lon0, lats, lons)

        return self._cache[key] <= radius_km


# ---------------------------------------------------------------------------
# TemporalFilter
# ---------------------------------------------------------------------------

class TemporalFilter:
    """Temporal feasibility checks for attractions and events."""

    @staticmethod
    def attraction_mask(df: pd.DataFrame, dow: int) -> np.ndarray:
        """Items open on day_of_week dow (0=Monday, 6=Sunday).

        Returns all-True if days_open_vector is absent or malformed.
        """
        def _open(vec) -> bool:
            return bool(vec[dow]) if isinstance(vec, list) and len(vec) == 7 else True

        return np.array([_open(v) for v in df["days_open_vector"]])

    @staticmethod
    def event_mask(df: pd.DataFrame, session_date: pd.Timestamp) -> np.ndarray:
        """Events whose [startDate, endDate] range includes session_date."""
        start = pd.to_datetime(
            df.get("startDate", pd.Series(dtype="datetime64[ns]")), errors="coerce"
        ).fillna(pd.Timestamp("2010-01-01"))
        end = pd.to_datetime(
            df.get("endDate", pd.Series(dtype="datetime64[ns]")), errors="coerce"
        ).fillna(pd.Timestamp("2030-12-31"))
        return ((start <= session_date) & (session_date <= end)).to_numpy()


# ---------------------------------------------------------------------------
# InteractionGenerator
# ---------------------------------------------------------------------------

@dataclass
class _Pool:
    """Preprocessed item corpus for one item type."""

    item_type: ItemType
    df: pd.DataFrame     # preprocessed features + lat/lon + temporal cols
    id_col: str
    pop_model: PopularityModel


class InteractionGenerator:
    """Generates synthetic (user, item, signal, timestamp) interaction tuples.

    Takes raw DataFrames and preprocesses them internally, merging lat/lon
    and temporal columns needed for geographic and temporal filtering.
    """

    _ID_COLS: Dict[ItemType, str] = {
        ItemType.ATTRACTION: "place_id",
        ItemType.ACCOMMODATION: "place_id",
        ItemType.EVENT: "event_id",
        ItemType.ARTICLE: "article_id",
    }
    _SCORE_COLS: Dict[ItemType, Optional[str]] = {
        ItemType.ATTRACTION: "log_view_count",
        ItemType.ACCOMMODATION: "log_view_count",
        ItemType.EVENT: "duration_days_norm",
        ItemType.ARTICLE: None,
    }
    # Raw columns to carry over into preprocessed pool df
    _EXTRA_COLS: Dict[ItemType, List[str]] = {
        ItemType.ATTRACTION: ["latitude", "longitude", "province_name"],
        ItemType.ACCOMMODATION: ["latitude", "longitude", "province_name"],
        ItemType.EVENT: ["latitude", "longitude", "province_name", "startDate", "endDate"],
        ItemType.ARTICLE: [],
    }

    def __init__(
        self,
        users_df: pd.DataFrame,
        raw_item_dfs: Dict[ItemType, pd.DataFrame],
        personas: List[PersonaConfig],
        popularity_power: float = 0.7,
        interactions_per_user: Tuple[int, int] = (5, 50),
        seed: int = 42,
    ) -> None:
        self._users = users_df.reset_index(drop=True)
        self._personas = personas
        self._pw = np.array([p.weight for p in personas], dtype=float)
        self._pw /= self._pw.sum()
        self._ipp_min, self._ipp_max = interactions_per_user
        self._rng = np.random.default_rng(seed)

        self._pools = self._build_pools(raw_item_dfs, popularity_power)
        self._province_centroids = self._build_province_centroids(raw_item_dfs)
        self._prov_name_to_id = self._build_province_name_map(raw_item_dfs)

        self._geo = GeographicFilter(
            {itype: pool.df for itype, pool in self._pools.items()},
            self._province_centroids,
        )

    def _build_pools(
        self,
        raw_dfs: Dict[ItemType, pd.DataFrame],
        popularity_power: float,
    ) -> Dict[ItemType, _Pool]:
        from src.data.preprocessing import (
            preprocess_articles,
            preprocess_attractions,
            preprocess_accommodations,
            preprocess_events,
        )
        preprocessors = {
            ItemType.ATTRACTION: preprocess_attractions,
            ItemType.ACCOMMODATION: preprocess_accommodations,
            ItemType.EVENT: preprocess_events,
            ItemType.ARTICLE: preprocess_articles,
        }
        pools: Dict[ItemType, _Pool] = {}
        for itype, raw_df in raw_dfs.items():
            if len(raw_df) == 0:
                continue
            prepped = preprocessors[itype](raw_df).reset_index(drop=True)
            for col in self._EXTRA_COLS.get(itype, []):
                if col in raw_df.columns:
                    prepped[col] = raw_df[col].values

            sc = self._SCORE_COLS[itype]
            scores = (
                prepped[sc].fillna(0).clip(lower=0).to_numpy(dtype=float)
                if sc and sc in prepped.columns
                else np.ones(len(prepped), dtype=float)
            )
            pools[itype] = _Pool(
                item_type=itype,
                df=prepped,
                id_col=self._ID_COLS[itype],
                pop_model=PopularityModel(scores, exponent=popularity_power),
            )
        return pools

    @staticmethod
    def _build_province_centroids(
        raw_dfs: Dict[ItemType, pd.DataFrame],
    ) -> Dict[int, Tuple[float, float]]:
        rows = []
        for df in raw_dfs.values():
            if "province_id" in df.columns and "latitude" in df.columns:
                rows.append(df[["province_id", "latitude", "longitude"]].dropna())
        if not rows:
            return {}
        combined = pd.concat(rows, ignore_index=True)
        return {
            int(pid): (float(g["latitude"].mean()), float(g["longitude"].mean()))
            for pid, g in combined.groupby("province_id")[["latitude", "longitude"]]
        }

    @staticmethod
    def _build_province_name_map(
        raw_dfs: Dict[ItemType, pd.DataFrame],
    ) -> Dict[str, int]:
        pmap: Dict[str, int] = {}
        for df in raw_dfs.values():
            if "province_name" in df.columns and "province_id" in df.columns:
                for name, pid in zip(df["province_name"], df["province_id"]):
                    if pd.notna(name) and pd.notna(pid):
                        pmap[str(name)] = int(pid)
        return pmap

    def _user_persona_weights(self, user_row: pd.Series) -> np.ndarray:
        """Compute per-user persona sampling weights from travel_style and travel_theme.

        Starts from global persona weights, adds theme/style bias, normalizes.
        Users with no matching themes fall back to global weights.
        """
        weights = self._pw.copy()  # shape: (n_personas,)
        persona_names = [p.name for p in self._personas]

        for theme in _parse_list(user_row.get("travel_theme", [])):
            for p_name, bias in _THEME_PERSONA_BIAS.get(str(theme), {}).items():
                if p_name in persona_names:
                    weights[persona_names.index(p_name)] += bias

        for style in _parse_list(user_row.get("travel_style", [])):
            for p_name, bias in _STYLE_PERSONA_BIAS.get(str(style), {}).items():
                if p_name in persona_names:
                    weights[persona_names.index(p_name)] += bias

        total = weights.sum()
        return weights / total if total > 0 else self._pw

    def _user_province_id(self, user_row: pd.Series) -> int:
        name = str(user_row.get("home_province", ""))
        return self._prov_name_to_id.get(name, 101)  # default: Bangkok (province_id 101)

    def _random_session_ts(self) -> pd.Timestamp:
        delta_days = (_SESSION_END - _SESSION_START).days
        return _SESSION_START + pd.Timedelta(
            days=int(self._rng.integers(0, delta_days)),
            hours=int(self._rng.integers(6, 22)),
        )

    def generate(self, n_users: Optional[int] = None) -> pd.DataFrame:
        """Generate interaction table for users.

        Returns:
            DataFrame with columns:
                user_id, item_id, item_type, signal, timestamp, session_id
        """
        users = self._users if n_users is None else self._users.head(n_users)
        records = []

        for _, user_row in users.iterrows():
            uid = user_row["user_id"]
            user_pw = self._user_persona_weights(user_row)
            p_idx = int(self._rng.choice(len(self._personas), p=user_pw))
            persona = self._personas[p_idx]
            n_sess = int(self._rng.integers(self._ipp_min, self._ipp_max + 1))
            prov_id = self._user_province_id(user_row)

            for s_idx in range(n_sess):
                ts = self._random_session_ts()
                session_id = f"{uid}_s{s_idx:04d}"

                # Choose item type for this session
                type_keys = list(persona.item_type_weights.keys())
                type_p = np.array(
                    [persona.item_type_weights[t] for t in type_keys], dtype=float
                )
                type_p /= type_p.sum()
                itype = ItemType(int(self._rng.choice(type_keys, p=type_p)))

                pool = self._pools.get(itype)
                if pool is None or len(pool.df) == 0:
                    continue

                # Build eligibility mask (geographic + temporal)
                mask = np.ones(len(pool.df), dtype=bool)

                geo_mask = self._geo.get_mask(prov_id, itype, persona.travel_radius_km)
                if geo_mask.sum() >= 1:
                    mask &= geo_mask

                if itype == ItemType.ATTRACTION:
                    t_mask = TemporalFilter.attraction_mask(pool.df, ts.dayofweek)
                    if t_mask.sum() >= 1:
                        mask &= t_mask
                elif itype == ItemType.EVENT:
                    t_mask = TemporalFilter.event_mask(pool.df, ts)
                    if t_mask.sum() >= 1:
                        mask &= t_mask

                if itype == ItemType.ARTICLE and "article_type_id" in pool.df.columns:
                    # article_type_id is 0-based idx; original typeId = idx + 1.
                    type_ids = pool.df["article_type_id"].values + 1

                    # Base: persona article-type preference
                    type_prefs = _PERSONA_ARTICLE_TYPE_PREFS.get(persona.name, {})
                    pref_weights = np.array(
                        [type_prefs.get(int(tid), 1.0) for tid in type_ids], dtype=float
                    )

                    # Additive: user travel_theme → article type bias
                    for theme in _parse_list(user_row.get("travel_theme", [])):
                        theme_bias = _THEME_ARTICLE_TYPE_BIAS.get(str(theme), {})
                        if theme_bias:
                            pref_weights += np.array(
                                [theme_bias.get(int(tid), 0.0) for tid in type_ids],
                                dtype=float,
                            )

                    # Language weight: home_country → is_thai preference
                    if "is_thai" in pool.df.columns:
                        country = str(user_row.get("home_country", "Other"))
                        thai_w, non_thai_w = _LANGUAGE_COUNTRY_PREF.get(country, (1.0, 1.5))
                        is_thai_vals = pool.df["is_thai"].fillna(0).to_numpy(dtype=float)
                        lang_weights = np.where(is_thai_vals > 0.5, thai_w, non_thai_w)
                        pref_weights = pref_weights * lang_weights

                    weights = pref_weights * mask.astype(float)
                    w_sum = weights.sum()
                    if w_sum > 0:
                        weights /= w_sum
                        item_idx = int(self._rng.choice(len(pool.df), p=weights))
                    else:
                        item_idx = int(pool.pop_model.sample(1, mask=mask, rng=self._rng)[0])
                else:
                    item_idx = int(pool.pop_model.sample(1, mask=mask, rng=self._rng)[0])
                item_id = str(pool.df.iloc[item_idx][pool.id_col])

                # Determine signal: click or view (exclusive, probability-based)
                r = float(self._rng.random())
                cp = persona.signal_probs.get("click", 0.15)
                signal = "click" if r < cp else "view"

                records.append({
                    "user_id": uid,
                    "item_id": item_id,
                    "item_type": int(itype),
                    "signal": signal,
                    "timestamp": ts,
                    "session_id": session_id,
                })

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# User profile augmentation
# ---------------------------------------------------------------------------

def augment_user_profiles(
    df: pd.DataFrame,
    target_n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap-augment user profiles to target_n rows.

    Preserves travel_style and travel_theme (keeps persona diversity).
    Perturbs demographics: age (±5 gaussian), home_province (random resample),
    home_country (20% random reassign), preferred_time_start_at (random).

    Args:
        df: raw user_profiles DataFrame with at least user_id, age,
            home_country, travel_style, travel_theme, home_province,
            preferred_time_start_at columns.
        target_n: desired total number of users (including original rows).
        seed: random seed.

    Returns:
        DataFrame with target_n rows, original users first then augmented.
    """
    from src.data.schema import HOME_COUNTRIES

    if len(df) >= target_n:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_aug = target_n - len(df)

    # Province pool: all unique home_province values that exist in the data
    province_pool = df["home_province"].dropna().unique().tolist()
    if not province_pool:
        province_pool = ["กรุงเทพมหานคร"]

    # Preferred time range (06:00–22:00 on random days)
    _t0 = pd.Timestamp("2024-01-01 06:00:00")
    _t1 = pd.Timestamp("2025-06-30 22:00:00")
    delta_minutes = int((_t1 - _t0).total_seconds() / 60)

    base_rows = df.sample(n=n_aug, replace=True, random_state=int(rng.integers(0, 2**31)))
    augmented = base_rows.copy().reset_index(drop=True)

    # New unique user_ids
    augmented["user_id"] = [f"aug_{i:06d}" for i in range(n_aug)]

    # Age perturbation: ±5 gaussian, clip to [18, 75]
    age_noise = rng.normal(0, 5, size=n_aug)
    augmented["age"] = np.clip(
        augmented["age"].values.astype(float) + age_noise, 18, 75
    ).astype(int)

    # home_province: random resample from pool (forces geographic diversity)
    augmented["home_province"] = rng.choice(province_pool, size=n_aug)

    # home_country: 80% keep, 20% random
    replace_mask = rng.random(size=n_aug) < 0.20
    augmented.loc[replace_mask, "home_country"] = rng.choice(
        HOME_COUNTRIES, size=int(replace_mask.sum())
    )

    # preferred_time_start_at: random
    rand_minutes = rng.integers(0, delta_minutes, size=n_aug)
    augmented["preferred_time_start_at"] = [
        _t0 + pd.Timedelta(minutes=int(m)) for m in rand_minutes
    ]

    return pd.concat([df, augmented], ignore_index=True)


# ---------------------------------------------------------------------------
# User behavior feature computation
# ---------------------------------------------------------------------------

def compute_behavior_features(
    interactions: pd.DataFrame,
    attraction_df: pd.DataFrame,
    event_df: pd.DataFrame,
    accommodation_df: Optional[pd.DataFrame] = None,
    article_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-user behavior features from interaction history.

    Attraction sub-categories occupy ITEM_CATEGORY_VOCAB positions 0..57;
    event categories occupy positions 58..68.

    Args:
        interactions: output of InteractionGenerator.generate()
        attraction_df: preprocessed attractions with place_id + sub_category_indices + province_id
        event_df: preprocessed events with event_id + category_indices + province_id
        accommodation_df: preprocessed accommodations with place_id + province_id (optional)
        article_df: preprocessed articles with article_id + article_type_id (optional)

    Returns:
        DataFrame indexed by user_id with columns:
            category_pref_indices         — int list, top-k ITEM_CATEGORY_VOCAB indices
            category_interaction_history  — float32 array (NUM_ITEM_CATEGORIES,)
            subcat_affinity               — float32 array (NUM_ATTRACTION_SUBCATS,)
            province_affinity             — float32 array (NUM_PROVINCES,)
            article_type_affinity         — float32 array (NUM_ARTICLE_TYPES,)
    """
    attr_lookup: Dict[str, List[int]] = {
        str(row["place_id"]): row["sub_category_indices"]
        for _, row in attraction_df.iterrows()
    }
    evt_lookup: Dict[str, List[int]] = {
        str(row["event_id"]): row["category_indices"]
        for _, row in event_df.iterrows()
    }

    # article_type_id lookup: article_id → sequential article type index
    art_type_lookup: Dict[str, int] = {}
    if article_df is not None and "article_type_id" in article_df.columns:
        for _, row in article_df.iterrows():
            art_type_lookup[str(row["article_id"])] = int(row["article_type_id"])

    # province_id lookup: item_id → sequential province index (already mapped by preprocessing)
    prov_lookup: Dict[str, int] = {}
    for df, id_col in [
        (attraction_df, "place_id"),
        (event_df, "event_id"),
        *([(accommodation_df, "place_id")] if accommodation_df is not None else []),
    ]:
        if "province_id" in df.columns:
            for _, row in df.iterrows():
                prov_lookup[str(row[id_col])] = int(row["province_id"])

    rows = []
    for uid, grp in interactions.groupby("user_id"):
        cat_hist = np.zeros(NUM_ITEM_CATEGORIES, dtype=np.float32)
        subcat_aff = np.zeros(NUM_ATTRACTION_SUBCATS, dtype=np.float32)
        prov_aff = np.zeros(NUM_PROVINCES, dtype=np.float32)
        art_type_aff = np.zeros(NUM_ARTICLE_TYPES, dtype=np.float32)

        for _, irow in grp.iterrows():
            w = float(_SIGNAL_WEIGHTS.get(str(irow["signal"]), 1.0))
            itype = int(irow["item_type"])
            iid = str(irow["item_id"])

            if itype == int(ItemType.ATTRACTION):
                # sub_category_indices values are 0..57 (sequential subcat idx)
                # They map directly to ITEM_CATEGORY_VOCAB positions 0..57
                for sc_idx in attr_lookup.get(iid, []):
                    if 0 <= sc_idx < NUM_ATTRACTION_SUBCATS:
                        cat_hist[sc_idx] += w
                        subcat_aff[sc_idx] += w

            elif itype == int(ItemType.EVENT):
                # category_indices values are 0..10 (sequential event cat idx)
                # Event categories occupy ITEM_CATEGORY_VOCAB positions 58..68
                for ec_idx in evt_lookup.get(iid, []):
                    vocab_idx = NUM_ATTRACTION_SUBCATS + ec_idx  # 58 + ec_idx
                    if 0 <= vocab_idx < NUM_ITEM_CATEGORIES:
                        cat_hist[vocab_idx] += w

            # Province affinity: accumulate for all geolocated item types
            prov_idx = prov_lookup.get(iid)
            if prov_idx is not None and 0 <= prov_idx < NUM_PROVINCES:
                prov_aff[prov_idx] += w

            # Article type affinity
            if itype == int(ItemType.ARTICLE):
                art_type_idx = art_type_lookup.get(iid)
                if art_type_idx is not None and 0 <= art_type_idx < NUM_ARTICLE_TYPES:
                    art_type_aff[art_type_idx] += w

        # Normalize to interaction rates
        total_cat = cat_hist.sum()
        cat_hist_norm = (cat_hist / total_cat).astype(np.float32) if total_cat > 0 else cat_hist
        total_sub = subcat_aff.sum()
        subcat_aff_norm = (subcat_aff / total_sub).astype(np.float32) if total_sub > 0 else subcat_aff
        total_prov = prov_aff.sum()
        prov_aff_norm = (prov_aff / total_prov).astype(np.float32) if total_prov > 0 else prov_aff
        total_art = art_type_aff.sum()
        art_type_aff_norm = (art_type_aff / total_art).astype(np.float32) if total_art > 0 else art_type_aff

        # Top-k category preference indices (categories with non-zero weight)
        pref_indices = [
            int(i) for i in np.argsort(-cat_hist)[:10] if cat_hist[i] > 0
        ]

        prov_pref_indices = [
            int(i) for i in np.argsort(-prov_aff)[:3] if prov_aff[i] > 0
        ]

        rows.append({
            "user_id": uid,
            "category_pref_indices": pref_indices,
            "category_interaction_history": cat_hist_norm,
            "subcat_affinity": subcat_aff_norm,
            "province_affinity": prov_aff_norm,
            "province_pref_indices": prov_pref_indices,
            "article_type_affinity": art_type_aff_norm,
        })

    return pd.DataFrame(rows).set_index("user_id")


# ---------------------------------------------------------------------------
# BehaviorDataset
# ---------------------------------------------------------------------------

@dataclass
class BehaviorDataset:
    """Container for generated interaction data with save/load helpers."""

    interactions: pd.DataFrame

    def save(self, path: str) -> None:
        """Save interactions to parquet, creating parent directories as needed."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.interactions.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str) -> "BehaviorDataset":
        """Load interactions from parquet."""
        return cls(interactions=pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self.interactions)

    def __repr__(self) -> str:
        if self.interactions.empty:
            return "BehaviorDataset(empty)"
        n_users = self.interactions["user_id"].nunique()
        n_items = self.interactions["item_id"].nunique()
        return f"BehaviorDataset({len(self.interactions)} interactions, {n_users} users, {n_items} items)"
