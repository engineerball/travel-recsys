"""Item towers: one per item type, all output 64-dim shared embedding space.

BaseItemTower provides shared layers (text projection, item-type embedding, MLP head).
Subclasses add type-specific embedding and feature extraction in _build_features().

All variable-length index inputs are padded with -1. Valid indices >= 0.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Embedding

from src.data.schema import (
    EMB_CONFIG,
    EmbeddingConfig,
    ItemType,
    NUM_AMENITY_CODES,
    NUM_ARTICLE_TYPES,
    NUM_ATTRACTION_SUBCATS,
    NUM_EVENT_CATEGORIES,
    NUM_ITEM_TYPES,
    NUM_PRICE_TIERS,
    NUM_PROVINCES,
    NUM_REGIONS,
)


class BaseItemTower(keras.Model):
    """Shared base for all item towers.

    Every item tower:
    - Projects text_embed [B, 256] → [B, 64] via trainable Dense
    - Embeds item_type_id [B] → [B, 8]
    - Passes (base_features ++ type_features) through a 256→128→64 MLP head
    - Returns L2-normalized embedding [B, 64]
    """

    def __init__(
        self,
        config: EmbeddingConfig = EMB_CONFIG,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = config

        # Shared base feature layers
        self.text_proj = Dense(config.output_dim)  # 256 → 64
        self.item_type_emb = Embedding(NUM_ITEM_TYPES, config.item_type_emb_dim)

        # Shared province + region embeddings used by Attraction / Accommodation / Event
        self.province_emb = Embedding(NUM_PROVINCES, config.province_emb_dim)
        self.region_emb = Embedding(NUM_REGIONS, 8)

        # Tower MLP head (subclasses reuse this)
        self.dense1 = Dense(256, activation="relu")
        self.dense2 = Dense(128, activation="relu")
        self.output_proj = Dense(config.output_dim)
        self.dropout = Dropout(dropout_rate)
        self.built = True

    @staticmethod
    def _masked_mean_pool(indices: tf.Tensor, emb_layer: Embedding) -> tf.Tensor:
        """Mean pool over valid (non -1 padding) embedding indices.

        Args:
            indices: int32 [B, L], -1 = padding.
            emb_layer: Keras Embedding layer.

        Returns:
            float32 [B, D].
        """
        mask = tf.cast(indices >= 0, tf.float32)           # [B, L]
        safe = tf.maximum(indices, 0)
        emb = emb_layer(safe)                               # [B, L, D]
        emb = emb * mask[:, :, tf.newaxis]
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.reduce_sum(emb, axis=1) / tf.maximum(count, 1.0)

    def _base_features(self, inputs: dict) -> tf.Tensor:
        """Compute shared base feature vector for any item type."""
        text = self.text_proj(tf.cast(inputs["text_embed"], tf.float32))  # [B, 64]
        item_type = self.item_type_emb(inputs["item_type_id"])             # [B, 8]
        return tf.concat([text, item_type], axis=-1)                       # [B, 72]

    def _type_features(self, inputs: dict) -> tf.Tensor:
        """Return type-specific feature vector. Overridden by each subclass."""
        raise NotImplementedError

    def _tower_head(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Shared MLP head: Dense(256) → Dense(128) → Dense(64) → L2-norm."""
        x = self.dropout(self.dense1(x), training=training)
        x = self.dropout(self.dense2(x), training=training)
        x = self.output_proj(x)
        return tf.nn.l2_normalize(x, axis=-1)

    def call(self, inputs: dict, training: bool = False) -> tf.Tensor:
        """Encode item features into the shared embedding space.

        Args:
            inputs: dict with keys matching the item's FeatureSchema field names.
            training: passed to Dropout layers.

        Returns:
            float32 [B, output_dim] L2-normalized item embedding.
        """
        base = self._base_features(inputs)
        specific = self._type_features(inputs)
        x = tf.concat([base, specific], axis=-1)
        return self._tower_head(x, training)


class AttractionTower(BaseItemTower):
    """Tower for attraction (destinations) items.

    Type-specific features:
        sub_category_indices int32 [B, ≤10]  padded with -1
        province_id          int32 [B]
        days_open_vector     float32 [B, 7]
        is_free              float32 [B]
        log_view_count       float32 [B]
    """

    def __init__(
        self,
        config: EmbeddingConfig = EMB_CONFIG,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__(config, dropout_rate)
        self.subcat_emb = Embedding(NUM_ATTRACTION_SUBCATS, config.attraction_subcat_emb_dim)

    def _type_features(self, inputs: dict) -> tf.Tensor:
        subcat = self._masked_mean_pool(inputs["sub_category_indices"], self.subcat_emb)
        province = self.province_emb(inputs["province_id"])
        region = self.region_emb(inputs["region_id"])
        days = tf.cast(inputs["days_open_vector"], tf.float32)
        is_free = tf.expand_dims(tf.cast(inputs["is_free"], tf.float32), -1)
        log_view = tf.expand_dims(tf.cast(inputs["log_view_count"], tf.float32), -1)
        return tf.concat([subcat, province, region, days, is_free, log_view], axis=-1)


class AccommodationTower(BaseItemTower):
    """Tower for accommodation (hotel) items.

    Type-specific features:
        amenity_indices  int32 [B, ≤24]  padded with -1
        province_id      int32 [B]
        price_tier_id    int32 [B]
        is_price_missing float32 [B]
        star_rating_norm float32 [B]
        log_view_count   float32 [B]
    """

    def __init__(
        self,
        config: EmbeddingConfig = EMB_CONFIG,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__(config, dropout_rate)
        self.amenity_emb = Embedding(NUM_AMENITY_CODES, config.amenity_emb_dim)
        self.price_tier_emb = Embedding(NUM_PRICE_TIERS, config.price_tier_emb_dim)

    def _type_features(self, inputs: dict) -> tf.Tensor:
        amenity = self._masked_mean_pool(inputs["amenity_indices"], self.amenity_emb)
        province = self.province_emb(inputs["province_id"])
        region = self.region_emb(inputs["region_id"])
        price_tier = self.price_tier_emb(inputs["price_tier_id"])
        is_missing = tf.expand_dims(tf.cast(inputs["is_price_missing"], tf.float32), -1)
        star = tf.expand_dims(tf.cast(inputs["star_rating_norm"], tf.float32), -1)
        log_view = tf.expand_dims(tf.cast(inputs["log_view_count"], tf.float32), -1)
        return tf.concat([amenity, province, region, price_tier, is_missing, star, log_view], axis=-1)


class EventTower(BaseItemTower):
    """Tower for event (activities) items.

    Type-specific features:
        category_indices   int32 [B, ≤3]  padded with -1
        province_id        int32 [B]
        duration_days_norm float32 [B]
        month_sin          float32 [B]
        month_cos          float32 [B]
    """

    def __init__(
        self,
        config: EmbeddingConfig = EMB_CONFIG,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__(config, dropout_rate)
        self.category_emb = Embedding(NUM_EVENT_CATEGORIES, config.event_category_emb_dim)

    def _type_features(self, inputs: dict) -> tf.Tensor:
        category = self._masked_mean_pool(inputs["category_indices"], self.category_emb)
        province = self.province_emb(inputs["province_id"])
        region = self.region_emb(inputs["region_id"])
        duration = tf.expand_dims(tf.cast(inputs["duration_days_norm"], tf.float32), -1)
        month_sin = tf.expand_dims(tf.cast(inputs["month_sin"], tf.float32), -1)
        month_cos = tf.expand_dims(tf.cast(inputs["month_cos"], tf.float32), -1)
        return tf.concat([category, province, region, duration, month_sin, month_cos], axis=-1)


class ArticleTower(BaseItemTower):
    """Tower for article items.

    Type-specific features:
        article_type_id int32 [B]
        pub_month_sin   float32 [B]
        pub_month_cos   float32 [B]

    No province_id in article data.
    """

    def __init__(
        self,
        config: EmbeddingConfig = EMB_CONFIG,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__(config, dropout_rate)
        self.article_type_emb = Embedding(NUM_ARTICLE_TYPES, config.article_type_emb_dim)

    def _type_features(self, inputs: dict) -> tf.Tensor:
        article_type = self.article_type_emb(inputs["article_type_id"])
        pub_sin = tf.expand_dims(tf.cast(inputs["pub_month_sin"], tf.float32), -1)
        pub_cos = tf.expand_dims(tf.cast(inputs["pub_month_cos"], tf.float32), -1)
        return tf.concat([article_type, pub_sin, pub_cos], axis=-1)


def get_item_tower(
    item_type: ItemType,
    config: EmbeddingConfig = EMB_CONFIG,
    dropout_rate: float = 0.1,
) -> BaseItemTower:
    """Factory: return the correct item tower for the given item type."""
    towers = {
        ItemType.ATTRACTION: AttractionTower,
        ItemType.ACCOMMODATION: AccommodationTower,
        ItemType.EVENT: EventTower,
        ItemType.ARTICLE: ArticleTower,
    }
    cls = towers.get(item_type)
    if cls is None:
        raise ValueError(f"No tower for item type: {item_type}")
    return cls(config, dropout_rate)
