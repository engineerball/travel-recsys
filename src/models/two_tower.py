"""TwoTowerModel: combines UserTower + 4 ItemTowers for joint retrieval training."""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from src.data.schema import EMB_CONFIG, EmbeddingConfig, ItemType
from src.models.item_towers import (
    AccommodationTower,
    ArticleTower,
    AttractionTower,
    BaseItemTower,
    EventTower,
)
from src.models.user_tower import UserTower


class TwoTowerModel(keras.Model):
    """Joint two-tower retrieval model.

    Maintains one UserTower and four ItemTowers (Attraction, Accommodation,
    Event, Article) all sharing the same 64-dim output space. Training is done
    jointly across item types so the user embedding works with all item types.

    Usage::

        model = TwoTowerModel()
        user_emb, item_emb = model(user_inputs, item_inputs, ItemType.ATTRACTION)
        loss = model.compute_retrieval_loss(user_emb, item_emb, signal_weights)
    """

    def __init__(
        self,
        config: EmbeddingConfig = EMB_CONFIG,
        dropout_rate: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.config = config
        self.temperature = temperature

        self.user_tower = UserTower(config, dropout_rate)

        # Explicit attributes so Keras tracks all sub-layer weights
        self.attraction_tower = AttractionTower(config, dropout_rate)
        self.accommodation_tower = AccommodationTower(config, dropout_rate)
        self.event_tower = EventTower(config, dropout_rate)
        self.article_tower = ArticleTower(config, dropout_rate)

        # Convenience mapping (not used for weight tracking)
        self._item_towers: dict[ItemType, BaseItemTower] = {
            ItemType.ATTRACTION: self.attraction_tower,
            ItemType.ACCOMMODATION: self.accommodation_tower,
            ItemType.EVENT: self.event_tower,
            ItemType.ARTICLE: self.article_tower,
        }
        self.built = True

    def call(
        self,
        inputs: dict,
        training: bool = False,
        item_type: ItemType = ItemType.ATTRACTION,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode user and item features into the shared embedding space.

        Args:
            inputs: nested dict with keys "user" (UserFeatureSchema) and
                "item" (item type's FeatureSchema).
            training: passed to Dropout layers.
            item_type: which ItemType tower to use (keyword arg).

        Returns:
            (user_emb, item_emb) both float32 [B, output_dim] L2-normalized.
        """
        tower = self._item_towers.get(item_type)
        if tower is None:
            raise ValueError(f"Unknown item type: {item_type}")

        user_emb = self.user_tower(inputs["user"], training=training)
        item_emb = tower(inputs["item"], training=training)
        return user_emb, item_emb

    def compute_retrieval_loss(
        self,
        user_emb: tf.Tensor,
        item_emb: tf.Tensor,
        signal_weights: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """InfoNCE loss over in-batch negatives with optional per-sample weighting.

        Treats each (user_i, item_i) as the positive pair; all other items in
        the batch are negatives. Embeddings must already be L2-normalized.

        Args:
            user_emb: float32 [B, D] L2-normalized user embeddings.
            item_emb: float32 [B, D] L2-normalized item embeddings.
            signal_weights: optional float32 [B] per-sample importance weights
                (e.g. 1.0=view, 3.0=click). Uniform if None.

        Returns:
            Scalar loss tensor.
        """
        # [B, B] cosine similarity matrix (already unit norm → dot = cosine)
        scores = tf.matmul(user_emb, item_emb, transpose_b=True) / self.temperature

        batch_size = tf.shape(user_emb)[0]
        labels = tf.eye(batch_size)  # [B, B] one-hot diagonal = positive pairs

        # Per-sample cross-entropy: each row is a B-class classification
        per_sample = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=scores
        )  # [B]

        if signal_weights is not None:
            weights = tf.cast(signal_weights, tf.float32)
            # Normalize weights so they sum to batch_size (preserve loss scale)
            weights = weights / (tf.reduce_mean(weights) + 1e-8)
            return tf.reduce_mean(weights * per_sample)

        return tf.reduce_mean(per_sample)

    def get_user_embedding(
        self, user_inputs: dict, training: bool = False
    ) -> tf.Tensor:
        """Convenience: encode a batch of users."""
        return self.user_tower(user_inputs, training=training)

    def get_item_embedding(
        self,
        item_inputs: dict,
        item_type: ItemType,
        training: bool = False,
    ) -> tf.Tensor:
        """Convenience: encode a batch of items."""
        tower = self._item_towers.get(item_type)
        if tower is None:
            raise ValueError(f"Unknown item type: {item_type}")
        return tower(item_inputs, training=training)
