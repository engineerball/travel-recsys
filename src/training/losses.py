"""Loss functions for two-tower retrieval training.

infonce_loss      — standard InfoNCE over in-batch negatives
mixed_negative_loss — InfoNCE + hard negative mining from cross-type items
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


def infonce_loss(
    user_emb: tf.Tensor,
    item_emb: tf.Tensor,
    temperature: float = 0.07,
    signal_weights: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """InfoNCE over in-batch negatives with optional per-sample weighting.

    Treats (user_i, item_i) as positives; all other items in the batch
    are negatives. Embeddings must be L2-normalised.

    Args:
        user_emb: float32 [B, D].
        item_emb: float32 [B, D].
        temperature: softmax temperature; lower = sharper.
        signal_weights: optional float32 [B] per-sample importance weights
            (e.g. view=1, click=3). Normalised to preserve loss scale.

    Returns:
        Scalar loss tensor.
    """
    scores = tf.matmul(user_emb, item_emb, transpose_b=True) / temperature  # [B, B]
    B = tf.shape(user_emb)[0]
    labels = tf.eye(B)  # [B, B] one-hot diagonal

    per_sample = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=scores)  # [B]

    if signal_weights is not None:
        w = tf.cast(signal_weights, tf.float32)
        w = w / (tf.reduce_mean(w) + 1e-8)  # normalise so mean weight = 1
        return tf.reduce_mean(w * per_sample)

    return tf.reduce_mean(per_sample)


# Log-space boost applied to hard negative logits (≈ 1.65× weight in denominator)
_HARD_NEG_BOOST: float = 0.5


def mixed_negative_loss(
    user_emb: tf.Tensor,
    item_emb: tf.Tensor,
    item_types: tf.Tensor,
    temperature: float = 0.07,
) -> tf.Tensor:
    """InfoNCE with mixed negative strategy.

    Negative composition per positive:
    - ~60%: in-batch same-type negatives (naturally present from stratified batch)
    - ~30%: in-batch cross-type negatives (naturally present)
    - ~10%: hard negatives — top-K highest-scoring cross-type items get a
            logit boost so they dominate the denominator more strongly.

    Args:
        user_emb:   float32 [B, D] L2-normalised.
        item_emb:   float32 [B, D] L2-normalised.
        item_types: int32   [B] item type id per sample.
        temperature: softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    B = tf.shape(user_emb)[0]
    scores = tf.matmul(user_emb, item_emb, transpose_b=True) / temperature  # [B, B]
    eye = tf.eye(B)  # [B, B]

    # Cross-type mask: True when types differ AND off-diagonal
    types_i = tf.expand_dims(tf.cast(item_types, tf.int32), 1)   # [B, 1]
    types_j = tf.expand_dims(tf.cast(item_types, tf.int32), 0)   # [1, B]
    same_type = tf.cast(tf.equal(types_i, types_j), tf.float32)  # [B, B]
    off_diag = 1.0 - eye
    cross_type = (1.0 - same_type) * off_diag                    # [B, B]

    # Hard negatives: top-K cross-type items by current similarity score
    # Mask non-cross-type entries to −∞ so top_k only picks cross-type
    masked_scores = scores - 1e9 * (1.0 - cross_type)            # [B, B]
    k = tf.maximum(1, tf.cast(tf.cast(B, tf.float32) * 0.1, tf.int32))
    _, hard_idx = tf.math.top_k(masked_scores, k=k)              # [B, k]

    # Convert hard_idx → dense [B, B] mask via one_hot
    hard_mask = tf.reduce_sum(
        tf.one_hot(hard_idx, depth=B, dtype=tf.float32), axis=1  # [B, k, B] → [B, B]
    )
    hard_mask = tf.minimum(hard_mask, 1.0)  # clip duplicate counts to binary

    # Boost hard negative logits (log-space ≈ multiplying probability by e^boost)
    augmented = scores + _HARD_NEG_BOOST * hard_mask
    # Don't alter the positive diagonal
    augmented = tf.where(tf.cast(eye, tf.bool), scores, augmented)

    labels = eye
    per_sample = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=augmented)
    return tf.reduce_mean(per_sample)
