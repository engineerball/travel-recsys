"""TwoTowerTrainer: training loop, gradient updates, checkpointing.

Usage::

    trainer = TwoTowerTrainer(model, config)
    history = trainer.train(train_ds, val_ds, epochs=20, steps_per_epoch=200)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.data.schema import ItemType
from src.training.losses import infonce_loss, mixed_negative_loss


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for ``warmup_steps``, then cosine decay to ``min_lr``."""

    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Linear warmup phase
        warmup_lr = self.max_lr * (step / tf.maximum(warmup, 1.0))

        # Cosine decay phase
        decay_steps = total - warmup
        completed = tf.maximum(step - warmup, 0.0) / tf.maximum(decay_steps, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * tf.minimum(completed, 1.0))
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


class TwoTowerTrainer:
    """Joint trainer for TwoTowerModel across all 4 item types.

    Each training step:
    1. Separates the mixed batch by item type using boolean masking.
    2. Forward-passes each type through its tower inside a single GradientTape.
    3. Concatenates all user/item embeddings and computes mixed_negative_loss.
    4. Applies gradients once over all towers jointly.

    This joint update is what keeps the user embedding aligned with all item
    types simultaneously — do NOT train towers separately.

    Args:
        model: TwoTowerModel instance.
        config: dict with keys ``learning_rate``, ``temperature``, optional
            ``clipnorm`` (gradient clip by norm), ``checkpoint_dir``.
        checkpoint_dir: directory for saving ``model.save_weights`` snapshots.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: Optional[Dict] = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        cfg = config or {}
        self.model = model
        self.temperature: float = float(cfg.get("temperature", 0.07))
        self.checkpoint_dir = checkpoint_dir
        self._cfg = cfg  # saved so train() can build the LR schedule after knowing total steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_user(user_features: Dict[str, tf.Tensor], mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {k: tf.boolean_mask(v, mask) for k, v in user_features.items()}

    @staticmethod
    def _mask_item(item_features: Dict[str, tf.Tensor], mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        # Pass ALL item feature keys; each tower ignores irrelevant ones.
        return {k: tf.boolean_mask(v, mask) for k, v in item_features.items()}

    # ------------------------------------------------------------------
    # Train / eval steps
    # ------------------------------------------------------------------

    @tf.function
    def train_step(self, batch: tuple) -> tf.Tensor:
        """Single gradient update step compiled as TF graph (no Python overhead).

        Stratified batches guarantee all item types present — no empty-mask guard needed.

        Returns:
            Scalar loss tensor (convert to float() in the caller).
        """
        user_features, item_features, item_types, _signal_weights = batch

        with tf.GradientTape() as tape:
            all_u: List[tf.Tensor] = []
            all_i: List[tf.Tensor] = []
            all_types_list: List[tf.Tensor] = []

            for itype in ItemType:  # unrolled at trace time
                mask = tf.equal(item_types, int(itype))
                u_batch = self._mask_user(user_features, mask)
                i_batch = self._mask_item(item_features, mask)
                u_emb, i_emb = self.model(
                    {"user": u_batch, "item": i_batch},
                    training=True,
                    item_type=itype,
                )
                all_u.append(u_emb)
                all_i.append(i_emb)
                all_types_list.append(tf.boolean_mask(item_types, mask))

            joint_u = tf.concat(all_u, axis=0)
            joint_i = tf.concat(all_i, axis=0)
            joint_types = tf.concat(all_types_list, axis=0)
            loss = mixed_negative_loss(joint_u, joint_i, joint_types, self.temperature)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def eval_step(self, batch: tuple) -> tf.Tensor:
        """Forward pass without gradient update, compiled as TF graph."""
        user_features, item_features, item_types, _signal_weights = batch

        all_u: List[tf.Tensor] = []
        all_i: List[tf.Tensor] = []
        all_types_list: List[tf.Tensor] = []

        for itype in ItemType:  # unrolled at trace time
            mask = tf.equal(item_types, int(itype))
            u_batch = self._mask_user(user_features, mask)
            i_batch = self._mask_item(item_features, mask)
            u_emb, i_emb = self.model(
                {"user": u_batch, "item": i_batch},
                training=False,
                item_type=itype,
            )
            all_u.append(u_emb)
            all_i.append(i_emb)
            all_types_list.append(tf.boolean_mask(item_types, mask))

        joint_u = tf.concat(all_u, axis=0)
        joint_i = tf.concat(all_i, axis=0)
        joint_types = tf.concat(all_types_list, axis=0)
        return mixed_negative_loss(joint_u, joint_i, joint_types, self.temperature)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def _build_optimizer(self, epochs: int, steps_per_epoch: int) -> None:
        """Instantiate Adam with WarmupCosineDecay schedule."""
        cfg = self._cfg
        max_lr = float(cfg.get("learning_rate", 1e-3))
        min_lr = max_lr * float(cfg.get("min_lr_ratio", 0.01))
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        clipnorm = cfg.get("clipnorm", None)

        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        if warmup_steps > 0 or cfg.get("min_lr_ratio"):
            schedule = WarmupCosineDecay(
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            lr_arg = schedule
        else:
            lr_arg = max_lr

        opt_kwargs: Dict = {"learning_rate": lr_arg}
        if clipnorm is not None:
            opt_kwargs["clipnorm"] = float(clipnorm)
        self.optimizer = tf.keras.optimizers.Adam(**opt_kwargs)

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 20,
        steps_per_epoch: int = 200,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """Run the full training loop with early stopping on val_loss.

        Saves best val checkpoint to ``{checkpoint_dir}/best/weights.weights.h5``.
        Stops early if val_loss has not improved by ``min_delta`` for ``patience``
        consecutive epochs. Early stopping only activates when val_dataset is provided.

        Args:
            train_dataset: infinite tf.data.Dataset.
            val_dataset: finite validation dataset; None to skip validation.
            epochs: maximum training epochs.
            steps_per_epoch: gradient steps per epoch.
            patience: epochs without improvement before stopping.
            min_delta: minimum improvement to count as progress.

        Returns:
            History dict: ``{"train_loss": [...], "val_loss": [...]}``.
        """
        self._build_optimizer(epochs, steps_per_epoch)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        best_val = float("inf")
        epochs_no_improve = 0
        best_epoch = 0
        best_ckpt = os.path.join(self.checkpoint_dir, "best", "weights.weights.h5")
        os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)

        train_iter = iter(train_dataset)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # --- Training ---
            step_losses: List[float] = []
            for _ in range(steps_per_epoch):
                batch = next(train_iter)
                loss_tensor = self.train_step(batch)
                step_losses.append(float(loss_tensor))

            avg_train = float(np.mean(step_losses))
            history["train_loss"].append(avg_train)

            # --- Validation ---
            avg_val: Optional[float] = None
            if val_dataset is not None:
                val_losses = [float(self.eval_step(b)) for b in val_dataset]
                avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
                history["val_loss"].append(avg_val)

            elapsed = time.time() - epoch_start

            # --- Early stopping ---
            stop_marker = ""
            if avg_val is not None:
                if avg_val < best_val - min_delta:
                    best_val = avg_val
                    best_epoch = epoch
                    epochs_no_improve = 0
                    self.model.save_weights(best_ckpt)
                    stop_marker = " *"
                else:
                    epochs_no_improve += 1

            if avg_val is not None:
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
                    f"({elapsed:.1f}s){stop_marker}"
                )
            else:
                print(
                    f"Epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}  ({elapsed:.1f}s)"
                )

            # --- Per-epoch checkpoint ---
            ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}", "weights.weights.h5")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            self.model.save_weights(ckpt_path)

            if val_dataset is not None and epochs_no_improve >= patience:
                print(f"\nEarly stop at epoch {epoch} — best val={best_val:.4f} @ epoch {best_epoch}")
                break

        if val_dataset is not None:
            print(f"Best checkpoint: {best_ckpt}  (epoch {best_epoch}, val={best_val:.4f})")

        return history

    # ------------------------------------------------------------------
    # Convenience: save / load towers
    # ------------------------------------------------------------------

    def save_towers(self, output_dir: str) -> None:
        """Save user tower and all item towers as separate weight files.

        Compatible with ``get_user_embedding`` / ``get_item_embedding`` at
        serving time — load each sub-model and call ``load_weights``.

        Args:
            output_dir: directory under which sub-directories are created.
        """
        os.makedirs(output_dir, exist_ok=True)

        user_tower_dir = os.path.join(output_dir, "user_tower")
        os.makedirs(user_tower_dir, exist_ok=True)
        self.model.user_tower.save_weights(os.path.join(user_tower_dir, "weights.weights.h5"))

        for itype in ItemType:
            tower_attr = f"{itype.name.lower()}_tower"
            tower = getattr(self.model, tower_attr, None)
            if tower is not None:
                tower_dir = os.path.join(output_dir, "item_towers", itype.name.lower())
                os.makedirs(tower_dir, exist_ok=True)
                tower.save_weights(os.path.join(tower_dir, "weights.weights.h5"))

        print(f"Towers saved to {output_dir}")
