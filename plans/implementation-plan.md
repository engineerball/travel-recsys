# Implementation Plan — Two-Tower Travel Recsys

## Phase 1: Data Foundation

### ✅ 1.1 `src/data/schema.py` — DONE
- `UserFeatureSchema` — profile features + 3 behavior features (category_pref_indices, category_interaction_history, subcat_affinity)
- `*FeatureSchema` per type — 4 types (Attraction, Accommodation, Event, Article); no Restaurant/Shop (no source data)
- `VocabRegistry` — `encode(field, value)`, `size(field)`, `fields()`; covers all 11 categorical vocabs
- `EmbeddingConfig` — frozen dataclass with all embedding dims
- `ITEM_CATEGORY_VOCAB` — unified 69-dim category space (58 attraction subcats + 11 event categories) for behavior features

> **Deviations from original plan:**
> - 4 item types not 6 — no restaurant/shop data in `data/raw/`
> - `home_country Embedding(7,...)` not `Embedding(50,...)` — real data has 7 relevant countries
> - `build_vocab_from_data(df)` not needed — vocabs are static (derived from full dataset investigation)

### ✅ 1.2 `src/data/preprocessing.py` — DONE
- `ZScoreNormalizer` — fit/transform/fit_transform; pre-fit defaults for age (34.1±11), star (3.86±0.77), duration (8.91±34.7)
- `MultiHotEncoder` — `encode`, `encode_unique`, `encode_series`; reusable with any vocab list
- `CyclicalEncoder` — `encode(value)`, `encode_series(series)`; parameterised by period
- `preprocess_users(df)` — produces profile features; behavior features added later by behavior_generator.py
- `preprocess_attractions(df)` — sub_category_indices, days_open_vector (Thai day name parsing), is_free, log_view_count
- `preprocess_accommodations(df)` — amenity_indices (facilities+services union), price_tier (6-bin THB), star_rating Z-score (null→mean fill)
- `preprocess_events(df)` — category_indices, duration_days Z-score (negatives clipped), month sin/cos
- `preprocess_articles(df)` — article_type_id, pub_month sin/cos
- `load_and_preprocess(item_type, raw_dir)` — convenience loader + dispatcher
- All item preprocessors produce `text_for_embed` column (null-safe, HTML stripped) for `scripts/embed_items.py`

> **Intentional omission:** `TextEmbedProjector` not in preprocessing — text embeddings are pre-computed offline by `scripts/embed_items.py` and joined in.

### ✅ 1.3 `src/data/behavior_generator.py` — DONE
- `PersonaConfig` — config สำหรับแต่ละ persona (ดึงจาก config.yaml)
- `PopularityModel` — power law distribution เหนือ item corpus
- `GeographicFilter` — filter items ตาม user home province
- `TemporalFilter` — filter ตาม opening hours + day_of_week ของ session (attractions only; events use startDate)
- `InteractionGenerator` — สร้าง (user, item, signal, timestamp) tuples
- `BehaviorDataset` — save/load generated interactions เป็น parquet
- Must also compute and attach behavior features to user profiles:
  - `category_pref_indices` — int[] into ITEM_CATEGORY_VOCAB (69-dim)
  - `category_interaction_history` — float32[69] interaction rate per category
  - `subcat_affinity` — float32[58] weighted affinity per attraction sub-category

**Output format:**
```
user_id | item_id | item_type | signal | timestamp | session_id
```

---

## ✅ Phase 2: Models

### ✅ 2.1 `src/models/user_tower.py` — DONE
- `UserTower` — profile + behavior features → LayerNorm → Dense(256) → Dense(128) → Dense(64) → L2-norm
- Variable-length multi-hot indices padded with -1; `_masked_mean_pool` handles masking
- Behavior projections: `category_interaction_history (69,) → Dense(32)`, `subcat_affinity (58,) → Dense(32)`

### ✅ 2.2 `src/models/item_towers.py` — DONE
- `BaseItemTower` — shared: `text_proj (256→64)`, `item_type_emb Embedding(4,8)`, `province_emb Embedding(77,16)`, MLP head
- `AttractionTower` — sub_category_indices (Embedding 58×16 + MeanPool), days_open_vector, is_free, log_view_count
- `AccommodationTower` — amenity_indices (Embedding 24×8 + MeanPool), price_tier_emb (6×8), is_price_missing, star_rating_norm, log_view_count
- `EventTower` — category_indices (Embedding 11×8 + MeanPool), duration_days_norm, month sin/cos
- `ArticleTower` — article_type_emb (12×8), pub_month sin/cos; no province_id
- `get_item_tower(item_type, config)` — factory

> **Deviation:** `call(inputs, training, item_type=...)` uses nested dict `{"user": ..., "item": ...}` — Keras rejects non-tensor positional args.

### ✅ 2.3 `src/models/two_tower.py` — DONE
- `TwoTowerModel` — holds UserTower + 4 ItemTowers as explicit attributes (Keras weight tracking)
- `call(inputs, training, item_type)` → `(user_emb, item_emb)` both `[B, 64]` L2-normalized
- `compute_retrieval_loss(user_emb, item_emb, signal_weights)` — InfoNCE over in-batch negatives; signal weights normalized to preserve loss scale
- `get_user_embedding`, `get_item_embedding` — convenience methods for offline index building

---

## ✅ Phase 3: Training

### ✅ 3.1 `src/training/losses.py` — DONE
- `infonce_loss(user_emb, item_emb, temperature, signal_weights)` — standard InfoNCE over in-batch negatives; optional float[B] signal weights normalised so mean=1 (preserves loss scale)
- `mixed_negative_loss(user_emb, item_emb, item_types, temperature)` — same InfoNCE base + hard negative mining: top-10% highest-scoring cross-type items per row get +0.5 log-space logit boost (≈1.65× denominator weight), implemented via `tf.one_hot` scatter to avoid ragged ops

### ✅ 3.2 `src/training/dataset.py` — DONE
- `stratified_batch_sampler(df, batch_size, item_types, type_col, seed)` — standalone infinite generator over interaction DataFrame; yields `pd.DataFrame` batches with `batch_size // num_types` rows per type
- `StratifiedInteractionDataset` — pre-materialises all interaction features into flat numpy arrays (fast int-indexed lookups); context (day/hour sin/cos) derived from interaction timestamp; unified item feature dict (union of all 4 schemas, zero/−1 padding for N/A type fields); `build(split, batch_size, val_fraction, seed)` returns infinite train or finite val `tf.data.Dataset`; output signature inferred from pre-materialised array dtypes/shapes
- Padding sizes: `MAX_TRAVEL_STYLES=5`, `MAX_TRAVEL_THEMES=15`, `MAX_CAT_PREF=10`, `MAX_SUBCAT_INDICES=10`, `MAX_CATEGORY_INDICES=3`, `MAX_AMENITY_INDICES=24`

### ✅ 3.3 `src/training/trainer.py` — DONE
- `TwoTowerTrainer(model, config, checkpoint_dir)` — Adam optimiser, optional `clipnorm`
- `train_step(batch)` — single `GradientTape` covers all 4 towers; boolean-masks batch per type, forward-passes each tower, concatenates all embeddings, calls `mixed_negative_loss` once; returns `loss` + `loss_{type}` per-type InfoNCE (diagnostic, no extra backward pass)
- `eval_step(batch)` — same forward without tape
- `train(train_ds, val_ds, epochs, steps_per_epoch)` — calls `next(iter(train_ds))` per step; prints loss + elapsed time per epoch; saves `model.save_weights` checkpoint per epoch; returns history dict
- `save_towers(output_dir)` — saves `user_tower` + `{type}_tower` weights to separate subdirs for serving

> **Design note:** single joint `GradientTape` is mandatory — training towers separately would pull user embedding in conflicting directions.

---

## Phase 4: Serving

### 4.1 `src/serving/retrieval.py` — TODO
```python
class FAISSRetriever:
    def build_index(self, item_embeddings):
        # สร้าง FAISS index จาก item embeddings ทั้งหมด

    def query(self, user_embedding, top_k=100):
        # ANN search → return top-K item IDs + scores
```

---

## Phase 5: Scripts

### `scripts/embed_items.py` — TODO
```
1. Load item data จาก data/raw/ (4 types)
2. สร้าง text_for_embed ผ่าน preprocess_* (already has this column)
3. Call Gemma embedding API (batch)
4. Truncate ที่ 256 dim (MRL)
5. Save เป็น numpy array + item_id mapping per type
```

### `scripts/generate_behavior.py` — TODO
```
1. Load preprocessed item features + user profiles
2. Run BehaviorDataset.generate(config)
3. Attach behavior features to user profiles (category_pref_indices, category_interaction_history, subcat_affinity)
4. Save interactions → data/generated/interactions.parquet
5. Save enriched user features → data/generated/user_features.parquet
```

### `scripts/train.py` — TODO
```
1. Load config
2. Load processed data
3. Build tf.data datasets (train/val)
4. Initialize TwoTowerModel
5. Run TwoTowerTrainer.train()
6. Save towers แยกกัน (user_tower, item_towers/)
7. Build FAISS index จาก item embeddings
```

---

## Data Requirements — RESOLVED

actual `data/raw/` structure:

```
data/raw/
├── destinations/    ← Attractions  (8,632 rows, parquet)
├── activities/      ← Events       (21,376 rows, parquet)
├── accommodations/  ← Hotels       (2,972 rows, parquet)
├── articles/        ← Articles     (1,559 rows, parquet)
└── user_profiles/   ← Users        (1,000 rows, parquet)
```

No restaurants or shops in source data.

---

## Open Questions

- [ ] Gemma MRL: ใช้ model string อะไร? API key มาจากที่ไหน?
- [x] Province/region: ~~item data มี location field ไหม?~~ → `province_id` (int32) already extracted in parquet; 77 provinces mapped. No region_id in data — dropped.
- [x] Accommodation amenities: → `facilities` + `services` arrays of `{code, name}` dicts; 24-code vocab built.
- [ ] Article: recommendation ขึ้นมาปนกับ places ใน same feed หรือแยก section?
- [ ] MMoE: อยากให้อยู่ใน codebase นี้ด้วย หรือแยก ranking model?
