# Implementation Plan — Two-Tower Travel Recsys

## Phase 1: Data Foundation

### 1.1 `src/data/schema.py`
- `UserFeatureSchema` — dataclass หรือ dict สำหรับ user features
- `ItemFeatureSchema` per type (Attraction, Restaurant, Events, Accommodation, Shop, Article)
- `VocabRegistry` — mapping จาก string → int สำหรับทุก categorical feature
- Helper: `build_vocab_from_data(df)` — fit vocab จาก real item data

### 1.2 `src/data/preprocessing.py`
- `AgePreprocessor` — Z-score normalize หรือ bucket เป็น life stage
- `MultiHotToIndices` — แปลง multi-hot vector → list of active indices
- `CyclicalEncoder` — sin/cos encoding สำหรับ hour/day/month
- `TextEmbedProjector` — load Gemma MRL embedding, truncate ที่ 256 dim
- `ItemPreprocessor` — pipeline รวม per item type
- `UserPreprocessor` — pipeline สำหรับ user features

### 1.3 `src/data/behavior_generator.py`
- `PersonaConfig` — config สำหรับแต่ละ persona (ดึงจาก config.yaml)
- `PopularityModel` — power law distribution เหนือ item corpus
- `GeographicFilter` — filter items ตาม user home region + travel radius
- `TemporalFilter` — filter ตาม opening hours + day_of_week ของ session
- `InteractionGenerator` — สร้าง (user, item, signal, timestamp) tuples
- `BehaviorDataset` — save/load generated interactions เป็น parquet

**Output format:**
```
user_id | item_id | item_type | signal | timestamp | session_id
```

---

## Phase 2: Models

### 2.1 `src/models/user_tower.py`
```python
class UserTower(keras.Model):
    # Inputs:
    #   age_norm            float (1,)
    #   home_country_id     int   → Embedding(50, 16)
    #   travel_style_ids    int[] → Embedding(12, 16) → MeanPool
    #   travel_theme_ids    int[] → Embedding(20, 16) → MeanPool
    #   category_pref_ids   int[] → Embedding(30, 16) → MeanPool
    #   category_history    float (30,) → Dense(32)
    #   subcat_affinity     float (80,) → Dense(32)
    #   ctx_day_sin/cos     float (2,)
    #   ctx_hour_sin/cos    float (2,)
    #
    # Architecture: Concat → LayerNorm → Dense(256, relu) → Dense(128, relu) → Dense(64)
```

### 2.2 `src/models/item_towers.py`
```python
class BaseItemTower(keras.Model):
    # Shared: text_embed → Dense(64), item_type_id → Embedding(6, 16)
    # Subclasses เพิ่ม type-specific layers

class AttractionTower(BaseItemTower): ...
class RestaurantTower(BaseItemTower): ...   # schema เหมือน Attraction
class ShopTower(BaseItemTower): ...         # schema เหมือน Attraction
class EventTower(BaseItemTower): ...
class AccommodationTower(BaseItemTower): ...
class ArticleTower(BaseItemTower): ...

def get_item_tower(item_type: str, config) -> BaseItemTower:
    # factory function
```

### 2.3 `src/models/two_tower.py`
```python
class TwoTowerModel(keras.Model):
    # user_tower: UserTower
    # item_towers: dict[str, BaseItemTower]
    #
    # call(user_inputs, item_inputs, item_type) → (user_emb, item_emb)
    # compute_retrieval_loss(user_emb, item_emb, signal_weights) → loss
```

---

## Phase 3: Training

### 3.1 `src/training/losses.py`
```python
def infonce_loss(user_emb, item_emb, temperature=0.07):
    # scores: (batch, batch) dot product matrix
    # labels: diagonal (positive pairs)
    # loss: mean(softmax_cross_entropy over rows)

def mixed_negative_loss(user_emb, item_emb, item_types, temperature):
    # 60% in-batch same type
    # 30% random cross-type
    # 10% hard negative (top-k excluding positives)
```

### 3.2 `src/training/dataset.py`
```python
def build_tf_dataset(interactions_df, item_features, user_features, config):
    # Stratified batch: sample พอๆ กันต่อ item type
    # Prefetch + cache สำหรับ performance
    # Returns tf.data.Dataset

def stratified_batch_sampler(df, batch_size, item_types):
    # sample batch_size // num_types จากแต่ละ type
```

### 3.3 `src/training/trainer.py`
```python
class TwoTowerTrainer:
    def train_step(self, batch):
        # Forward pass ทุก item type ใน batch
        # Compute mixed_negative_loss
        # Backward + optimizer step
        # Track metrics per type

    def train(self, dataset, epochs):
        # Training loop + validation + checkpoint
```

---

## Phase 4: Serving

### 4.1 `src/serving/retrieval.py`
```python
class FAISSRetriever:
    def build_index(self, item_embeddings):
        # สร้าง FAISS index จาก item embeddings ทั้งหมด

    def query(self, user_embedding, top_k=100):
        # ANN search → return top-K item IDs + scores
```

---

## Phase 5: Scripts

### `scripts/embed_items.py`
```
1. Load item data จาก data/raw/
2. สร้าง text field จาก name + description + category
3. Call Gemma embedding API (batch)
4. Truncate ที่ 256 dim (MRL)
5. Save เป็น numpy array + item_id mapping
```

### `scripts/generate_behavior.py`
```
1. Load item data + user profiles
2. Fit VocabRegistry
3. Run BehaviorDataset.generate(num_users, config)
4. Save เป็น data/generated/interactions.parquet
```

### `scripts/train.py`
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

## Data Requirements

ก่อนเริ่ม Phase 1 ต้องมีข้อมูลเหล่านี้ใน `data/raw/`:

```
data/raw/
├── attractions.csv (หรือ .json)
├── restaurants.csv
├── events.csv
├── accommodations.csv
├── shops.csv
├── articles.csv
└── users.csv        ← user profiles (age, home_country, travel_style, etc.)
```

**Columns ที่ต้องมี per file:**

| File | Required columns |
|---|---|
| attractions | id, name, description, category, province, region, days_open, time_slots, avg_hours, is_free, is_24h |
| restaurants | id, name, description, category, province, region, days_open, time_slots, avg_hours, is_24h |
| events | id, name, description, category, province, start_date, end_date |
| accommodations | id, name, description, price_tier, province |
| shops | id, name, description, category, province, days_open, time_slots |
| articles | id, title, content, category, publish_date |
| users | id, age, home_country, travel_style (list), travel_theme (list) |

---

## Open Questions

- [ ] Gemma MRL: ใช้ model string อะไร? API key มาจากที่ไหน?
- [ ] Province/region: item data มี location field ไหม? format อะไร?
- [ ] Accommodation: มี amenities data ไหม?
- [ ] Article: recommendation ขึ้นมาปนกับ places ใน same feed หรือแยก section?
- [ ] MMoE: อยากให้อยู่ใน codebase นี้ด้วย หรือแยก ranking model?
