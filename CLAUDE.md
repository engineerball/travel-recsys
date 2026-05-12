# Travel Recommendation System — Claude Instructions

## Project Overview

Two-Tower + MMoE recommendation system สำหรับแนะนำสถานที่ท่องเที่ยวไทย  
ใช้ TensorFlow/Keras เป็น framework หลัก

---

## Repository Setup

- ตรวจสอบว่า directory เป็น git repository ก่อนเสมอ
- ใช้ `uv` สำหรับ Python environment management
- Commit บ่อยๆ พร้อม message ที่ชัดเจน
- ใช้ `author="AI <ai@engineerball.com>"` สำหรับทุก commit

---

## Coding Style

- ทำตาม PEP8
- เขียน docstring ทุก function/class
- เพิ่ม comment อธิบายทุก block สำคัญ
- ฟังก์ชันเล็กๆ อ่านง่ายกว่า function ใหญ่ที่ซับซ้อน
- ไม่ต้องเขียน unit test ยกเว้นจะบอก

---

## Architecture Overview

```
User Tower ──────────────────────────────────────────────────────┐
  age, country, travel_style, travel_theme,                       │
  category_history, subcat_affinity, context (day/hour)           ├─→ dot product → score
                                                                   │
Item Tower (แยกต่อประเภท, shared embedding space 64-dim)  ────────┘
  Attraction | Restaurant | Events | Accommodation | Shop | Article
         ↓
   Unified ANN Index (FAISS)
         ↓
   MMoE Ranking (2 tasks: engagement vs satisfaction)
```

### Key Design Decisions

1. **แยก item tower ต่อประเภท** — feature schema ต่างกัน 6 แบบ ไม่ควรรวม tower เดียว
2. **Shared embedding space 64-dim** — ทุก tower output เป็น dim เดียวกัน ทำให้ rank รวมกันได้
3. **Train jointly** — loss รวมจากทุก item type พร้อมกัน เพื่อให้ user embedding works กับทุก type
4. **Mixed negatives** — 60% in-batch same type, 30% cross-type, 10% hard negative
5. **MMoE ที่ Ranking stage** — หลัง retrieval, 2 tasks: engagement (view) และ satisfaction (like/bookmark)

---

## Feature Schema

### User Tower

| Feature | Type | Preprocessing |
|---|---|---|
| `age_norm` | float | Z-score (ไม่ใช้ MinMaxScaler เพราะ outlier) |
| `home_country_id` | int | → Embedding(num_countries, 16) |
| `travel_style_multihot` | multi-hot | active indices → Embedding → MeanPool |
| `travel_theme_multihot` | multi-hot | active indices → Embedding → MeanPool |
| `category_preference_multihot` | multi-hot | active indices → Embedding → MeanPool |
| `category_interaction_history` | float vector | Dense(32) |
| `subcat_affinity` | float vector | Dense(32) |
| `context_day_sin/cos` | float | cyclical encoding ของ day_of_week |
| `context_hour_sin/cos` | float | cyclical encoding ของ hour_of_day |

### Item Towers

**ทุก item type มี:**
- `text_embed` — Gemma embedding ด้วย MRL (truncate ที่ 256 dim) → trainable Dense(64)
- `item_type_id` — Embedding(6, 16) สำคัญมากสำหรับ single space

**Attraction / Restaurant / Shop (ใช้ schema เดียวกัน)**

| Feature | Preprocessing |
|---|---|
| `category_multihot` | Embedding → MeanPool |
| `days_open_vector` | 7-dim binary, concat โดยตรง |
| `time_slots_multihot` | Embedding → MeanPool |
| `avg_hours_norm` | Z-score |
| `is_24_hours` | binary |
| `is_free` | binary |
| `province_id` | → Embedding(77, 16) |
| `region_id` | → Embedding(8, 8) |
| `log_view_count` | log1p(view_count) |

**Events**

| Feature | Preprocessing |
|---|---|
| `category_multihot` | Embedding → MeanPool |
| `duration_days_norm` | Z-score |
| `month_sin / month_cos` | cyclical (มีอยู่แล้ว) |
| `province_id` | → Embedding(77, 16) |

**Accommodation**

| Feature | Preprocessing |
|---|---|
| `price_tier_id` | → Embedding(5, 8) |
| `is_price_missing` | binary |
| `star_rating_norm` | Z-score |
| `amenities_multihot` | Embedding → MeanPool |
| `province_id` | → Embedding(77, 16) |

**Article**

| Feature | Preprocessing |
|---|---|
| `category_multihot` | Embedding → MeanPool |
| `pub_month_sin/cos` | cyclical |

> **หมายเหตุ:** `weekend_opened` ตัดออกทุก type เพราะ redundant กับ `days_open_vector`

---

## Interaction Signals & MMoE Tasks

```
Signal      Weight    MMoE Task
─────────────────────────────────────
view         1.0      Task A: Engagement
like         3.0      Task B: Satisfaction
bookmark     4.0      Task B: Satisfaction
```

**2 MMoE Tasks:**
- **Task A (Engagement):** predict view → ป้องกัน popularity bias
- **Task B (Satisfaction):** predict like/bookmark → push quality content

---

## Text Embedding

- ใช้ **Gemma Embedding** (หรือ `text-embedding-004`) พร้อม MRL
- Truncate ที่ **256 dimensions** (ตรวจสอบว่า model รองรับ MRL ก่อน)
- ต่อด้วย **trainable Dense(64)** เพื่อ fine-tune บน task นี้
- Script สำหรับ pre-compute embeddings: `scripts/embed_items.py`

---

## Behavior Generation (Synthetic)

เนื่องจากยังไม่มี real user interaction data ใช้ **persona-based simulation**:

```
5 Personas: backpacker, family, culture_seeker, foodie, luxury
  ↓
Power law item popularity (exponent 0.7)
  ↓
Geographic filter (ตาม travel_radius_km ของ persona)
  ↓
Temporal filter (opening hours, day_of_week)
  ↓
Signal generation: view / like / bookmark ตาม preference score
```

Config อยู่ที่ `configs/config.yaml` → `behavior` section

---

## Project Structure

```
travel-recsys/
├── configs/
│   └── config.yaml          ← hyperparameters, vocab sizes, behavior config
├── data/
│   ├── raw/                 ← item data จริง (ไม่ commit)
│   ├── processed/           ← หลัง preprocessing (ไม่ commit)
│   └── generated/           ← synthetic interactions (ไม่ commit)
├── src/
│   ├── data/
│   │   ├── schema.py        ← feature schema definitions ทุก item type
│   │   ├── preprocessing.py ← normalizers, encoders, embedding pooling
│   │   └── behavior_generator.py  ← persona-based interaction simulation
│   ├── models/
│   │   ├── user_tower.py    ← UserTower Keras model
│   │   ├── item_towers.py   ← 6 item towers + BaseItemTower
│   │   └── two_tower.py     ← TwoTowerModel combining everything
│   ├── training/
│   │   ├── dataset.py       ← tf.data pipeline, batch builder
│   │   ├── losses.py        ← InfoNCE loss, mixed negative sampling
│   │   └── trainer.py       ← training loop, joint loss
│   └── serving/
│       └── retrieval.py     ← FAISS index, query pipeline
├── scripts/
│   ├── embed_items.py       ← pre-compute Gemma embeddings
│   ├── generate_behavior.py ← run behavior generator
│   └── train.py             ← entry point สำหรับ training
├── notebooks/               ← exploration, visualization
├── pyproject.toml
└── CLAUDE.md                ← this file
```

---

## Development Notes

### Dependencies

```bash
uv add tensorflow tensorflow-recommenders numpy pandas scikit-learn pyyaml tqdm google-generativeai faiss-cpu
```

### สิ่งที่ยังต้องทำ (ตามลำดับ priority)

1. **`src/data/schema.py`** — define feature schemas, vocab mappings
2. **`src/data/preprocessing.py`** — normalizers, multi-hot → index conversion
3. **`src/data/behavior_generator.py`** — persona simulation
4. **`src/models/user_tower.py`** — UserTower Keras model
5. **`src/models/item_towers.py`** — BaseItemTower + 6 item towers
6. **`src/models/two_tower.py`** — combined model + joint training
7. **`src/training/losses.py`** — InfoNCE + mixed negatives
8. **`src/training/dataset.py`** — tf.data pipeline
9. **`src/training/trainer.py`** — training loop
10. **`scripts/train.py`** — entry point

### Known Issues / Decisions Pending

- [ ] ตรวจสอบว่า Gemma embedding model ที่ใช้ support MRL จริงไหม
- [ ] กำหนด `num_categories`, `num_subcategories` จาก real data ก่อน fit vocab
- [ ] Location features (province_id, region_id) ต้องมีใน item data จริง — ถ้าไม่มีต้อง geocode เพิ่ม
- [ ] Article: พิจารณาว่าควรอยู่ใน same embedding space หรือแยก tower
- [ ] Accommodation: ต้องการ amenities_multihot และ star_rating ซึ่งอาจไม่มีใน raw data

### Negative Sampling Strategy

```python
# ใน training loop
# 60% in-batch same type — items อื่นใน batch เดียวกันที่เป็น type เดียวกัน
# 30% random cross-type — sample จาก type อื่น
# 10% hard negative — items ที่ score สูงแต่ไม่มี interaction
```

### Embedding Space Alignment

ทุก item tower ต้อง output `(batch, 64)` และ train ร่วมกับ user tower ใน loss เดียวกัน  
ห้าม train แยก tower ทีละตัว เพราะ user embedding จะถูก pull ไปคนละทิศ

---

## Context จาก Conversation

ระบบนี้ถูก design ตาม discussion กับ Claude ใน Cowork session โดยมี decisions ดังนี้:

- เริ่มจาก single tower แล้วเปลี่ยนเป็น **separate towers per item type** เพราะ feature schema ต่างกันมาก
- **Behavior generation** ปัจจุบันเป็น mock → กำลัง improve ให้สมจริงด้วย persona + power law + geographic constraints
- **text_embed_pca** เดิมเปลี่ยนเป็น **Gemma MRL** เพราะ PCA ไม่ task-aware
- **MMoE** อยู่ที่ **ranking stage** หลัง retrieval ไม่ใช่ใน retrieval tower
- **age_norm** ควรเปลี่ยนจาก MinMaxScaler → Z-score หรือ age bucketing
- **weekend_opened** ตัดออก (redundant กับ days_open_vector)
