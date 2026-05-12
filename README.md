# Travel Recommendation System

Two-tower retrieval model สำหรับแนะนำสถานที่ท่องเที่ยวไทย  
Built with TensorFlow/Keras + FAISS.

## Architecture

```
User Tower ──────────────────────────────────────────────────┐
  age, country, travel_style, travel_theme,                   │
  category_history, subcat_affinity, context (day/hour)       ├─→ cosine similarity → score
                                                              │
Item Towers (one per type, shared 64-dim embedding space) ───┘
  Attraction | Accommodation | Event | Article
        ↓
  FAISS ANN Index (IndexFlatIP, 64-dim)
        ↓
  Top-K retrieval
```

Trained jointly across all 4 item types with InfoNCE + mixed cross-type negatives.

## Project Structure

```
travel-recsys/
├── configs/
│   └── config.yaml              # hyperparameters, vocab sizes, behavior config
├── data/
│   ├── raw/                     # source parquets (destinations, accommodations,
│   │   │                        #   activities, articles, user_profiles)
│   ├── processed/               # pre-computed text embeddings (embed_items.py)
│   └── generated/               # synthetic interactions (generate_behavior.py)
├── scripts/
│   ├── embed_items.py           # pre-compute Gemma MRL embeddings (256-dim)
│   ├── generate_behavior.py     # persona-based interaction simulation
│   └── train.py                 # end-to-end training entry point
├── src/
│   ├── data/
│   │   ├── schema.py            # feature schemas, vocab registries
│   │   ├── preprocessing.py     # normalizers, encoders, per-type feature extractors
│   │   └── behavior_generator.py# InteractionGenerator, compute_behavior_features
│   ├── models/
│   │   ├── user_tower.py        # UserTower (profile + behavior features)
│   │   ├── item_towers.py       # 4 ItemTowers + BaseItemTower
│   │   └── two_tower.py         # TwoTowerModel, joint loss
│   ├── training/
│   │   ├── losses.py            # infonce_loss, mixed_negative_loss
│   │   ├── dataset.py           # StratifiedInteractionDataset, tf.data pipeline
│   │   └── trainer.py           # TwoTowerTrainer, checkpointing
│   └── serving/
│       └── retrieval.py         # FAISSRetriever, MultiTypeIndex, RetrievalPipeline
└── plans/
    └── implementation-plan.md   # phase-by-phase decisions + deviations
```

## Data

| Source dir       | Item type      | Rows   |
|------------------|----------------|--------|
| `destinations/`  | Attraction     | 8,632  |
| `activities/`    | Event          | 21,376 |
| `accommodations/`| Accommodation  | 2,972  |
| `articles/`      | Article        | 1,559  |
| `user_profiles/` | Users          | 1,000  |

No real interaction data — training uses **persona-based synthetic interactions** (5 personas: backpacker, family, culture_seeker, foodie, luxury).

## Setup

```bash
uv sync
```

## Run

### Step 1 — Pre-compute text embeddings

```bash
HF_TOKEN=hf_... uv run python scripts/embed_items.py
```

Encodes `text_for_embed` fields using `google/embeddinggemma-300m` (MRL, truncated to 256-dim).  
Saves `data/processed/{type}_text_embeds.npy` + `{type}_item_ids.npy`.

### Step 2 — Generate synthetic interactions

```bash
uv run python scripts/generate_behavior.py
```

Runs persona simulation over all users → saves `data/generated/interactions.parquet` and `data/generated/user_features.parquet`.

### Step 3 — Train

```bash
uv run python scripts/train.py
```

Trains TwoTowerModel, saves towers to `models/towers/`, builds FAISS index to `models/index/`.

```bash
# Override config values
uv run python scripts/train.py --epochs 30 --batch-size 256 --steps-per-epoch 300
```

## Feature Schema

### User Tower

| Feature | Shape | Description |
|---------|-------|-------------|
| `age_norm` | `[B]` | Z-score |
| `home_country_id` | `[B]` | Embedding(7, 8) |
| `travel_style_indices` | `[B, 5]` | Embedding(5, 8) + MeanPool |
| `travel_theme_indices` | `[B, 15]` | Embedding(15, 8) + MeanPool |
| `context_day_sin/cos` | `[B]` | cyclical day-of-week |
| `context_hour_sin/cos` | `[B]` | cyclical hour-of-day |
| `category_pref_indices` | `[B, 10]` | top-k interaction categories |
| `category_interaction_history` | `[B, 69]` | weighted interaction rate per category |
| `subcat_affinity` | `[B, 58]` | weighted affinity per attraction sub-category |

### Item Towers

All towers output `[B, 64]` L2-normalized into the shared embedding space.

| Feature | Attraction | Accommodation | Event | Article |
|---------|:---:|:---:|:---:|:---:|
| `text_embed` (256→64 Dense) | ✓ | ✓ | ✓ | ✓ |
| `item_type_id` Emb(4,8) | ✓ | ✓ | ✓ | ✓ |
| `province_id` Emb(77,16) | ✓ | ✓ | ✓ | |
| `sub_category_indices` Emb(58,16)+Pool | ✓ | | | |
| `days_open_vector` [7] | ✓ | | | |
| `is_free`, `log_view_count` | ✓ | ✓ | | |
| `amenity_indices` Emb(24,8)+Pool | | ✓ | | |
| `price_tier_id` Emb(6,8) | | ✓ | | |
| `star_rating_norm` | | ✓ | | |
| `category_indices` Emb(11,8)+Pool | | | ✓ | |
| `duration_days_norm` | | | ✓ | |
| `month_sin/cos` | | | ✓ | |
| `article_type_id` Emb(12,8) | | | | ✓ |
| `pub_month_sin/cos` | | | | ✓ |

## Training Details

- **Loss:** InfoNCE over in-batch negatives + cross-type hard negative boost
- **Temperature:** 0.07
- **Optimizer:** Adam, lr=0.001
- **Batch:** stratified (equal samples per item type per batch)
- **Checkpoints:** saved every epoch to `checkpoints/epoch_{N:03d}/`

## Serving

```python
from src.serving.retrieval import MultiTypeIndex, RetrievalPipeline

# Load saved index
pipeline = RetrievalPipeline(model, multi_index)
pipeline.index.load("models/index")

# Retrieve top-100 items for a user
results = pipeline.retrieve(user_inputs, top_k_total=100)
# results: List[RetrievalResult(item_id, item_type, score)]
```
