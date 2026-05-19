# Experiments

Architecture details: [architecture.md](architecture.md)

| Exp | Date | Description | Overall Hit@10 | Overall Hit@100 |
|---|---|---|---|---|
| [Exp-001](experiments/exp-001.md) | 2026-05-18 | Baseline | 0.073 | 0.250 |
| [Exp-002](experiments/exp-002.md) | 2026-05-18 | Scale + LR schedule + harder negatives | 0.071 | 0.281 |
| [Exp-003](experiments/exp-003.md) | 2026-05-19 | Catalog pruning + real text embeddings | 0.080 | 0.334 |
| [Exp-004](experiments/exp-004.md) | 2026-05-19 | Article feature additions (`is_thai`, `pub_recency_norm`) | 0.080 | 0.334 |
| [Exp-005](experiments/exp-005.md) | 2026-05-19 | Article behavior: theme + language bias in generator | 0.082 | 0.439 |
| [Exp-006](experiments/exp-006.md) | 2026-05-19 | Scale dataset to 5k users (stable baseline, 31.7k val) | 0.072 | 0.346 |
