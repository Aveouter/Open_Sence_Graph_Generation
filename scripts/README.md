# Scripts

Utility and data-preparation scripts for the SGG project.

## Directory Layout

```
scripts/
├── clip/                         # CLIP-based prototype/embedding builders
│   ├── build_clip_prototypes.py
│   └── build_clip_predicate_embeddings.py
├── cluster/                      # Clustering utilities
│   └── GBC.py
├── configs/                      # Auxiliary config files
│   └── ThyroTriples.yaml
└── output/                       # Generated output artifacts
    ├── meta.json
    ├── predicate_paragraphs.json
    ├── test.py
    └── text_rel.npy
```

## Script Descriptions

### CLIP
- `build_clip_prototypes.py` — Build CLIP text prototypes for hierarchical predicate alignment. Uses OpenCLIP.
- `build_clip_predicate_embeddings.py` — Generate CLIP embeddings for predicate descriptions.

### Cluster
- `GBC.py` — Gradient-based clustering for predicate grouping/discovery.

### Configs
- `ThyroTriples.yaml` — Dataset configuration for the ThyroTriples dataset.
