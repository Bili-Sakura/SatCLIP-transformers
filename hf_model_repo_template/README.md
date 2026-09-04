---
language: en
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - satclip
  - geospatial
  - location-embeddings
---

# SatCLIP Transformers Template

This folder is a Hugging Face model-repo-style template for loading SatCLIP with native `transformers` and `trust_remote_code=True`.

## Layout

- `config.json`: model config + `auto_map`
- `custom_code/`: custom `configuration`, `modeling`, and `pipeline`

## Example usage

```python
from transformers import AutoModel, AutoConfig, pipeline

repo_id = "<your-namespace>/<your-satclip-model>"
config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

pipe = pipeline(
    task="feature-extraction",
    model=model,
    tokenizer=None,
    trust_remote_code=True,
)

emb = pipe([[12.5, 48.1], [-73.99, 40.73]])
```

## Converting original checkpoint

Use `SatCLIPModel.from_satclip_checkpoint(...)` from the same custom code to load an original `.ckpt`, then call `save_pretrained()` and upload the saved weights and this custom code.
