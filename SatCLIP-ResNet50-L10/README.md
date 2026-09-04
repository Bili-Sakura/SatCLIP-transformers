---
library_name: transformers
license: mit
tags:
  - satclip
  - geospatial
  - satellite-imagery
  - clip
  - location-encoder
pipeline_tag: feature-extraction
base_model: BiliSakura/SatCLIP-transformers
---

# SatCLIP-ResNet50-L10

SatCLIP checkpoint with **RESNET50** vision encoder and **L=10** spherical harmonics location encoding.

Part of [BiliSakura/SatCLIP-transformers](https://huggingface.co/BiliSakura/SatCLIP-transformers) — load via the `subfolder` argument.

## Usage

```python
from transformers import AutoModel, pipeline

model = AutoModel.from_pretrained(
    "BiliSakura/SatCLIP-transformers",
    subfolder="SatCLIP-ResNet50-L10",
    trust_remote_code=True,
)

extractor = pipeline(
    "feature-extraction",
    model=model,
    trust_remote_code=True,
)

# Location embeddings (default modality)
emb = extractor({"longitude": 10.5, "latitude": 48.1}, return_tensors=True)
print(emb.shape)  # torch.Size([1, 512])

# Image embeddings
# emb = extractor(image, modality="image", return_tensors=True)
```

## Files

| File | Description |
| --- | --- |
| `config.json` | Model hyperparameters and `auto_map` |
| `preprocessor_config.json` | Sentinel-2 image preprocessing |
| `model.safetensors` | Model weights (after checkpoint conversion) |
| `satclip/` | Custom Transformers-compatible code |
