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
---

# SatCLIP ResNet18 L10

SatCLIP is a contrastive model that learns joint embeddings of Sentinel-2 satellite images and geographic coordinates. This checkpoint uses a MoCo-pretrained ResNet18 vision encoder and spherical harmonics location encoding with **L=10** Legendre polynomials.

## Model Details

| Property | Value |
| --- | --- |
| Vision encoder | MoCo ResNet18 (Sentinel-2) |
| Location encoding | Spherical harmonics (L=10) |
| Neural network | SIREN |
| Embedding dimension | 512 |
| Input image size | 256×256, 13 bands |

## Installation

```bash
pip install transformers torch timm torchgeo einops
```

For local development of this repository:

```bash
pip install -e .
```

## Usage

### Location embeddings (most common downstream use case)

```python
from transformers import AutoModel, pipeline

model = AutoModel.from_pretrained(
    "microsoft/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

pipe = pipeline(
    "satclip-location-encoding",
    model=model,
    trust_remote_code=True,
)

embeddings = pipe(longitude=10.5, latitude=48.1)
print(embeddings["embeddings"].shape)  # (1, 512)
```

### Direct model API

```python
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "microsoft/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "microsoft/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

coords = torch.tensor([[10.5, 48.1]], dtype=torch.float32)
location_features = model.encode_location(coords)
```

### Image localization

```python
from transformers import AutoModel, AutoProcessor, pipeline

model = AutoModel.from_pretrained(
    "microsoft/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "microsoft/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

pipe = pipeline(
    "satclip-image-localization",
    model=model,
    processor=processor,
    trust_remote_code=True,
)

# image: numpy array with shape (12 or 13, H, W) reflectance values
# candidate_coords: list of [lon, lat] pairs
result = pipe(image=image, coords=candidate_coords)
similarity = result["similarity"]
```

## Repository layout

This model repository follows the Hugging Face custom-code layout:

```
SatCLIP-ResNet18-L10/
├── README.md
├── config.json
├── preprocessor_config.json
├── model.safetensors          # produced by scripts/convert_ckpt_to_hf.py
└── satclip/                   # custom Transformers-compatible code
    ├── configuration_satclip.py
    ├── modeling_satclip.py
    ├── image_processing_satclip.py
    └── ...
```

The `satclip/` subfolder lets you run inference with only `transformers` installed — no need for the original training repository.

## Converting legacy Lightning checkpoints

```bash
python scripts/convert_ckpt_to_hf.py \
  path/to/satclip-resnet18-l10.ckpt \
  model_repo/SatCLIP-ResNet18-L10 \
  --preset SatCLIP-ResNet18-L10
```

## Citation

```bibtex
@article{klemmer2025satclip,
    title={SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery},
    volume={39},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/32457},
    number={4},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Klemmer, Konstantin and Rolf, Esther and Robinson, Caleb and Mackey, Lester and Rußwurm, Marc},
    year={2025}
}
```
