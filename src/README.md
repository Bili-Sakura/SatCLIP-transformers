# SatCLIP Transformers Integration

The `src/satclip/` package provides a Hugging Face Transformers–compatible implementation of SatCLIP.

## Install

```bash
pip install -e ".[vision,dev]"
```

## Quick start

```python
import satclip  # registers Auto* classes
from transformers import AutoModel, AutoProcessor, pipeline

model = AutoModel.from_pretrained(
    "model_repo/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "model_repo/SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

# Location encoding
pipe = pipeline("satclip-location-encoding", model=model, trust_remote_code=True)
emb = pipe(longitude=10.5, latitude=48.1)
```

## Package structure

| Module | Description |
| --- | --- |
| `configuration_satclip.py` | `SatCLIPConfig` (`PretrainedConfig`) |
| `modeling_satclip.py` | `SatCLIPModel`, `SatCLIPLocationEncoderModel` |
| `modeling_vision.py` | Vision backbones (ResNet, ViT, MoCo) |
| `modeling_location_encoder.py` | Location encoder (SIREN, MLP, spherical harmonics) |
| `image_processing_satclip.py` | 13-band Sentinel-2 preprocessing |
| `processing_satclip.py` | Combined image + coordinate processor |
| `pipeline_satclip.py` | `satclip-location-encoding`, `satclip-image-encoding`, `satclip-image-localization` |

## Converting Lightning checkpoints

```bash
python scripts/convert_ckpt_to_hf.py checkpoint.ckpt model_repo/SatCLIP-ResNet18-L10 --preset SatCLIP-ResNet18-L10
```

## Model repo layout

See `model_repo/README.md` for Hugging Face Hub publishing instructions.
