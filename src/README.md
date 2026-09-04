# SatCLIP Transformers

Hugging Face Transformers integration for [SatCLIP](https://arxiv.org/abs/2311.17179) — global, general-purpose location embeddings from satellite imagery.

**Hub repository:** [`BiliSakura/SatCLIP-transformers`](https://huggingface.co/BiliSakura/SatCLIP-transformers)

Each pretrained variant lives in a self-contained subfolder:

```
BiliSakura/SatCLIP-transformers/
├── SatCLIP-ResNet18-L10/
├── SatCLIP-ResNet18-L40/
├── SatCLIP-ResNet50-L10/
├── SatCLIP-ResNet50-L40/
├── SatCLIP-ViT16-L10/
├── SatCLIP-ViT16-L40/
└── src/satclip/          # development package (also vendored per variant)
```

## Install

```bash
pip install -e ".[vision]"
```

## Inference

```python
from transformers import AutoModel, pipeline

model = AutoModel.from_pretrained(
    "BiliSakura/SatCLIP-transformers",
    subfolder="SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

extractor = pipeline("feature-extraction", model=model, trust_remote_code=True)

# Location embeddings (default)
emb = extractor({"longitude": 10.5, "latitude": 48.1}, return_tensors=True)
print(emb.shape)  # torch.Size([1, 512])

# Image embeddings
# emb = extractor(image_array, modality="image", return_tensors=True)
```

## Convert Lightning checkpoints

```bash
python scripts/convert_ckpt_to_hf.py \
  path/to/satclip-resnet18-l10.ckpt \
  SatCLIP-ResNet18-L10 \
  --preset SatCLIP-ResNet18-L10
```

## Development

| Path | Description |
| --- | --- |
| `src/satclip/` | Transformers-compatible package (`SatCLIPConfig`, `SatCLIPModel`, processors, pipeline) |
| `SatCLIP-*/` | Self-contained Hub subfolders with `config.json`, `satclip/` code, and (after conversion) `model.safetensors` |
| `scripts/` | Checkpoint conversion, template generation, code sync |

Sync `src/satclip` into all variant folders:

```bash
python scripts/sync_model_code.py
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
