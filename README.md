# SatCLIP Transformers

Hugging Face Transformers integration for [SatCLIP](https://arxiv.org/abs/2311.17179) — global, general-purpose location embeddings from satellite imagery.

**Hub repository:** [`BiliSakura/SatCLIP-transformers`](https://huggingface.co/BiliSakura/SatCLIP-transformers)

## Repository layout

Each pretrained variant is a **self-contained subfolder** with its own `config.json`, `preprocessor_config.json`, `satclip/` custom code, and (after conversion) `model.safetensors`:

```
BiliSakura/SatCLIP-transformers/
├── SatCLIP-ResNet18-L10/
├── SatCLIP-ResNet18-L40/
├── SatCLIP-ResNet50-L10/
├── SatCLIP-ResNet50-L40/
├── SatCLIP-ViT16-L10/
├── SatCLIP-ViT16-L40/
└── src/satclip/              # development package
```

| Variant | Vision encoder | L |
| --- | --- | --- |
| `SatCLIP-ResNet18-L10` | MoCo ResNet18 | 10 |
| `SatCLIP-ResNet18-L40` | MoCo ResNet18 | 40 |
| `SatCLIP-ResNet50-L10` | MoCo ResNet50 | 10 |
| `SatCLIP-ResNet50-L40` | MoCo ResNet50 | 40 |
| `SatCLIP-ViT16-L10` | MoCo ViT-B/16 | 10 |
| `SatCLIP-ViT16-L40` | MoCo ViT-B/16 | 40 |

## Install

```bash
pip install -e ".[vision]"
```

## Inference

Inference requires only `transformers` (plus `timm`/`torchgeo` for MoCo vision backbones). Use the standard **`feature-extraction`** pipeline:

```python
from transformers import pipeline

extractor = pipeline(
    "feature-extraction",
    model="BiliSakura/SatCLIP-transformers",
    subfolder="SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

# Location embeddings (default modality)
emb = extractor({"longitude": 10.5, "latitude": 48.1}, return_tensors=True)
print(emb.shape)  # torch.Size([1, 512])

# Image embeddings
# emb = extractor(image_array, modality="image", return_tensors=True)
```

Direct model API:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "BiliSakura/SatCLIP-transformers",
    subfolder="SatCLIP-ResNet18-L10",
    trust_remote_code=True,
)

location_features = model.get_location_features(coords_tensor)
image_features = model.get_image_features(pixel_values)
```

## Convert Lightning checkpoints

```bash
python scripts/convert_ckpt_to_hf.py \
  path/to/satclip-resnet18-l10.ckpt \
  SatCLIP-ResNet18-L10 \
  --preset SatCLIP-ResNet18-L10
```

## Development

```bash
python scripts/sync_model_code.py          # sync src/satclip into all variant folders
python scripts/generate_model_repo_templates.py  # regenerate variant configs/READMEs
```

See `src/README.md` for package module details.

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

## License

MIT — see [LICENSE](LICENSE).
