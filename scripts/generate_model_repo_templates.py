#!/usr/bin/env python3
"""Generate self-contained Hugging Face variant subfolders for BiliSakura/SatCLIP-transformers."""

import json
from pathlib import Path

REPO_ID = "BiliSakura/SatCLIP-transformers"

BASE_CONFIG = {
    "architectures": ["SatCLIPModel"],
    "auto_map": {
        "AutoConfig": "satclip.configuration_satclip.SatCLIPConfig",
        "AutoModel": "satclip.modeling_satclip.SatCLIPModel",
        "AutoImageProcessor": "satclip.image_processing_satclip.SatCLIPImageProcessor",
        "AutoProcessor": "satclip.processing_satclip.SatCLIPProcessor",
    },
    "custom_pipelines": {
        "feature-extraction": {
            "impl": "satclip.pipeline_satclip.SatCLIPFeatureExtractionPipeline",
            "pt": ["AutoModel"],
        }
    },
    "model_type": "satclip",
    "pipeline_tag": "feature-extraction",
    "torch_dtype": "float32",
    "transformers_version": "4.40.0",
    "embed_dim": 512,
    "image_resolution": 256,
    "vision_width": 768,
    "vision_patch_size": 32,
    "in_channels": 13,
    "le_type": "sphericalharmonics",
    "pe_type": "siren",
    "frequency_num": 16,
    "max_radius": 360.0,
    "min_radius": 1.0,
    "harmonics_calculation": "analytic",
    "sh_embedding_dims": 16,
    "num_hidden_layers": 2,
    "capacity": 256,
    "return_location_encoder_only": False,
}

PREPROCESSOR_CONFIG = {
    "image_processor_type": "SatCLIPImageProcessor",
    "image_size": 256,
    "in_channels": 13,
    "scale_factor": 10000.0,
    "insert_b10_band": True,
    "do_center_crop": False,
    "do_resize": False,
    "auto_map": {
        "AutoImageProcessor": "satclip.image_processing_satclip.SatCLIPImageProcessor",
        "AutoProcessor": "satclip.processing_satclip.SatCLIPProcessor",
    },
}

VARIANTS = {
    "SatCLIP-ResNet18-L10": {"vision_layers": "moco_resnet18", "legendre_polys": 10},
    "SatCLIP-ResNet18-L40": {"vision_layers": "moco_resnet18", "legendre_polys": 40},
    "SatCLIP-ResNet50-L10": {"vision_layers": "moco_resnet50", "legendre_polys": 10},
    "SatCLIP-ResNet50-L40": {"vision_layers": "moco_resnet50", "legendre_polys": 40},
    "SatCLIP-ViT16-L10": {"vision_layers": "moco_vit16", "legendre_polys": 10},
    "SatCLIP-ViT16-L40": {"vision_layers": "moco_vit16", "legendre_polys": 40},
}

README_TEMPLATE = """---
library_name: transformers
license: mit
tags:
  - satclip
  - geospatial
  - satellite-imagery
  - clip
  - location-encoder
pipeline_tag: feature-extraction
base_model: {repo_id}
---

# {name}

SatCLIP checkpoint with **{backbone}** vision encoder and **L={legendre_polys}** spherical harmonics location encoding.

Part of [{repo_id}](https://huggingface.co/{repo_id}) — load via the `subfolder` argument.

## Usage

```python
from transformers import AutoModel, pipeline

model = AutoModel.from_pretrained(
    "{repo_id}",
    subfolder="{name}",
    trust_remote_code=True,
)

extractor = pipeline(
    "feature-extraction",
    model=model,
    trust_remote_code=True,
)

# Location embeddings (default modality)
emb = extractor({{"longitude": 10.5, "latitude": 48.1}}, return_tensors=True)
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
"""


def main():
    repo_root = Path(__file__).resolve().parents[1]
    src_pkg = repo_root / "src" / "satclip"

    for name, overrides in VARIANTS.items():
        variant_dir = repo_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)

        config = {**BASE_CONFIG, **overrides}
        with open(variant_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        with open(variant_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
            json.dump(PREPROCESSOR_CONFIG, f, indent=2)
            f.write("\n")

        backbone = overrides["vision_layers"].replace("moco_", "").upper()
        readme = README_TEMPLATE.format(
            repo_id=REPO_ID,
            name=name,
            backbone=backbone,
            legendre_polys=overrides["legendre_polys"],
        )
        with open(variant_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme)

        dst_pkg = variant_dir / "satclip"
        if dst_pkg.exists():
            import shutil

            shutil.rmtree(dst_pkg)
        import shutil

        shutil.copytree(
            src_pkg,
            dst_pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        print(f"Generated {variant_dir}")


if __name__ == "__main__":
    main()
