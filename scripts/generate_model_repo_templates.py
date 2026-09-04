#!/usr/bin/env python3
"""Generate Hugging Face model repo templates for all SatCLIP variants."""

import json
from pathlib import Path

BASE_CONFIG = {
    "architectures": ["SatCLIPModel"],
    "auto_map": {
        "AutoConfig": "satclip.configuration_satclip.SatCLIPConfig",
        "AutoModel": "satclip.modeling_satclip.SatCLIPModel",
        "AutoImageProcessor": "satclip.image_processing_satclip.SatCLIPImageProcessor",
        "AutoProcessor": "satclip.processing_satclip.SatCLIPProcessor",
    },
    "model_type": "satclip",
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


def main():
    repo_root = Path(__file__).resolve().parents[1] / "model_repo"
    src_pkg = Path(__file__).resolve().parents[1] / "src" / "satclip"

    for name, overrides in VARIANTS.items():
        model_dir = repo_root / name
        model_dir.mkdir(parents=True, exist_ok=True)

        config = {**BASE_CONFIG, **overrides}
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        with open(model_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
            json.dump(PREPROCESSOR_CONFIG, f, indent=2)
            f.write("\n")

        dst_pkg = model_dir / "satclip"
        if not dst_pkg.exists():
            import shutil

            shutil.copytree(
                src_pkg,
                dst_pkg,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        readme = model_dir / "README.md"
        if not readme.exists():
            backbone = overrides["vision_layers"].replace("moco_", "").upper()
            readme.write_text(
                f"# {name}\n\n"
                f"SatCLIP checkpoint with {backbone} vision encoder and L={overrides['legendre_polys']} "
                f"spherical harmonics location encoding.\n\n"
                f"See [SatCLIP-ResNet18-L10](../SatCLIP-ResNet18-L10/README.md) for usage instructions.\n",
                encoding="utf-8",
            )

        print(f"Generated template: {model_dir}")


if __name__ == "__main__":
    main()
