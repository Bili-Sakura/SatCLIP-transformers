#!/usr/bin/env python3
"""Convert a PyTorch Lightning SatCLIP checkpoint to Hugging Face format."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from satclip.configuration_satclip import SatCLIPConfig
from satclip.modeling_satclip import SatCLIPModel, load_lightning_state_dict


PRESET_CONFIGS = {
    "SatCLIP-ResNet18-L10": {
        "vision_layers": "moco_resnet18",
        "legendre_polys": 10,
    },
    "SatCLIP-ResNet18-L40": {
        "vision_layers": "moco_resnet18",
        "legendre_polys": 40,
    },
    "SatCLIP-ResNet50-L10": {
        "vision_layers": "moco_resnet50",
        "legendre_polys": 10,
    },
    "SatCLIP-ResNet50-L40": {
        "vision_layers": "moco_resnet50",
        "legendre_polys": 40,
    },
    "SatCLIP-ViT16-L10": {
        "vision_layers": "moco_vit16",
        "legendre_polys": 10,
    },
    "SatCLIP-ViT16-L40": {
        "vision_layers": "moco_vit16",
        "legendre_polys": 40,
    },
}

DEFAULT_MODEL_KWARGS = {
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
    "num_hidden_layers": 2,
    "capacity": 256,
}


def build_config_from_checkpoint(ckpt_path: Path, preset: str | None = None) -> SatCLIPConfig:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = SatCLIPConfig.from_lightning_hyperparameters(ckpt["hyper_parameters"])
    if preset and preset in PRESET_CONFIGS:
        for key, value in PRESET_CONFIGS[preset].items():
            setattr(config, key, value)
    return config


def convert_checkpoint(
    ckpt_path: Path,
    output_dir: Path,
    preset: str | None = None,
    copy_custom_code: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config_from_checkpoint(ckpt_path, preset=preset)
    config.auto_map = {
        "AutoConfig": "satclip.configuration_satclip.SatCLIPConfig",
        "AutoModel": "satclip.modeling_satclip.SatCLIPModel",
        "AutoImageProcessor": "satclip.image_processing_satclip.SatCLIPImageProcessor",
        "AutoProcessor": "satclip.processing_satclip.SatCLIPProcessor",
    }
    config.architectures = ["SatCLIPModel"]

    model = SatCLIPModel(config)
    state_dict = load_lightning_state_dict(str(ckpt_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    config.save_pretrained(output_dir)
    save_file(model.state_dict(), output_dir / "model.safetensors")

    preprocessor_config = {
        "image_processor_type": "SatCLIPImageProcessor",
        "image_size": config.image_resolution,
        "in_channels": config.in_channels,
        "scale_factor": 10000.0,
        "insert_b10_band": True,
        "do_center_crop": False,
        "do_resize": False,
        "auto_map": {
            "AutoImageProcessor": "satclip.image_processing_satclip.SatCLIPImageProcessor",
            "AutoProcessor": "satclip.processing_satclip.SatCLIPProcessor",
        },
    }
    with open(output_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor_config, f, indent=2)

    if copy_custom_code:
        src_pkg = Path(__file__).resolve().parents[1] / "src" / "satclip"
        dst_pkg = output_dir / "satclip"
        if dst_pkg.exists():
            import shutil

            shutil.rmtree(dst_pkg)
        import shutil

        shutil.copytree(src_pkg, dst_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    print(f"Converted checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to Lightning .ckpt file")
    parser.add_argument("output_dir", type=Path, help="Output Hugging Face model directory")
    parser.add_argument(
        "--preset",
        choices=list(PRESET_CONFIGS.keys()),
        help="Optional preset name to override checkpoint hyperparameters",
    )
    parser.add_argument("--no-copy-code", action="store_true", help="Do not copy custom code into output dir")
    args = parser.parse_args()

    convert_checkpoint(
        ckpt_path=args.checkpoint,
        output_dir=args.output_dir,
        preset=args.preset,
        copy_custom_code=not args.no_copy_code,
    )


if __name__ == "__main__":
    main()
