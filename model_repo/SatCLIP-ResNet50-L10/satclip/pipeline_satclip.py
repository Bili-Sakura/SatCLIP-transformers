"""SatCLIP inference pipelines."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Pipeline, PreTrainedModel
from transformers.utils import is_torch_available

from .modeling_satclip import SatCLIPLocationEncoderModel, SatCLIPModel


class SatCLIPPipeline(Pipeline):
    """
    Pipeline for SatCLIP inference tasks.

    Supported tasks:
    - ``satclip-location-encoding``: encode (lon, lat) coordinates
    - ``satclip-image-encoding``: encode satellite images
    - ``satclip-image-localization``: rank locations for an image
    """

    def __call__(self, inputs=None, *args, **kwargs):
        if inputs is None:
            inputs = {
                key: kwargs.pop(key)
                for key in ("image", "images", "coords", "longitude", "latitude")
                if key in kwargs
            }
            if not inputs and not kwargs:
                raise ValueError(
                    "SatCLIPPipeline requires coordinates and/or images. "
                    "Pass `longitude`/`latitude`, `coords`, or `image`."
                )
        return super().__call__(inputs, *args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {
            "return_tensors": "pt",
        }
        forward_params = {}
        postprocess_params = {
            "return_tensors": kwargs.pop("return_tensors", False),
        }
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            image = inputs.get("image", inputs.get("images"))
            coords = inputs.get("coords")
            longitude = inputs.get("longitude")
            latitude = inputs.get("latitude")
        else:
            image = inputs
            coords = longitude = latitude = None

        if self.task == "satclip-location-encoding":
            if coords is None and longitude is not None and latitude is not None:
                if isinstance(longitude, (int, float)):
                    coords = [[longitude, latitude]]
                else:
                    coords = list(zip(longitude, latitude))
            return {"coords": torch.tensor(coords, dtype=torch.float32, device=self.device)}

        if self.processor is not None:
            inputs = self.processor(
                images=image,
                coords=coords,
                longitude=longitude,
                latitude=latitude,
                return_tensors="pt",
            )
            return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        batch = {}
        if image is not None:
            batch["pixel_values"] = image
        if coords is not None or (longitude is not None and latitude is not None):
            if coords is None:
                if isinstance(longitude, (int, float)):
                    coords = [[longitude, latitude]]
                else:
                    coords = list(zip(longitude, latitude))
            batch["coords"] = torch.tensor(coords, dtype=torch.float32, device=self.device)
        return batch

    def _forward(self, model_inputs: Dict[str, Any], **kwargs):
        model = self.model
        if self.task == "satclip-location-encoding":
            if isinstance(model, SatCLIPLocationEncoderModel):
                embeddings = model.encode_location(model_inputs["coords"])
            else:
                embeddings = model.encode_location(model_inputs["coords"])
            return {"embeddings": embeddings}

        if self.task == "satclip-image-encoding":
            embeddings = model.get_image_features(model_inputs["pixel_values"])
            return {"embeddings": embeddings}

        if self.task == "satclip-image-localization":
            image_embeds = model.get_image_features(model_inputs["pixel_values"])
            location_embeds = model.get_location_features(model_inputs["coords"])
            similarity = image_embeds @ location_embeds.t()
            return {"similarity": similarity, "image_embeds": image_embeds, "location_embeds": location_embeds}

        return model(**model_inputs)

    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs
        outputs = {}
        for key, value in model_outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy()
            else:
                outputs[key] = value
        return outputs


SATCLIP_TASKS = {
    "satclip-location-encoding": {
        "impl": SatCLIPPipeline,
        "default": {"model": (SatCLIPModel, SatCLIPLocationEncoderModel)},
    },
    "satclip-image-encoding": {
        "impl": SatCLIPPipeline,
        "default": {"model": (SatCLIPModel,)},
    },
    "satclip-image-localization": {
        "impl": SatCLIPPipeline,
        "default": {"model": (SatCLIPModel,)},
    },
}


def register_satclip_pipeline():
    """Register SatCLIP pipeline tasks with the Transformers library."""
    from transformers.pipelines import PIPELINE_REGISTRY

    for task, spec in SATCLIP_TASKS.items():
        PIPELINE_REGISTRY.register_pipeline(
            task,
            pipeline_class=spec["impl"],
            pt_model=spec["default"]["model"],
        )
