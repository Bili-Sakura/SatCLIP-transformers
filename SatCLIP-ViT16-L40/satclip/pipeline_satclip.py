"""SatCLIP feature-extraction pipeline."""

import numpy as np
import torch
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.utils import add_end_docstrings

from .image_processing_satclip import SatCLIPImageProcessor
from .modeling_satclip import SatCLIPModel


@add_end_docstrings(
    """
        modality (`str`, *optional*, defaults to `"location"`):
            Which encoder to use: `"location"` for (lon, lat) coordinates or `"image"` for satellite imagery.
    """,
)
class SatCLIPFeatureExtractionPipeline(FeatureExtractionPipeline):
    """
    Extract SatCLIP embeddings for geographic coordinates or Sentinel-2 images.

    Example (location encoding):

    ```python
    >>> from transformers import pipeline
    >>> extractor = pipeline(
    ...     "feature-extraction",
    ...     model="BiliSakura/SatCLIP-transformers",
    ...     subfolder="SatCLIP-ResNet18-L10",
    ...     trust_remote_code=True,
    ... )
    >>> result = extractor({"longitude": 10.5, "latitude": 48.1}, return_tensors=True)
    >>> result.shape
    torch.Size([1, 512])
    ```
    """

    _load_processor = False
    _load_tokenizer = False
    _load_image_processor = False
    _load_feature_extractor = False

    def _sanitize_parameters(self, modality=None, return_tensors=None, **kwargs):
        preprocess_params = {}
        if modality is not None:
            preprocess_params["modality"] = modality
        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors
        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs=None, *args, modality="location", **kwargs):
        if inputs is None and args:
            inputs = args[0]
        if inputs is None:
            coord_keys = ("longitude", "latitude", "coords", "image", "images")
            inputs = {key: kwargs.pop(key) for key in coord_keys if key in kwargs}
            if not inputs:
                raise ValueError(
                    "SatCLIP feature extraction expects coordinates or images. "
                    "Pass a dict with `longitude`/`latitude`, `coords`, or `image`."
                )
        return super().__call__(inputs, modality=modality, **kwargs)

    def _image_batch(self, image):
        if self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"]
        else:
            processed = SatCLIPImageProcessor()(image, return_tensors="pt")
            pixel_values = processed["pixel_values"]
        return {
            "pixel_values": pixel_values.to(self.device),
            "modality": "image",
        }

    def preprocess(self, inputs, modality="location", **kwargs):
        if isinstance(inputs, dict):
            if modality == "location" or ("longitude" in inputs or "latitude" in inputs or "coords" in inputs):
                if "coords" in inputs:
                    coords = np.asarray(inputs["coords"], dtype=np.float32)
                else:
                    lon = inputs["longitude"]
                    lat = inputs["latitude"]
                    if isinstance(lon, (int, float)):
                        coords = np.array([[lon, lat]], dtype=np.float32)
                    else:
                        coords = np.array(list(zip(lon, lat)), dtype=np.float32)
                return {
                    "coords": torch.tensor(coords, dtype=torch.float32, device=self.device),
                    "modality": "location",
                }

            image = inputs.get("image", inputs.get("images"))
            if image is not None:
                return self._image_batch(image)

        if modality == "image":
            return self._image_batch(inputs)

        coords = np.asarray(inputs, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords[None, :]
        return {
            "coords": torch.tensor(coords, dtype=torch.float32, device=self.device),
            "modality": "location",
        }

    def _forward(self, model_inputs):
        modality = model_inputs.pop("modality", "location")
        if modality == "image":
            embeddings = self.model.get_image_features(model_inputs["pixel_values"])
        else:
            embeddings = self.model.get_location_features(model_inputs["coords"])
        return embeddings

    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs
        return model_outputs.detach().cpu().numpy()


CUSTOM_PIPELINES = {
    "feature-extraction": {
        "impl": "satclip.pipeline_satclip.SatCLIPFeatureExtractionPipeline",
        "pt": ["AutoModel"],
    }
}
