"""Coordinate preprocessing for SatCLIP."""

from typing import List, Optional, Union

import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from .image_processing_satclip import SatCLIPImageProcessor


class SatCLIPProcessor(ProcessorMixin):
    """
    Combined processor for SatCLIP image and coordinate inputs.

    Coordinates are expected as ``[longitude, latitude]`` in degrees.
    """

    attributes = ["image_processor"]
    image_processor_class = "SatCLIPImageProcessor"

    def __init__(self, image_processor: Optional[SatCLIPImageProcessor] = None, **kwargs):
        super().__init__(image_processor=image_processor)

    def __call__(
        self,
        images=None,
        coords=None,
        longitude: Optional[Union[float, List[float]]] = None,
        latitude: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchEncoding:
        if coords is None and longitude is not None and latitude is not None:
            if isinstance(longitude, (int, float)):
                coords = np.array([[longitude, latitude]], dtype=np.float32)
            else:
                coords = np.array(list(zip(longitude, latitude)), dtype=np.float32)

        encoding = BatchEncoding()
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors)
            encoding.update(image_inputs)

        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if return_tensors == "pt":
                import torch

                encoding["coords"] = torch.tensor(coords)
            else:
                encoding["coords"] = coords

        return encoding
