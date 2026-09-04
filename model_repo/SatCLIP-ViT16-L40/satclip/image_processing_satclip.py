"""Image preprocessing for SatCLIP."""

from typing import List, Optional, Union

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, to_numpy_array
from transformers.utils import TensorType


class SatCLIPImageProcessor(BaseImageProcessor):
    """
    Preprocesses multi-spectral Sentinel-2 images for SatCLIP.

    The processor:
    - scales reflectance values by 1/10000
    - inserts a zero B10 band to obtain 13 channels
    - optionally crops/resizes to the model resolution
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 13,
        scale_factor: float = 10000.0,
        insert_b10_band: bool = True,
        do_center_crop: bool = False,
        do_resize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.insert_b10_band = insert_b10_band
        self.do_center_crop = do_center_crop
        self.do_resize = do_resize

    def _prepare_channels(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = image[None, ...]

        if image.shape[0] not in (12, 13):
            raise ValueError(f"Expected 12 or 13 spectral bands, got shape {image.shape}.")

        image = image.astype(np.float32) / self.scale_factor

        if self.insert_b10_band and image.shape[0] == 12:
            b10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
            image = np.concatenate([image[:10], b10, image[10:]], axis=0)

        if image.shape[0] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels after preprocessing, got {image.shape[0]}.")

        return image

    def _maybe_resize(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[-2], image.shape[-1]
        if self.do_resize and (height != self.image_size or width != self.image_size):
            import torch
            import torch.nn.functional as F

            tensor = torch.from_numpy(image).unsqueeze(0)
            tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            image = tensor.squeeze(0).numpy()
        elif self.do_center_crop:
            top = max((height - self.image_size) // 2, 0)
            left = max((width - self.image_size) // 2, 0)
            image = image[:, top : top + self.image_size, left : left + self.image_size]
        return image

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            image_list = [images]

        processed: List[np.ndarray] = []

        for image in image_list:
            if not is_valid_image(image):
                raise ValueError(f"Invalid image input: {type(image)}")
            array = to_numpy_array(image)
            array = self._prepare_channels(array)
            array = self._maybe_resize(array)
            processed.append(array)

        data = {"pixel_values": processed}
        return BatchFeature(data=data, tensor_type=return_tensors)
