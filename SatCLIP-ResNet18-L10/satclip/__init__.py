"""SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."""

from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor

from .configuration_satclip import SatCLIPConfig
from .image_processing_satclip import SatCLIPImageProcessor
from .modeling_satclip import SatCLIPLocationEncoderModel, SatCLIPModel
from .pipeline_satclip import CUSTOM_PIPELINES, SatCLIPFeatureExtractionPipeline
from .processing_satclip import SatCLIPProcessor

__version__ = "0.1.0"

__all__ = [
    "SatCLIPConfig",
    "SatCLIPModel",
    "SatCLIPLocationEncoderModel",
    "SatCLIPImageProcessor",
    "SatCLIPProcessor",
    "SatCLIPFeatureExtractionPipeline",
    "CUSTOM_PIPELINES",
]

AutoConfig.register("satclip", SatCLIPConfig)
AutoModel.register(SatCLIPConfig, SatCLIPModel)
AutoImageProcessor.register(SatCLIPConfig, SatCLIPImageProcessor)
AutoProcessor.register(SatCLIPConfig, SatCLIPProcessor)
