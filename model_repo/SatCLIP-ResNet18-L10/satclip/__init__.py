"""SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."""

from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor

from .configuration_satclip import SatCLIPConfig
from .image_processing_satclip import SatCLIPImageProcessor
from .modeling_satclip import SatCLIPLocationEncoderModel, SatCLIPModel
from .pipeline_satclip import SatCLIPPipeline, register_satclip_pipeline
from .processing_satclip import SatCLIPProcessor

__version__ = "0.1.0"

__all__ = [
    "SatCLIPConfig",
    "SatCLIPModel",
    "SatCLIPLocationEncoderModel",
    "SatCLIPImageProcessor",
    "SatCLIPProcessor",
    "SatCLIPPipeline",
    "register_satclip",
]

AutoConfig.register("satclip", SatCLIPConfig)
AutoModel.register(SatCLIPConfig, SatCLIPModel)
AutoImageProcessor.register(SatCLIPConfig, SatCLIPImageProcessor)
AutoProcessor.register(SatCLIPConfig, SatCLIPProcessor)


def register_satclip():
    """Register SatCLIP classes with Transformers Auto* APIs and pipeline registry."""
    register_satclip_pipeline()


# Register on import so `import satclip` is enough for local usage.
register_satclip()
