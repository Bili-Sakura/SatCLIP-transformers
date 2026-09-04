"""SatCLIP model configuration."""

from typing import Optional, Union

from transformers import PretrainedConfig


class SatCLIPConfig(PretrainedConfig):
    """
    Configuration for SatCLIP models.

    SatCLIP is a contrastive model that learns joint embeddings of Sentinel-2
    satellite images and (longitude, latitude) coordinates.
    """

    model_type = "satclip"

    def __init__(
        self,
        embed_dim: int = 512,
        image_resolution: int = 256,
        vision_layers: Union[str, int, list, tuple] = "moco_resnet18",
        vision_width: int = 768,
        vision_patch_size: int = 32,
        in_channels: int = 13,
        le_type: str = "sphericalharmonics",
        pe_type: str = "siren",
        frequency_num: int = 16,
        max_radius: float = 360.0,
        min_radius: float = 1.0,
        harmonics_calculation: str = "analytic",
        legendre_polys: int = 10,
        sh_embedding_dims: int = 16,
        num_hidden_layers: int = 2,
        capacity: int = 256,
        return_location_encoder_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.in_channels = in_channels
        self.le_type = le_type
        self.pe_type = pe_type
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.harmonics_calculation = harmonics_calculation
        self.legendre_polys = legendre_polys
        self.sh_embedding_dims = sh_embedding_dims
        self.num_hidden_layers = num_hidden_layers
        self.capacity = capacity
        self.return_location_encoder_only = return_location_encoder_only

    @property
    def vision_backbone(self) -> str:
        """Human-readable vision backbone identifier."""
        if isinstance(self.vision_layers, str):
            return self.vision_layers
        if isinstance(self.vision_layers, int):
            return f"vit_{self.vision_layers}"
        return "modified_resnet"

    @classmethod
    def from_lightning_hyperparameters(cls, hyper_parameters: dict, **kwargs) -> "SatCLIPConfig":
        """Build a config from a PyTorch Lightning checkpoint hyper_parameters dict."""
        hp = dict(hyper_parameters)
        for key in ("eval_downstream", "air_temp_data_path", "election_data_path", "learning_rate", "weight_decay"):
            hp.pop(key, None)
        hp.update(kwargs)
        return cls(**hp)
