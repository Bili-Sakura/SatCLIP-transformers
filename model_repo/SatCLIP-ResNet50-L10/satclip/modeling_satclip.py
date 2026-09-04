"""SatCLIP model implementation compatible with Hugging Face Transformers."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .configuration_satclip import SatCLIPConfig
from .modeling_location_encoder import LocationEncoder, get_neural_network, get_positional_encoding
from .modeling_vision import ModifiedResNet, VisionTransformer, build_vision_encoder

logger = logging.get_logger(__name__)


@dataclass
class SatCLIPOutput(ModelOutput):
    """Output of SatCLIP forward pass."""

    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_location: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    location_embeds: Optional[torch.FloatTensor] = None


@dataclass
class SatCLIPEncoderOutput(ModelOutput):
    """Output of SatCLIP encoder methods."""

    embeddings: torch.FloatTensor = None


class SatCLIPPreTrainedModel(PreTrainedModel):
    config_class = SatCLIPConfig
    base_model_prefix = "satclip"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LocationEncoder"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SatCLIPModel(SatCLIPPreTrainedModel):
    """
    SatCLIP contrastive model with image and location encoders.

    This model can be loaded with ``AutoModel.from_pretrained`` when the SatCLIP
    package is installed or when ``trust_remote_code=True`` is used with a Hub
    repository that ships the custom code.
    """

    def __init__(self, config: SatCLIPConfig):
        super().__init__(config)
        self.config = config

        if not config.return_location_encoder_only:
            self.visual = build_vision_encoder(
                embed_dim=config.embed_dim,
                image_resolution=config.image_resolution,
                vision_layers=config.vision_layers,
                vision_width=config.vision_width,
                vision_patch_size=config.vision_patch_size,
                in_channels=config.in_channels,
            )
        else:
            self.visual = None

        self.posenc = get_positional_encoding(
            name=config.le_type,
            harmonics_calculation=config.harmonics_calculation,
            legendre_polys=config.legendre_polys,
            min_radius=config.min_radius,
            max_radius=config.max_radius,
            frequency_num=config.frequency_num,
        ).double()
        self.nnet = get_neural_network(
            name=config.pe_type,
            input_dim=self.posenc.embedding_dim,
            num_classes=config.embed_dim,
            dim_hidden=config.capacity,
            num_layers=config.num_hidden_layers,
        ).double()
        self.location = LocationEncoder(self.posenc, self.nnet).double()

        if not config.return_location_encoder_only:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self._initialize_vision_parameters()

        self.post_init()

    def _initialize_vision_parameters(self):
        if self.visual is None:
            return
        if isinstance(self.visual, ModifiedResNet) and self.visual.attnpool is not None:
            std = self.visual.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self) -> torch.dtype:
        if self.visual is None:
            return next(self.location.parameters()).dtype
        try:
            import timm

            if isinstance(self.visual, timm.models.vision_transformer.VisionTransformer):
                return self.visual.patch_embed.proj.weight.dtype
        except ImportError:
            pass
        if isinstance(self.visual, VisionTransformer):
            return self.visual.conv1.weight.dtype
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.FloatTensor:
        if self.visual is None:
            raise ValueError("This model was initialized with return_location_encoder_only=True.")
        return self.visual(image.type(self.dtype))

    def encode_location(self, coords: torch.Tensor) -> torch.FloatTensor:
        return self.location(coords.double()).float()

    def get_image_features(self, image: torch.Tensor) -> torch.FloatTensor:
        features = self.encode_image(image)
        return features / features.norm(dim=-1, keepdim=True)

    def get_location_features(self, coords: torch.Tensor) -> torch.FloatTensor:
        features = self.encode_location(coords)
        return features / features.norm(dim=-1, keepdim=True)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SatCLIPOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = None
        location_features = None
        logits_per_image = None
        logits_per_location = None

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
        if coords is not None:
            location_features = self.get_location_features(coords)

        if image_features is not None and location_features is not None:
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ location_features.t()
            logits_per_location = logits_per_image.t()

        if not return_dict:
            return tuple(
                v
                for v in (logits_per_image, logits_per_location, image_features, location_features)
                if v is not None
            )

        return SatCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_location=logits_per_location,
            image_embeds=image_features,
            location_embeds=location_features,
        )


class SatCLIPLocationEncoderModel(SatCLIPPreTrainedModel):
    """Location encoder only — matches the default downstream inference use case."""

    def __init__(self, config: SatCLIPConfig):
        config.return_location_encoder_only = True
        super().__init__(config)
        self.config = config

        self.posenc = get_positional_encoding(
            name=config.le_type,
            harmonics_calculation=config.harmonics_calculation,
            legendre_polys=config.legendre_polys,
            min_radius=config.min_radius,
            max_radius=config.max_radius,
            frequency_num=config.frequency_num,
        ).double()
        self.nnet = get_neural_network(
            name=config.pe_type,
            input_dim=self.posenc.embedding_dim,
            num_classes=config.embed_dim,
            dim_hidden=config.capacity,
            num_layers=config.num_hidden_layers,
        ).double()
        self.location = LocationEncoder(self.posenc, self.nnet).double()
        self.post_init()

    def forward(
        self,
        coords: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[SatCLIPEncoderOutput, torch.FloatTensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embeddings = self.location(coords.double()).float()
        if not return_dict:
            return embeddings
        return SatCLIPEncoderOutput(embeddings=embeddings)

    def encode_location(self, coords: torch.Tensor) -> torch.FloatTensor:
        return self.forward(coords, return_dict=False)


def load_lightning_state_dict(checkpoint_path: str, device: Union[str, torch.device] = "cpu") -> dict:
    """Load and normalize a Lightning checkpoint state dict for SatCLIPModel."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    return {key.replace("model.", "", 1): value for key, value in state_dict.items() if key.startswith("model.")}


def load_location_encoder_state_dict(checkpoint_path: str, device: Union[str, torch.device] = "cpu") -> dict:
    """Load only location encoder weights from a Lightning checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {}
    for key, value in ckpt["state_dict"].items():
        if "nnet" in key:
            normalized = key[key.index("nnet") :]
            state_dict[normalized] = value
    return state_dict
