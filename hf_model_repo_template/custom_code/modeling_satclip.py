from typing import Optional, Union

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .configuration_satclip import SatCLIPConfig
from .layers import LocationEncoder, get_neural_network, get_positional_encoding
from .modeling_outputs import SatCLIPModelOutput


class SatCLIPPreTrainedModel(PreTrainedModel):
    config_class = SatCLIPConfig
    base_model_prefix = "satclip"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class SatCLIPModel(SatCLIPPreTrainedModel):
    def __init__(self, config: SatCLIPConfig):
        super().__init__(config)
        posenc = get_positional_encoding(
            name=config.le_type,
            legendre_polys=config.legendre_polys,
            harmonics_calculation=config.harmonics_calculation,
            min_radius=config.min_radius,
            max_radius=config.max_radius,
            frequency_num=config.frequency_num,
        )
        nnet = get_neural_network(
            name=config.pe_type,
            input_dim=posenc.embedding_dim,
            num_classes=config.embed_dim,
            dim_hidden=config.capacity,
            num_layers=config.num_hidden_layers,
        )
        self.location = LocationEncoder(posenc, nnet).double()
        self.post_init()

    def forward(
        self,
        coordinates: torch.Tensor,
        normalize: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SatCLIPModelOutput, tuple]:
        if coordinates is None:
            raise ValueError("`coordinates` is required and must be shaped as [batch, 2].")
        if not torch.is_tensor(coordinates):
            coordinates = torch.as_tensor(coordinates)
        coordinates = coordinates.to(device=self.device)

        location_embeddings = self.location(coordinates.double()).float()
        should_normalize = self.config.normalize if normalize is None else normalize
        if should_normalize:
            location_embeddings = location_embeddings / location_embeddings.norm(dim=1, keepdim=True)

        output = SatCLIPModelOutput(location_embeddings=location_embeddings)
        if return_dict is False:
            return (location_embeddings,)
        return output

    @classmethod
    def from_satclip_checkpoint(cls, ckpt_path: str, map_location: Optional[str] = None):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        config = SatCLIPConfig.from_satclip_hyperparameters(ckpt["hyper_parameters"])
        model = cls(config)

        mapped_state_dict = {}
        for key, value in ckpt.get("state_dict", {}).items():
            if "nnet" in key:
                idx = key.index("nnet")
                mapped_state_dict[f"location.{key[idx:]}"] = value

        model.load_state_dict(mapped_state_dict, strict=False)
        model.eval()
        return model
