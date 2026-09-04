from transformers.configuration_utils import PretrainedConfig


class SatCLIPConfig(PretrainedConfig):
    model_type = "satclip"

    def __init__(
        self,
        embed_dim=512,
        le_type="sphericalharmonics",
        pe_type="siren",
        frequency_num=16,
        max_radius=360,
        min_radius=1,
        harmonics_calculation="analytic",
        legendre_polys=10,
        num_hidden_layers=2,
        capacity=256,
        normalize=True,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.le_type = le_type
        self.pe_type = pe_type
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.harmonics_calculation = harmonics_calculation
        self.legendre_polys = legendre_polys
        self.num_hidden_layers = num_hidden_layers
        self.capacity = capacity
        self.normalize = normalize
        super().__init__(**kwargs)

    @classmethod
    def from_satclip_hyperparameters(cls, hparams):
        data = dict(hparams)
        for key in ["eval_downstream", "air_temp_data_path", "election_data_path", "learning_rate", "weight_decay", "sh_embedding_dims", "image_resolution", "vision_layers", "vision_width", "vision_patch_size", "in_channels"]:
            data.pop(key, None)
        return cls(**data)
