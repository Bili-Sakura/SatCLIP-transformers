from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils import ModelOutput


@dataclass
class SatCLIPModelOutput(ModelOutput):
    location_embeddings: torch.FloatTensor = None
    image_embeddings: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_location: Optional[torch.FloatTensor] = None
