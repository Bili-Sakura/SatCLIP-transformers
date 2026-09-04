from typing import Any, Dict, List

import torch
from transformers import Pipeline


class SatCLIPCoordinatePipeline(Pipeline):
    def preprocess(self, inputs: Any) -> Dict[str, torch.Tensor]:
        coords = inputs["coordinates"] if isinstance(inputs, dict) else inputs
        tensor = torch.as_tensor(coords, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != 2:
            raise ValueError("Expected coordinates shaped as [batch, 2] or [2].")
        return {"coordinates": tensor}

    def _forward(self, model_inputs: Dict[str, torch.Tensor]):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, return_tensors: bool = False):
        embeddings = model_outputs.location_embeddings
        if return_tensors:
            return embeddings
        return embeddings.detach().cpu().tolist()
