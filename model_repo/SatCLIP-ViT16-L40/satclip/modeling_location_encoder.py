"""Location encoder and neural network building blocks."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .positional_encoding import (
    Cartesian3D,
    Direct,
    DiscretizedSphericalHarmonics,
    GridAndSphere,
    SphericalHarmonics,
    Theory,
    Wrap,
)


class ResLayer(nn.Module):
    def __init__(self, linear_size: int):
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class FCNet(nn.Module):
    def __init__(self, num_inputs: int, num_classes: int, dim_hidden: int):
        super().__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, dim_hidden),
            nn.ReLU(inplace=True),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_emb(self.feats(x))


class MLP(nn.Module):
    def __init__(self, input_dim: int, dim_hidden: int, num_layers: int, out_dims: int):
        super().__init__()
        layers = [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()]
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)]
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _exists(val) -> bool:
    return val is not None


def _cast_tuple(val, repeat: int = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: Optional[nn.Module] = None,
        dropout: bool = False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self._init_weights(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def _init_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor], c: float, w0: float):
        dim = self.dim_in
        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        return self.activation(out)


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: Optional[nn.Module] = None,
        degreeinput: bool = False,
        dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList()
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=dropout,
                )
            )

        final_activation = nn.Identity() if not _exists(final_activation) else final_activation
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=False,
        )

    def forward(self, x: torch.Tensor, mods=None) -> torch.Tensor:
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = _cast_tuple(mods, self.num_layers)
        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if _exists(mod):
                x *= rearrange(mod, "d -> () d")
        return self.last_layer(x)


def get_positional_encoding(
    name: str,
    legendre_polys: int = 10,
    harmonics_calculation: str = "analytic",
    min_radius: float = 1.0,
    max_radius: float = 360.0,
    frequency_num: int = 10,
):
    if name == "direct":
        return Direct()
    if name == "cartesian3d":
        return Cartesian3D()
    if name == "sphericalharmonics":
        if harmonics_calculation == "discretized":
            return DiscretizedSphericalHarmonics(legendre_polys=legendre_polys)
        return SphericalHarmonics(
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
        )
    if name == "theory":
        return Theory(min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num)
    if name == "wrap":
        return Wrap()
    if name in ("grid", "spherec", "spherecplus", "spherem", "spheremplus"):
        return GridAndSphere(
            min_radius=min_radius,
            max_radius=max_radius,
            frequency_num=frequency_num,
            name=name,
        )
    raise ValueError(f"{name} is not a known positional encoding.")


def get_neural_network(
    name: str,
    input_dim: int,
    num_classes: int = 256,
    dim_hidden: int = 256,
    num_layers: int = 2,
):
    if name == "linear":
        return nn.Linear(input_dim, num_classes)
    if name == "mlp":
        return MLP(input_dim=input_dim, dim_hidden=dim_hidden, num_layers=num_layers, out_dims=num_classes)
    if name == "siren":
        return SirenNet(
            dim_in=input_dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            dim_out=num_classes,
        )
    if name == "fcnet":
        return FCNet(num_inputs=input_dim, num_classes=num_classes, dim_hidden=dim_hidden)
    raise ValueError(f"{name} is not a known neural network.")


class LocationEncoder(nn.Module):
    """Encodes (longitude, latitude) coordinates into embedding vectors."""

    def __init__(self, posenc: nn.Module, nnet: nn.Module):
        super().__init__()
        self.posenc = posenc
        self.nnet = nnet

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.nnet(self.posenc(coords))
