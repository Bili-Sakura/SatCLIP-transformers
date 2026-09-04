import math

import torch
from torch import nn


class Direct(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 2

    def forward(self, coords):
        return torch.deg2rad(coords) - torch.pi


class Cartesian3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 3

    def forward(self, coords):
        coords = torch.deg2rad(coords)
        cos_lon = torch.cos(coords[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(coords[:, 0]).unsqueeze(-1)
        cos_lat = torch.cos(coords[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(coords[:, 1]).unsqueeze(-1)
        return torch.cat((cos_lon * cos_lat, sin_lon * cos_lat, sin_lat), dim=1)


class Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 4

    def forward(self, coords):
        coords = torch.deg2rad(coords)
        cos_lon = torch.cos(coords[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(coords[:, 0]).unsqueeze(-1)
        cos_lat = torch.cos(coords[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(coords[:, 1]).unsqueeze(-1)
        return torch.cat((cos_lon, sin_lon, cos_lat, sin_lat), dim=1)


def _freq_list(frequency_num, min_radius, max_radius, device, dtype):
    if frequency_num <= 1:
        return torch.tensor([1.0 / float(min_radius)], device=device, dtype=dtype)
    inc = math.log(float(max_radius) / float(min_radius)) / (frequency_num - 1)
    idx = torch.arange(frequency_num, device=device, dtype=dtype)
    timescales = min_radius * torch.exp(idx * inc)
    return 1.0 / timescales


class Theory(nn.Module):
    def __init__(self, frequency_num=16, max_radius=10000, min_radius=1000):
        super().__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.embedding_dim = int(2 * 3 * frequency_num)
        self.register_buffer("unit_vec", torch.tensor([
            [1.0, 0.0],
            [-0.5, math.sqrt(3) / 2.0],
            [-0.5, -math.sqrt(3) / 2.0],
        ], dtype=torch.float64), persistent=False)

    def forward(self, coords):
        coords = coords.to(torch.float64)
        freq = _freq_list(self.frequency_num, self.min_radius, self.max_radius, coords.device, coords.dtype)
        angle = coords @ self.unit_vec.T
        angle = torch.stack([angle, angle], dim=-1).reshape(coords.shape[0], 1, 6)
        x = angle * freq.view(1, -1, 1)
        x[..., 0::2] = torch.sin(x[..., 0::2])
        x[..., 1::2] = torch.cos(x[..., 1::2])
        return x.reshape(coords.shape[0], -1)


class GridAndSphere(nn.Module):
    def __init__(self, frequency_num=16, max_radius=0.01, min_radius=0.00001, name="grid"):
        super().__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.name = name
        factors = {
            "grid": 4,
            "spherec": 6,
            "spherecplus": 12,
            "spherem": 10,
            "spheremplus": 16,
        }
        if name not in factors:
            raise ValueError(f"Unknown grid/sphere encoding: {name}")
        self.embedding_dim = factors[name] * frequency_num

    def forward(self, coords):
        coords = coords.to(torch.float64)
        freq = _freq_list(self.frequency_num, self.min_radius, self.max_radius, coords.device, coords.dtype)
        lon = coords[:, 0:1].unsqueeze(-1) * freq.view(1, 1, -1)
        lat = coords[:, 1:2].unsqueeze(-1) * freq.view(1, 1, -1)
        lon_sin, lon_cos = torch.sin(lon), torch.cos(lon)
        lat_sin, lat_cos = torch.sin(lat), torch.cos(lat)

        if self.name == "grid":
            out = torch.cat([torch.sin(lon), torch.cos(lon), torch.sin(lat), torch.cos(lat)], dim=1)
        elif self.name == "spherec":
            out = torch.cat([lat_sin, lat_cos * lon_cos, lat_cos * lon_sin], dim=1)
        elif self.name == "spherecplus":
            out = torch.cat([lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_cos, lat_cos * lon_sin], dim=1)
        elif self.name == "spherem":
            lon0, lat0 = coords[:, 0:1], coords[:, 1:2]
            lon0_sin, lon0_cos = torch.sin(lon0).unsqueeze(-1), torch.cos(lon0).unsqueeze(-1)
            lat0_cos = torch.cos(lat0).unsqueeze(-1)
            out = torch.cat([lat_sin, lat_cos * lon0_cos, lat0_cos * lon_cos, lat_cos * lon0_sin, lat0_cos * lon_sin], dim=1)
        else:
            lon0, lat0 = coords[:, 0:1], coords[:, 1:2]
            lon0_sin, lon0_cos = torch.sin(lon0).unsqueeze(-1), torch.cos(lon0).unsqueeze(-1)
            lat0_cos = torch.cos(lat0).unsqueeze(-1)
            out = torch.cat([
                lat_sin, lat_cos, lon_sin, lon_cos,
                lat_cos * lon0_cos, lat0_cos * lon_cos,
                lat_cos * lon0_sin, lat0_cos * lon_sin,
            ], dim=1)
        return out.reshape(coords.shape[0], -1)


def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _sh_renorm(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m)))


def _real_sh(m, l, phi, theta):
    if m == 0:
        return _sh_renorm(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    if m > 0:
        return math.sqrt(2.0) * _sh_renorm(l, m) * torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    return math.sqrt(2.0) * _sh_renorm(l, -m) * torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))


class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys=10):
        super().__init__()
        self.L = int(legendre_polys)
        self.embedding_dim = self.L * self.L

    def forward(self, lonlat):
        lon = lonlat[:, 0]
        lat = lonlat[:, 1]
        phi = torch.deg2rad(lon + 180.0)
        theta = torch.deg2rad(lat + 90.0)
        y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y.append(_real_sh(m, l, phi, theta))
        return torch.stack(y, dim=-1)


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, dim_hidden):
        super().__init__()
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=False)
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.class_emb(self.feats(x))


class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super().__init__()
        layers = [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()]
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1.0, c=6.0, is_first=False, use_bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        w_std = (1 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        weight.uniform_(-w_std, w_std)
        if bias is not None:
            bias.uniform_(-w_std, w_std)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0)

    def forward(self, x):
        return self.activation(torch.nn.functional.linear(x, self.weight, self.bias))


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1.0, w0_initial=30.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            is_first = idx == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            self.layers.append(Siren(layer_dim_in, dim_hidden, w0=layer_w0, is_first=is_first))
        self.last_layer = Siren(dim_hidden, dim_out, w0=w0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class LocationEncoder(nn.Module):
    def __init__(self, posenc, nnet):
        super().__init__()
        self.posenc = posenc
        self.nnet = nnet

    def forward(self, x):
        return self.nnet(self.posenc(x))


def get_positional_encoding(name, legendre_polys=10, harmonics_calculation="analytic", min_radius=1, max_radius=360, frequency_num=10):
    if name == "direct":
        return Direct()
    if name == "cartesian3d":
        return Cartesian3D()
    if name == "sphericalharmonics":
        return SphericalHarmonics(legendre_polys=legendre_polys)
    if name == "theory":
        return Theory(min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num)
    if name == "wrap":
        return Wrap()
    if name in ["grid", "spherec", "spherecplus", "spherem", "spheremplus"]:
        return GridAndSphere(min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num, name=name)
    raise ValueError(f"{name} not a known positional encoding")


def get_neural_network(name, input_dim, num_classes=256, dim_hidden=256, num_layers=2):
    if name == "linear":
        return nn.Linear(input_dim, num_classes)
    if name == "mlp":
        return MLP(input_dim=input_dim, dim_hidden=dim_hidden, num_layers=num_layers, out_dims=num_classes)
    if name == "siren":
        return SirenNet(dim_in=input_dim, dim_hidden=dim_hidden, num_layers=num_layers, dim_out=num_classes)
    if name == "fcnet":
        return FCNet(num_inputs=input_dim, num_classes=num_classes, dim_hidden=dim_hidden)
    raise ValueError(f"{name} not a known neural network")
