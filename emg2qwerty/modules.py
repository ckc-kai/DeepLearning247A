# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn
import logging
from typing import List, Optional
logger = logging.getLogger(__name__)

class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride,
        norm,
        activation,
        dropout1d):
        super().__init__()

        # default padding ensures T dimension does not shrink if stride=1
        padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.norm = self.make_norm(norm, out_channels)
        self.activation = self.make_activation(activation)
        self.dropout = nn.Dropout1d(dropout1d) if dropout1d and dropout1d > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    def make_norm(self, norm, out_channels):
        if norm == "batchnorm":
            return nn.BatchNorm1d(out_channels)
        elif norm == "layernorm":
            return nn.GroupNorm(1, out_channels)
        elif norm == "groupnorm":
            return nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            logger.warning(f"Unknown norm type: {norm}. Do nothing.")
            return nn.Identity()
    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            logger.warning(f"Unknown activation type: {activation}. Use ReLU instead.")
            return nn.ReLU(inplace=True)

class BatchNorm1dPermute(nn.Module):
    """
    Standard PyTorch BatchNorm1d expects (Batch, Channels, Time) but our sequence tensor
    features in the Fully Connected head are passed as (Time, Batch, Channels).
    This wrapper performs the dimensions swap automatically.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected as (T, N, C)
        x = x.permute(1, 2, 0) # (T, N, C) -> (N, C, T)
        x = self.bn(x)
        return x.permute(2, 0, 1) # (N, C, T) -> (T, N, C)

class CNNCustome(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int] = (64, 64, 128, 128, 256), 
        convs_per_block: Sequence[int] = (2, 2, 3, 3, 3),   
        kernel_size: int = 3,
        norm: Optional[str] = "batchnorm",
        activation: str = "relu",
        dropout2d: float = 0.0,
        fc_dims: Sequence[int] = (512, 256), 
        fc_dropout: float = 0.0,
    ):
        super().__init__()
        assert len(channels) == len(convs_per_block), "channels and convs_per_block must have the same length"

        self.features = nn.Sequential()
        prev = in_channels

        for index, (out_f, convs) in enumerate(zip(channels, convs_per_block)):
            layers: List[nn.Module] = []
            for _ in range(convs):
                layers.append(
                    ConvBlock(
                        in_channels= prev,
                        out_channels= out_f,
                        kernel_size=kernel_size,
                        stride=1,
                        norm=norm,
                        activation=activation,
                        dropout1d=dropout2d
                    )
                )
                prev = out_f
            
            # Downsample temporally by 2 after each block of convolutions
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            self.features.add_module(f"block{index}", nn.Sequential(*layers))
    
        head: List[nn.Module] = []
        for i, dim in enumerate(fc_dims):
            head.append(nn.Linear(prev, dim))
            head.append(self.make_norm_1d(norm, dim))
            head.append(self.make_activation(activation))
            head.append(nn.Dropout(fc_dropout))
            prev = dim

        self.head = nn.Sequential(*head)
        self.output_size = prev

    def forward(self, x):
        # x is (Time, Batch, Channels)
        # 1. Permute to (Batch, Channels, Time) for Conv1d
        x = x.permute(1, 2, 0)
        
        # 2. Pass through CNN layers
        x = self.features(x)
        
        # 3. Permute back to (Time, Batch, Channels) for Linear layers and CTC Loss
        x = x.permute(2, 0, 1)
        
        # 4. Apply the fully connected head over the channels dimension
        x = self.head(x)
        return x

    def make_norm_1d(self, norm, out_channels):
        if norm == "batchnorm":
            return BatchNorm1dPermute(out_channels)
        elif norm == "layernorm":
            return nn.LayerNorm(out_channels)
        elif norm == "groupnorm":
            return nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            logger.warning(f"Unknown norm type: {norm}. Do nothing.")
            return nn.Identity()

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            logger.warning(f"Unknown activation type: {activation}. Use ReLU instead.")
            return nn.ReLU(inplace=True)


class CyRoPE(nn.Module):
    """Cylindrical Rotary Positional Embedding for 2D grids (time × electrode).

    For the **electrode axis** the positions wrap around (circular / cylindrical),
    matching the physical ring of 16 electrodes on the wrist band.
    For the **time axis** positions are linear (standard RoPE frequencies).

    The embedding is returned as (cos, sin) pairs that can be used to rotate
    Q/K vectors in an attention layer.

    Args:
        d_model (int): Model dimensionality (must be divisible by 4 to split
            across two 2-D axes, each needing pairs).
        n_electrodes (int): Number of electrode positions (default: 16).
        max_time (int): Maximum temporal length to pre-compute (default: 2048).
        base_freq (float): Base for geometric frequency progression (default: 10000).
    """

    def __init__(
        self,
        d_model: int,
        n_electrodes: int = 16,
        max_time: int = 2048,
        base_freq: float = 10000.0,
    ) -> None:
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2-D RoPE"
        self.base_freq = base_freq
        
        half = d_model // 2  # dims per axis
        quarter = half // 2  # pairs per axis

        # Frequency bands for time axis (standard RoPE)
        freq_time = 1.0 / (
            base_freq ** (torch.arange(0, quarter, dtype=torch.float32) / quarter)
        )
        # Frequency bands for electrode axis (cylindrical – 2π period)
        freq_elec = 2.0 * math.pi * torch.arange(1, quarter + 1, dtype=torch.float32)

        # Pre-compute time positions [max_time] × freqs → [max_time, quarter]
        t_pos = torch.arange(max_time, dtype=torch.float32)
        theta_time = torch.outer(t_pos, freq_time)  # (max_time, quarter)

        # Pre-compute electrode positions [n_electrodes] × freqs → [n_elec, quarter]
        e_pos = torch.arange(n_electrodes, dtype=torch.float32) / n_electrodes
        theta_elec = torch.outer(e_pos, freq_elec)  # (n_electrodes, quarter)

        self.register_buffer("theta_time", theta_time, persistent=False)
        self.register_buffer("theta_elec", theta_elec, persistent=False)

    def _get_time_angles(self, T: int, quarter: int, device: torch.device) -> torch.Tensor:
        """Dynamically generate time angles if T exceeds pre-computed max_time."""
        if T <= self.theta_time.shape[0]:
            return self.theta_time[:T]
            
        # T exceeds max_time (e.g., during testing with full sessions)
        # Compute frequencies dynamically without bounding
        freq_time = 1.0 / (
            self.base_freq ** (torch.arange(0, quarter, dtype=torch.float32, device=device) / quarter)
        )
        t_pos = torch.arange(T, dtype=torch.float32, device=device)
        return torch.outer(t_pos, freq_time)  # (T, quarter)

    def forward(
        self, T: int, C: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape (T, C, d_model//2) for Q/K rotation.

        First quarter dims encode time, second quarter encode electrode.
        """
        quarter = self.theta_time.shape[1]
        
        # Time angles: (T, quarter) → (T, 1, quarter) → broadcast to (T, C, quarter)
        theta_time = self._get_time_angles(T, quarter, self.theta_time.device)
        t_cos = theta_time.unsqueeze(1).expand(-1, C, -1).cos()
        t_sin = theta_time.unsqueeze(1).expand(-1, C, -1).sin()

        # Electrode angles: (C, quarter) → (1, C, quarter) → broadcast to (T, C, quarter)
        # C is strictly bounded by n_electrodes (e.g. 16)
        e_cos = self.theta_elec[:C].unsqueeze(0).expand(T, -1, -1).cos()
        e_sin = self.theta_elec[:C].unsqueeze(0).expand(T, -1, -1).sin()

        cos_pe = torch.cat([t_cos, e_cos], dim=-1)  # (T, C, half)
        sin_pe = torch.cat([t_sin, e_sin], dim=-1)  # (T, C, half)
        return cos_pe, sin_pe


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to tensor x of shape (..., d).

    Splits last dim in half, applies rotation, concatenates back.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ElectrodeMixerAttention(nn.Module):
    """Multi-head attention over the **electrode** dimension for a single band.

    Input shape: (T, N, C, F)  where C = electrode channels, F = freq bins.
    Output shape: (T, N, d_out).

    At each time step, the C=16 electrodes are the "sequence" for attention.
    CyRoPE encodes the 2-D (time, electrode) position as a bias.

    Args:
        in_features (int): Input feature size per electrode (= freq bins or MLP output).
        d_model (int): Projection dimension for Q/K/V (must be div by 4 for CyRoPE).
        n_heads (int): Number of attention heads.
        n_electrodes (int): Number of electrodes per band (default: 16).
        use_cyrope (bool): Whether to inject CyRoPE (default: True).
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_heads: int = 4,
        n_electrodes: int = 16,
        use_cyrope: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.use_cyrope = use_cyrope

        self.q_proj = nn.Linear(in_features, d_model)
        self.k_proj = nn.Linear(in_features, d_model)
        self.v_proj = nn.Linear(in_features, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Projection from in_features to d_model for skip connection
        self.input_proj = (
            nn.Linear(in_features, d_model)
            if in_features != d_model
            else nn.Identity()
        )

        if use_cyrope:
            self.cyrope = CyRoPE(
                d_model=self.head_dim,
                n_electrodes=n_electrodes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, N, C, F) → output: (T, N, d_model)"""
        T, N, C, F_in = x.shape

        # Flatten T*N for batched attention over C electrodes
        x_flat = x.reshape(T * N, C, F_in)  # (T*N, C, F)

        q = self.q_proj(x_flat)  # (T*N, C, d_model)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # Reshape to multi-head: (T*N, C, n_heads, head_dim)
        q = q.view(T * N, C, self.n_heads, self.head_dim)
        k = k.view(T * N, C, self.n_heads, self.head_dim)
        v = v.view(T * N, C, self.n_heads, self.head_dim)

        if self.use_cyrope:
            # CyRoPE over (T, C)
            cos, sin = self.cyrope(T, C)  # each (T, C, head_dim//2)
            # Expand for batch: (T, 1, C, head_dim//2) → (T*N, C, head_dim//2)
            cos = cos.unsqueeze(1).expand(-1, N, -1, -1).reshape(T * N, C, -1)
            sin = sin.unsqueeze(1).expand(-1, N, -1, -1).reshape(T * N, C, -1)
            # Apply per head
            cos_h = cos.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            sin_h = sin.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            q = _apply_rotary(q, cos_h, sin_h)
            k = _apply_rotary(k, cos_h, sin_h)

        # Transpose to (T*N, n_heads, C, head_dim) for attention
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (T*N, n_heads, C, head_dim)

        # Pool over electrodes (mean) → (T*N, n_heads, head_dim)
        attn_out = attn_out.mean(dim=2)
        attn_out = attn_out.reshape(T * N, self.d_model)  # (T*N, d_model)

        out = self.out_proj(attn_out)  # (T*N, d_model)

        # Skip connection + layer norm
        # Pool input over electrodes for residual
        residual = self.input_proj(x_flat.mean(dim=1))  # (T*N, d_model)
        out = self.layer_norm(out + residual)

        return out.view(T, N, self.d_model)  # (T, N, d_model)


class MultiBandElectrodeMixer(nn.Module):
    """Applies ElectrodeMixerAttention independently per band.

    Input: (T, N, num_bands, C, F)
    Output: (T, N, num_bands, d_model)

    Args:
        in_features (int): Per-electrode feature dim (freq bins).
        d_model (int): Output dim per band.
        n_heads (int): Attention heads.
        n_electrodes (int): Electrodes per band.
        use_cyrope (bool): Whether to use CyRoPE.
        num_bands (int): Number of bands (default: 2).
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_heads: int = 4,
        n_electrodes: int = 16,
        use_cyrope: bool = True,
        num_bands: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.mixers = nn.ModuleList(
            [
                ElectrodeMixerAttention(
                    in_features=in_features,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_electrodes=n_electrodes,
                    use_cyrope=use_cyrope,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, N, bands, C, F) → (T, N, bands, d_model)"""
        assert x.shape[2] == self.num_bands
        bands = x.unbind(dim=2)
        outputs = [mixer(band) for mixer, band in zip(self.mixers, bands)]
        return torch.stack(outputs, dim=2)


# ---------------------------------------------------------------------------
# Attention Refinement Head
# ---------------------------------------------------------------------------


class AttentionRefinementHead(nn.Module):
    """Small Transformer encoder for CTC-alignment refinement.

    Optionally injects CyRoPE along the time axis only (electrodes already
    mixed at this point).

    Args:
        d_model (int): Input/output dimension.
        n_layers (int): Number of Transformer encoder layers.
        n_heads (int): Number of attention heads.
        dim_feedforward (int): FFN hidden dimension.
        dropout (float): Dropout rate.
        use_cyrope (bool): Whether to inject CyRoPE on the time axis.
        max_time (int): Maximum time length for positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cyrope: bool = True,
        max_time: int = 2048,
    ) -> None:
        super().__init__()
        self.use_cyrope = use_cyrope
        self.d_model = d_model

        if use_cyrope:
            # Time-only CyRoPE: use d_model-sized positional encoding
            # We need d_model divisible by 4 for CyRoPE; pad/truncate if needed
            self.rope_dim = (d_model // 4) * 4
            if self.rope_dim > 0:
                self._build_time_rope(max_time)
        else:
            # Standard sinusoidal positional encoding
            self._build_standard_pe(max_time)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # expects (T, N, D)
            norm_first=True,    # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _build_time_rope(self, max_time: int, device: Optional[torch.device] = None) -> None:
        """Build sinusoidal positional encoding (time-only, no electrode axis)."""
        d = self.rope_dim
        pos = torch.arange(max_time, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / d)
        )
        pe = torch.zeros(max_time, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("time_pe", pe, persistent=False)  # (max_time, d)

    def _build_standard_pe(self, max_time: int, device: Optional[torch.device] = None) -> None:
        """Build full d_model sinusoidal PE when CyRoPE isn't used."""
        pos = torch.arange(max_time, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_time, 1, self.d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_embed", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, N, D) → (T, N, D)"""
        T = x.shape[0]

        if self.use_cyrope and self.rope_dim > 0:
            if T > self.time_pe.shape[0]:
                self._build_time_rope(T, device=x.device)
                
            # Add sinusoidal time positional encoding
            pe = self.time_pe[:T].unsqueeze(1)  # (T, 1, rope_dim)
            if self.rope_dim < self.d_model:
                # Pad with zeros for remaining dims
                padding = torch.zeros(
                    T, 1, self.d_model - self.rope_dim,
                    device=x.device, dtype=x.dtype,
                )
                pe = torch.cat([pe, padding], dim=-1)
            x = x + pe
        elif not self.use_cyrope:
            if T > self.pos_embed.shape[0]:
                self._build_standard_pe(T, device=x.device)
            x = x + self.pos_embed[:T]

        return self.encoder(x)



# ---------------------------------------------------------------------------
# Top-level Composite Model
# ---------------------------------------------------------------------------


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone for temporal modeling.

    Operates on (T, N, D). Caller should ensure this shape.

    Args:
        d_model (int): Model dimension.
        n_layers (int): Number of Transformer layers.
        n_heads (int): Number of attention heads.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        # Precompute initial sinusoidal positional encodings
        self.max_time = 4096
        self._build_pe(self.max_time)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # expects (T, N, D)
            norm_first=True,    # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _build_pe(self, max_time: int, device: Optional[torch.device] = None) -> None:
        pos = torch.arange(max_time, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_time, 1, self.d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_embed", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, N, D) → (T, N, D)"""
        T = x.shape[0]
        if T > self.pos_embed.shape[0]:
            self._build_pe(T, device=x.device)
            
        x = x + self.pos_embed[:T]
        return self.encoder(x)


class CyRo2FormersEncoder(nn.Module):
    """CyRo2Formers: core temporal encoder and dual-branch fusion.

    Pipeline:
        [Input from frontend] → TransformerBackbone → [optional: fuse with RawWaveformEncoder] →
        [optional: AttentionRefinementHead]
        
    Args:
        backbone_d_model (int): Transformer model dimension.
        backbone_n_layers (int): Number of Transformer blocks.
        transformer_n_heads (int): Number of attention heads.
        transformer_dim_feedforward (int): Transformer FFN dim.
        transformer_dropout (float): Transformer dropout.
        use_attention_head (bool): Toggle attention refinement head.
        attn_n_layers (int): Number of attention layers.
        attn_n_heads (int): Number of attention heads.
        attn_dim_feedforward (int): Attention FFN dim.
        attn_dropout (float): Attention dropout.
        use_cyrope (bool): Toggle CyRoPE in attention head.
        use_spectral_branch (bool): Toggle raw waveform branch + fusion.
        spectral_conv_channels (list): Raw branch conv channels.
        spectral_kernel_size (int): Raw branch kernel size.
        fusion_type (str): Fusion method ("gated", "concat", "add").
        num_bands (int): Number of EMG bands (default: 2).
        n_electrodes (int): Electrodes per band (default: 16).
    """

    def __init__(
        self,
        backbone_d_model: int = 768,
        backbone_n_layers: int = 4,
        transformer_n_heads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_dropout: float = 0.1,
        # Attention refinement
        use_attention_head: bool = True,
        attn_n_layers: int = 2,
        attn_n_heads: int = 8,
        attn_dim_feedforward: int = 1024,
        attn_dropout: float = 0.1,
        use_cyrope: bool = True,
        # Spectral Pre-training
        pretraining_mode: bool = False,
        masking_ratio: float = 0.30,
        num_clusters: int = 500,
    ) -> None:
        super().__init__()
        self.use_attention_head = use_attention_head
        self.pretraining_mode = pretraining_mode
        self.masking_ratio = masking_ratio
        
        if pretraining_mode:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, backbone_d_model))
            # Initialize with small variance
            torch.nn.init.normal_(self.mask_token, std=0.02)
            
            # Lightweight MLP head to predict cluster IDs from masked embeddings
            self.reconstruction_head = nn.Sequential(
                nn.Linear(backbone_d_model, backbone_d_model * 2),
                nn.GELU(),
                nn.LayerNorm(backbone_d_model * 2),
                nn.Linear(backbone_d_model * 2, num_clusters)
            )

        # temporal backbone
        self.backbone = TransformerBackbone(
            d_model=backbone_d_model,
            n_layers=backbone_n_layers,
            n_heads=transformer_n_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )


        # Attention refinement head
        if use_attention_head:
            self.attn_head = AttentionRefinementHead(
                d_model=backbone_d_model,
                n_layers=attn_n_layers,
                n_heads=attn_n_heads,
                dim_feedforward=attn_dim_feedforward,
                dropout=attn_dropout,
                use_cyrope=use_cyrope,
            )

    def forward(
        self,
        spectral_features: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectral_features: (T, N, D) — output from frontend processing.
            mask_indices: (T, N) — Optional boolean mask where True means apply mask_token.

        Returns:
            features: (T, N, D) — encoded features.
            logits: (T, N, num_clusters) — ONLY returned if pretraining_mode=True
        """
        # Apply mask token if pre-training
        if self.pretraining_mode and mask_indices is not None:
            # spectral_features is (T, N, D)
            # mask_indices is (T, N) boolean array
            mask_expanded = mask_indices.unsqueeze(-1) # (T, N, 1)
            # Replace True indices with the learned mask_token
            x = torch.where(mask_expanded, self.mask_token.expand_as(spectral_features), spectral_features)
        else:
            x = spectral_features

        # Temporal backbone
        x = self.backbone(x)    # Transformer expects (T, N, D)

        # Attention refinement head
        if self.use_attention_head:
            x = self.attn_head(x)  # (T, N, D)

        if self.pretraining_mode:
            # Predict cluster probabilities for the pre-training task
            logits = self.reconstruction_head(x) # (T, N, num_clusters)
            return x, logits

        return x