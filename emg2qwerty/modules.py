# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
from torch import nn


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


# -----------------------------------------------------------------------------
# Conformer (Convolution + Transformer) for ConformerCTCModule
# -----------------------------------------------------------------------------


def lengths_to_padding_mask(
    lengths: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    """Build a key-padding mask of shape (N, T) from per-sample lengths."""

    if max_len is None:
        max_len = int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) >= lengths.unsqueeze(1)


def apply_padding_mask(
    x: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Zero out padded timesteps for time-first tensors shaped as (T, N, D)."""

    if padding_mask is None:
        return x
    return x.masked_fill(padding_mask.T.unsqueeze(-1), 0.0)


class ConformerFeedForward(nn.Module):
    """Half-step feed-forward in Conformer: Linear -> Swish -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class ConformerConv(nn.Module):
    """Depthwise conv + pointwise in Conformer. Input (T, N, D) -> (T, N, D)."""

    def __init__(self, d_model: int, kernel_size: int = 31) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (T, N, D) -> (N, D, T)
        x = apply_padding_mask(x, padding_mask)
        x = x.permute(1, 2, 0)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.permute(2, 0, 1)  # (T, N, D)
        x = self.layer_norm(x)
        return apply_padding_mask(x, padding_mask)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-first inputs (T, N, D)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._build_pe(max_len))

    def _build_pe(self, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        if T > self.pe.shape[0]:
            self.pe = self._build_pe(T).to(device=x.device)
        return self.dropout(x + self.pe[:T].to(dtype=x.dtype))


class TemporalSubsampling1d(nn.Module):
    """Lightweight 1D conv subsampling for time-first inputs (T, N, D)."""

    def __init__(
        self,
        d_model: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        assert stride >= 1
        assert kernel_size % 2 == 1
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        x = x.permute(1, 2, 0)  # (N, D, T)
        x = self.conv(x)
        x = self.activation(x)
        x = x.permute(2, 0, 1)  # (T', N, D)
        return self.layer_norm(x)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return lengths
        return ((lengths + 2 * self.padding - self.kernel_size) // self.stride) + 1


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN -> MHSA -> Conv -> FFN with half-step residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_window: int = 0,
    ) -> None:
        super().__init__()
        self.attention_window = attention_window
        self.ffn1 = ConformerFeedForward(d_model, expansion=ffn_expansion, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = ConformerConv(d_model, kernel_size=conv_kernel_size)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn2 = ConformerFeedForward(d_model, expansion=ffn_expansion, dropout=dropout)
        self.norm4 = nn.LayerNorm(d_model)

    def _attention_mask(self, T: int, device: torch.device) -> torch.Tensor | None:
        if self.attention_window <= 0:
            return None
        # Full-session test sequences can be very long; avoid allocating a dense
        # T x T mask in that case and fall back to unmasked attention.
        if T > 4096:
            return None
        idx = torch.arange(T, device=device)
        distance = (idx[:, None] - idx[None, :]).abs()
        return distance > self.attention_window

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (T, N, D)
        x = apply_padding_mask(x, padding_mask)
        x = self.norm1(x + 0.5 * self.ffn1(x))
        x = apply_padding_mask(x, padding_mask)
        attn_mask = self._attention_mask(x.shape[0], x.device)
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + attn_out)
        x = apply_padding_mask(x, padding_mask)
        x = self.norm3(x + self.conv(x, padding_mask=padding_mask))
        x = apply_padding_mask(x, padding_mask)
        x = self.norm4(x + 0.5 * self.ffn2(x))
        return apply_padding_mask(x, padding_mask)


class ConformerEncoder(nn.Module):
    """Stack of ConformerBlock with optional temporal subsampling."""

    def __init__(
        self,
        num_features: int,
        num_layers: int = 12,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        time_reduction_stride: int = 1,
        time_reduction_kernel_size: int = 3,
        attention_window: int = 0,
    ) -> None:
        super().__init__()
        self.subsampling = TemporalSubsampling1d(
            d_model=num_features,
            stride=time_reduction_stride,
            kernel_size=time_reduction_kernel_size,
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            num_features, dropout=dropout
        )
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=num_features,
                    n_heads=n_heads,
                    ffn_expansion=ffn_expansion,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    attention_window=attention_window,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        intermediate_layer: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.subsampling(inputs)  # (T', N, num_features)
        padding_mask = None
        if lengths is not None:
            output_lengths = self.output_lengths(lengths).clamp_max(x.shape[0])
            padding_mask = lengths_to_padding_mask(output_lengths, max_len=x.shape[0])
        x = self.pos_encoding(x)
        x = apply_padding_mask(x, padding_mask)
        intermediate_output = None
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, padding_mask=padding_mask)
            if intermediate_layer is not None and idx == intermediate_layer:
                intermediate_output = apply_padding_mask(x, padding_mask)

        x = apply_padding_mask(x, padding_mask)

        if intermediate_output is None:
            return x
        return x, intermediate_output

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.subsampling.output_lengths(lengths)


