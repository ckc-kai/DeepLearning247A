# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
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
    
            
        
        