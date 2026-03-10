# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import logging

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    ConformerEncoder,
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class ConformerCTCModule(pl.LightningModule):
    """Conformer (Conv + Transformer) encoder with CTC. Same front-end as TDSConvCTCModule."""

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        num_layers: int,
        n_heads: int,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        entropy_weight: float = 0.0,
        blank_logit_sub: float = 0.0,
        blank_sub_anneal_epochs: int = 5,
        time_reduction_stride: int = 1,
        time_reduction_kernel_size: int = 3,
        local_attention_window: int = 0,
        intermediate_ctc_weight: float = 0.0,
        intermediate_ctc_layer: int = 0,
        pretrained_frontend: str | None = None,
        freeze_frontend: bool = False,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.encoder = ConformerEncoder(
            num_features=num_features,
            num_layers=num_layers,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            time_reduction_stride=time_reduction_stride,
            time_reduction_kernel_size=time_reduction_kernel_size,
            attention_window=local_attention_window,
        )
        self.input_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS
        )
        self.band_mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.classifier = nn.Linear(num_features, charset().num_classes, bias=True)
        self.model = nn.Sequential(
            self.input_norm,
            self.band_mlp,
            self.flatten,
            self.encoder,
            self.classifier,
        )

        # Load pre-trained front-end (SpectrogramNorm + MLP) from TDS checkpoint
        if pretrained_frontend:
            log = logging.getLogger(__name__)
            ckpt = torch.load(pretrained_frontend, map_location="cpu")
            sd = ckpt["state_dict"]
            frontend_sd = {
                k: v for k, v in sd.items()
                if k.startswith("model.0.") or k.startswith("model.1.")
            }
            missing, unexpected = self.load_state_dict(frontend_sd, strict=False)
            log.info(f"Loaded {len(frontend_sd)} frontend params from {pretrained_frontend}")
            if missing:
                log.warning(f"Missing params when loading frontend: {missing}")
            if unexpected:
                log.warning(f"Unexpected params when loading frontend: {unexpected}")
            if freeze_frontend:
                for param in self.model[0].parameters():
                    param.requires_grad = False
                for param in self.model[1].parameters():
                    param.requires_grad = False
                log.info("Froze SpectrogramNorm + MLP frontend")

        with torch.no_grad():
            self.classifier.bias.zero_()
        self.blank_id = charset().null_class
        self.entropy_weight = entropy_weight
        self.blank_logit_sub = blank_logit_sub
        self.blank_sub_anneal_epochs = blank_sub_anneal_epochs
        self.intermediate_ctc_weight = intermediate_ctc_weight
        self.intermediate_ctc_layer = intermediate_ctc_layer
        if self.intermediate_ctc_weight > 0 and not 1 <= self.intermediate_ctc_layer <= num_layers:
            raise ValueError(
                "intermediate_ctc_layer must be within [1, num_layers] when "
                "intermediate_ctc_weight > 0."
            )

        self.ctc_loss = nn.CTCLoss(blank=self.blank_id)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        self._printed_debug = False

    def _current_blank_sub(self) -> float:
        if self.blank_sub_anneal_epochs <= 0 or self.blank_logit_sub <= 0:
            return self.blank_logit_sub
        progress = min(1.0, self.current_epoch / self.blank_sub_anneal_epochs)
        return self.blank_logit_sub * (1.0 - progress)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_norm(inputs)
        x = self.band_mlp(x)
        x = self.flatten(x)
        encoder_output = self.encoder(x, lengths=input_lengths)
        if isinstance(encoder_output, tuple):
            encoder_output = encoder_output[0]
        logits = self.classifier(encoder_output)
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        x = self.input_norm(inputs)
        x = self.band_mlp(x)
        x = self.flatten(x)
        encoder_output = self.encoder(
            x,
            lengths=input_lengths,
            intermediate_layer=self.intermediate_ctc_layer
            if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layer > 0
            else None,
        )
        intermediate_hidden = None
        if isinstance(encoder_output, tuple):
            final_hidden, intermediate_hidden = encoder_output
        else:
            final_hidden = encoder_output

        raw_logits = self.classifier(final_hidden)
        emission_lengths = self.encoder.output_lengths(input_lengths).clamp_max(
            final_hidden.shape[0]
        )

        # Blank logit subtraction during training (annealed over epochs)
        ctc_logits = raw_logits
        if phase == "train" and self.blank_logit_sub > 0:
            sub = self._current_blank_sub()
            if sub > 0:
                ctc_logits = raw_logits.clone()
                ctc_logits[..., self.blank_id] -= sub

        emissions = torch.nn.functional.log_softmax(ctc_logits, dim=-1)

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        if intermediate_hidden is not None:
            aux_logits = self.classifier(intermediate_hidden)
            aux_emissions = torch.nn.functional.log_softmax(aux_logits, dim=-1)
            aux_loss = self.ctc_loss(
                log_probs=aux_emissions,
                targets=targets.transpose(0, 1),
                input_lengths=emission_lengths,
                target_lengths=target_lengths,
            )
            loss = loss + self.intermediate_ctc_weight * aux_loss
            self.log(f"{phase}/aux_ctc_loss", aux_loss, batch_size=N, sync_dist=True)

        # Entropy regularization (supplementary)
        if self.training and self.entropy_weight > 0:
            probs = torch.exp(emissions)
            entropy = -(probs * emissions).sum(dim=-1).mean()
            loss = loss - self.entropy_weight * entropy

        # Monitor and decode using clean (unmodified) emissions so metrics stay
        # aligned with the model's actual output distribution.
        with torch.no_grad():
            clean_emissions = torch.nn.functional.log_softmax(
                raw_logits.detach(), dim=-1)
            _p = torch.exp(clean_emissions)
            self.log(f"{phase}/p_blank", _p[..., self.blank_id].mean(),
                     batch_size=N, sync_dist=True)
            self.log(f"{phase}/entropy",
                     -(_p * clean_emissions).sum(dim=-1).mean(),
                     batch_size=N, sync_dist=True)

        predictions = self.decoder.decode_batch(
            emissions=clean_emissions.cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()

        # Per-epoch val debug: target/pred text, lengths, pred token top-5, blank prob
        if phase == "val" and not self._printed_debug:
            self._printed_debug = True
            probs = torch.exp(emissions.detach())
            mean_p_blank = probs[..., self.blank_id].mean().item()
            n_show = min(2, N)
            all_pred_ids = []
            for i in range(N):
                all_pred_ids.extend(predictions[i].labels.tolist())
            top5 = Counter(all_pred_ids).most_common(5)
            lines = [
                "[ConformerCTCModule val debug (once)]",
                f"blank_id={self.blank_id}  mean_P(blank)={mean_p_blank:.4f}",
                f"pred token top-5: {top5}",
            ]
            for i in range(n_show):
                t = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
                target_text = ascii(t.text[:60])
                pred_text = ascii(predictions[i].text[:60])
                lines.append(
                    f"  [{i}] target_len={len(t)} pred_len={len(predictions[i])}  "
                    f"target={target_text}  pred={pred_text}"
                )
            log = logging.getLogger(__name__)
            log.info("\n".join(lines))

        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_start(self) -> None:
        self._printed_debug = False

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
