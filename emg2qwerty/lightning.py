# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
import math
from typing import Any, ClassVar

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
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    CNNCustome,
    MultiBandElectrodeMixer,
    CyRo2FormersEncoder,
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
        pretraining_mode: bool = False,
        masking_ratio: float = 0.3,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.pretraining_mode = pretraining_mode
        self.masking_ratio = masking_ratio
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
        # Dynamically inject the spectral masking transform if pre-training is on
        if self.pretraining_mode:
            from emg2qwerty.transforms import MaskedSpectralTransform, Compose
            if isinstance(self.train_transform, Compose):
                if not any(isinstance(t, MaskedSpectralTransform) for t in self.train_transform.transforms):
                    # Create a new Compose to safely append without mutating references
                    new_transforms = list(self.train_transform.transforms)
                    new_transforms.append(MaskedSpectralTransform(masking_ratio=self.masking_ratio))
                    self.train_transform = Compose(new_transforms)
            else:
                self.train_transform = Compose([self.train_transform, MaskedSpectralTransform(masking_ratio=self.masking_ratio)])

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
                    # window_length=None,
                    # padding=(0, 0),
                    window_length=self.window_length,
                    padding=self.padding,
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
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        # However, due to the window size limitation, we cannot feed the entire
        # session at once. Instead, we feed the session in chunks of
        # window_length.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
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


class CNNCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_channels: Sequence[int],
        convs_per_block: Sequence[int],
        kernel_size: int,
        fc_hidden_channels: Sequence[int],
        dropout2d: float,
        fc_dropout: float,
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
            CNNCustome(
                in_channels=num_features,
                channels=hidden_channels,
                convs_per_block=convs_per_block,
                kernel_size=kernel_size,
                dropout2d=dropout2d,
                fc_dims=fc_hidden_channels,
                fc_dropout=fc_dropout,
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(fc_hidden_channels[-1] if fc_hidden_channels else hidden_channels[-1], charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion (skip the bad results)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

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
        x = self.model(inputs)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Calculate the temporal downsampling ratio caused by MaxPool1d/striding
        T_in = inputs.shape[0]
        T_out = emissions.shape[0]
        # Shrink the input_lengths by the exact downsampling ratio
        emission_lengths = torch.div(input_lengths * T_out, T_in, rounding_mode='floor')
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


class CyRo2FormersCTCModule(pl.LightningModule):
    """Lightning module for the CyRo2Formers architecture.

    Supports dual-input (spectral + optional raw EMG) and toggleable
    components via YAML config for ablation studies.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        # Electrode mixer
        electrode_mixer_heads: int = 4,
        electrode_mixer_dim: int = 384,
        use_cyrope: bool = True,
        # Backbone
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
        # Spectral Pre-training
        pretraining_mode: bool = False,
        masking_ratio: float = 0.3,
        num_clusters: int = 500,
        # Spectral branch
        use_spectral_branch: bool = True,
        spectral_conv_channels: Sequence[int] = (128, 256, 256),
        spectral_kernel_size: int = 5,
        fusion_type: str = "gated",
        # Training
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        #self.use_spectral_branch = use_spectral_branch

        # Derive freq_bins from in_features / electrode_channels
        freq_bins = in_features // self.ELECTRODE_CHANNELS  # 528 // 16 = 33

        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        
        # Branch 1: Local rotation-invariant dense features (uses flattened C x freq)
        self.mlp_extractor = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        
        # Branch 2: Global self-attention across electrodes (uses raw C, freq)
        self.attention_mixer = MultiBandElectrodeMixer(
            in_features=freq_bins,
            d_model=electrode_mixer_dim,
            n_heads=electrode_mixer_heads,
            n_electrodes=self.ELECTRODE_CHANNELS,
            use_cyrope=use_cyrope,
            num_bands=self.NUM_BANDS,
        )
        
        # Combine features from both branches: 
        # (T, N, bands, mlp_dim) and (T, N, bands, attn_dim) -> (T, N, bands, mlp_dim + attn_dim)
        mlp_dim = mlp_features[-1]
        combined_dim = self.NUM_BANDS * (mlp_dim + electrode_mixer_dim)
        self.frontend_proj = nn.Linear(combined_dim, backbone_d_model) if combined_dim != backbone_d_model else nn.Identity()

        # Build the CyRo2Formers core encoder
        self.encoder = CyRo2FormersEncoder(
            backbone_d_model=backbone_d_model,
            backbone_n_layers=backbone_n_layers,
            transformer_n_heads=transformer_n_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_dropout=transformer_dropout,
            use_attention_head=use_attention_head,
            attn_n_layers=attn_n_layers,
            attn_n_heads=attn_n_heads,
            attn_dim_feedforward=attn_dim_feedforward,
            attn_dropout=attn_dropout,
            use_cyrope=use_cyrope,
            pretraining_mode=pretraining_mode,
            masking_ratio=masking_ratio,
            num_clusters=num_clusters
        )

        self.pretraining_mode = pretraining_mode
        self.num_clusters = num_clusters

        self.classifier = nn.Sequential(
            nn.Linear(backbone_d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        if self.pretraining_mode:
            self.pretrain_loss = nn.CrossEntropyLoss()

        # Decoder (Always instantiated for Character Error Rate logging)
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self,
        spectral_input: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 1. Normalize spectrograms
        x_norm = self.spec_norm(spectral_input)
        
        # 2. Extract parallel features
        x_mlp = self.mlp_extractor(x_norm)    # (T, N, bands, mlp_dim)
        x_attn = self.attention_mixer(x_norm) # (T, N, bands, attn_dim)
        
        # 3. Concatenate and project to backbone_d_model
        x_combined = torch.cat([x_mlp, x_attn], dim=-1) # (T, N, bands, mlp + attn)
        x_combined = x_combined.flatten(start_dim=2)    # (T, N, bands * (mlp + attn))
        x_proj = self.frontend_proj(x_combined)         # (T, N, backbone_d_model)
        
        # 4. Sequence Modeling & Classification
        if self.pretraining_mode:
            x_enc, pretrain_logits = self.encoder(x_proj, mask_indices)
            return self.classifier(x_enc), pretrain_logits
        else:
            x_enc = self.encoder(x_proj)
            return self.classifier(x_enc)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        if self.pretraining_mode and "mask_indices" in batch:
            # Multi-Task Learning: Compute Masked Spectral Pre-training logic 
            import pickle
            from pathlib import Path
            
            # 1. Unpack forward pass
            mask_indices = batch["mask_indices"] # (T, N)
            emissions, pretrain_logits = self.forward(inputs, mask_indices)
            
            # 2. Compute TRUE Pseudo-labels for the masked frames
            T, N, bands, C, freq = inputs.shape
            flattened_inputs = inputs.reshape(T * N, bands * C * freq).detach().cpu().numpy()
            
            # Load the K-Means offline model lazily
            if not hasattr(self, "kmeans_model"):
                kmeans_path = Path(f"data/spectre_kmeans_{self.num_clusters}.pkl")
                if not kmeans_path.exists():
                    raise FileNotFoundError(f"generate_spectre_clusters.py has not been run for K={self.num_clusters}!")
                with open(kmeans_path, "rb") as f:
                    self.kmeans_model = pickle.load(f)
            
            # Predict the true cluster IDs
            true_clusters = self.kmeans_model.predict(flattened_inputs) # (T*N,)
            true_clusters = torch.from_numpy(true_clusters).to(inputs.device).long()
            true_clusters = true_clusters.view(T, N)
            
            # 3. Filter logits and targets to ONLY the masked positions
            masked_logits = pretrain_logits[mask_indices] # (num_masked_ops, num_clusters)
            masked_targets = true_clusters[mask_indices]  # (num_masked_ops,)
            
            if masked_logits.shape[0] == 0:
                pretrain_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
            else:
                pretrain_loss = self.pretrain_loss(masked_logits, masked_targets)
            
            self.log(f"{phase}/pretrain_loss", pretrain_loss, batch_size=N, sync_dist=True)

        else:
            # Standard CTC Fine-tuning logic
            if self.pretraining_mode:
                emissions, _ = self.forward(inputs)
            else:
                emissions = self.forward(inputs)
            pretrain_loss = 0.0

        # CTC Loss Computation (always happens)
        T_in = inputs.shape[0]
        T_out = emissions.shape[0]
        emission_lengths = torch.div(
            input_lengths * T_out, T_in, rounding_mode="floor"
        )
    
        ctc_loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )
        
        # Combine losses if doing Multi-Task Learning
        # The pretrain_loss acts as an auxiliary self-supervised regularizer
        total_loss = ctc_loss + pretrain_loss
    
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
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)
    
        self.log(f"{phase}/loss", total_loss, batch_size=N, sync_dist=True)
        self.log(f"{phase}/ctc_loss", ctc_loss, batch_size=N, sync_dist=True)
        
        return total_loss

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
        opt_cfg = self.hparams.optimizer
        assert opt_cfg is not None, "Optimizer must be specified in config"

        # Separate raw_encoder parameters for ablation stability
        raw_encoder_params = set()
        if hasattr(self.encoder, "raw_encoder") and self.encoder.raw_encoder is not None:
            raw_encoder_params = set(self.encoder.raw_encoder.parameters())

        global_params = [p for p in self.parameters() if p not in raw_encoder_params]
        
        # 1D CNNs on raw signals have sharper gradients. Lower LR by 10x to prevent explosion
        param_groups = [
            {"params": global_params, "lr": opt_cfg.lr},
        ]
        if raw_encoder_params:
            param_groups.append(
                {"params": list(raw_encoder_params), "lr": opt_cfg.lr * 0.1}
            )

        # Re-implement utils hook manually to handle param groups
        import hydra
        optimizer = hydra.utils.instantiate(opt_cfg, param_groups)
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler.scheduler, optimizer)
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, scheduler=scheduler)
        
        import omegaconf
        return {
            "optimizer": optimizer,
            "lr_scheduler": omegaconf.OmegaConf.to_container(lr_scheduler),
        }

