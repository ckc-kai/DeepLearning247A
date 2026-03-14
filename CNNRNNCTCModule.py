# ===== 新增：CNN + RNN =====
class CNNRNNCTCModule(pl.LightningModule):

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: int,
        kernel_size: int,
        num_cnn_layers: int,
        rnn_hidden_size: int,
        rnn_layers: int,
        dropout: float,
        bidirectional: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # 前端和原 CNN 版本保持一致
        num_features = self.NUM_BANDS * mlp_features[-1]  # e.g. 2 * 384 = 768

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T, N, F)
        )

        # CNN 提局部时序特征
        layers = []
        in_ch = num_features
        
        for i in range(num_cnn_layers):
            stride = 2 if i == 0 else 1
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=cnn_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = cnn_channels

        self.temporal_cnn = nn.Sequential(*layers)

        # RNN 建模长时依赖
        self.rnn = nn.GRU(
            input_size=cnn_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_features = rnn_hidden_size * (2 if bidirectional else 1)

        # CTC head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.frontend(inputs)      # (T, N, F)
        x = x.permute(1, 2, 0)         # (N, F, T)
        x = self.temporal_cnn(x)       # (N, C, T)
        x = x.permute(2, 0, 1)         # (T, N, C)
        x, _ = self.rnn(x)             # (T, N, H) or (T, N, 2H)
        emissions = self.classifier(x) # (T, N, num_classes)
        return emissions

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # CNN padding 保持时间长度不变，GRU 也不改时间长度
        emission_lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
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
