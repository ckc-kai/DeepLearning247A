# C147/247 Final Project

### Winter 2026

## Table of Contents

1. [CNN Baseline Model](#1-cnn-baseline-model)
2. [Transformer Model (CyRo2Formers — Encoder-Only with CTC)](#2-transformer-model-cyro2formers)
3. [Fine-tuning Stage](#3-fine-tuning-stage)
4. [CTC Beam Search Decoding Results](#4-ctc-beam-search-decoding-results)
5. [Encoder-Decoder Transformer (CyRo2Formers Seq2Seq)](#5-encoder-decoder-transformer-cyro2formers-seq2seq)
6. [How to Run](#6-how-to-run)

## Overview

This repository contains the course project from the `ckc` branch. The project predicts keystroke characters from raw surface electromyography (sEMG) signals. We explore three architectures:

1. **CNN Baseline** — convolutional encoder with CTC loss
2. **CyRo2Formers (Encoder-Only)** — Transformer encoder with CyRoPE, self-supervised pretraining, and CTC decoding
3. **CyRo2Formers (Encoder-Decoder)** — Full Transformer with autoregressive decoder, trained with cross-entropy and teacher forcing

---

## 1. CNN Baseline Model

As a baseline, we implement a Convolutional Neural Network (CNN) with the following architecture:

**Input -> [Conv * 2 -> Max Pool] * 2 -> [Conv * 3 -> Max Pool] * 3 -> Linear(512) -> Linear(128) -> Output**

- **Conv Modules**: Each module includes a 1D convolution, batch normalization, activation, and dropout.
- **Max Pooling**: Kernel size of 2 and stride of 2.
- **Gradient Monitoring**: To mitigate gradient explosion/vanishing, `gradient_clip_val` is set to `1.0` using the `"norm"` algorithm.
- **Configuration**: Further details can be found in `config/model/cnn_ctc.yaml`.

### 1.1 Experimental Results

**Validation Metrics**

| Metric | Score |
| --- | --- |
| CER | 85.27 |
| DER | 0.399 |
| IER | 71.53 |
| SER | 13.34 |
| Loss | 0.623 |

**Test Metrics**

| Metric | Score |
| --- | --- |
| CER | 76.46 |
| DER | 0.0216 |
| IER | 63.11 |
| SER | 13.33 |
| Loss | 0.0 |

**Analysis**:
The results indicate that the CNN baseline model is ill-suited for this task. The Character Error Rate (CER) remains extremely high on both the validation (85.27%) and test (76.46%) sets, signifying that the majority of predicted characters are incorrect. Furthermore, the Insertion Error Rate (IER) exceeds 60%, suggesting the model frequently hallucinates characters. Conversely, the Deletion Error Rate (DER) is near zero, implying the model rarely drops characters but rather predicts incorrect or redundant outputs.

This pattern is consistent with a model lacking sufficient temporal context to correctly align EMG signal patterns with sequence transcriptions. Because the CNN relies on stride-1 convolutions, its receptive field spans only a short temporal window. Specifically, the receptive field is $1 + 13 \times 4 = 53$ frames. At a 2kHz sampling rate, this equates to $53 / 2000 = 0.0265$ seconds (26.5 ms) within a 4-second time window. Consequently, the model fails to capture the comprehensive structure of each gesture, leading to instability in character predictions. This architectural limitation necessitates the exploration of models with broader receptive fields.

---

## 2. Transformer Model (CyRo2Formers)

To address the limitations of the CNN baseline, we introduce a Transformer-based architecture incorporating the following components:

1. **CyRoPE Positional Encoding**: Applied to time and electrode positions to learn consistent spatial relations. CyRoPE is utilized in both the MultibandElectrodeMixer and the Transformer backbone.
2. **MultiBandElectrodeMixer**: Learns global spatial relationships across 4 attention heads using CyRoPE.
3. **Self-supervised Pretraining**: Input data is clustered using K-means, and the model is trained to predict masked cluster labels (similar to Masked Autoencoding).
4. **Transformer Backbone**: A Pre-LN Transformer architecture.
5. **Attention Refinement Head**: Acts as the decoder module.
6. **Fine-tuning**: The pretrained model is fine-tuned with a reduced learning rate over fewer epochs.
7. **Decoding**: Character-level 6-gram language model beam search decoding, consistent with the original paper.

### 2.1 Model Checkpoints and Hyperparameters

The following section tracks the hyperparameters and evaluation metrics across various experimental configurations. **Note:** Hyperparameters have been verified against the respective `config.yaml` logs.

#### Ex 1: Transformer Base (No Pretraining)
- **Checkpoint**: `logs/2026-02-26/01-30-51/checkpoints/epoch=134-step=16200.ckpt`
- **Backbone**: 2 layers, 1024 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 2 layers, 1024 d_ff, 8 heads, 0.10 dropout
- **Masking Strategy**: None

| Dataset | CER | DER | IER | SER | Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 18.077 | 2.326 | 6.358 | 9.393 | 0.629 |
| **Test** | 19.424 | 3.270 | 4.786 | 11.369 | 0.722 |

#### Ex 2: Introduction of Pretraining
- **Checkpoint**: `logs/2026-02-27/01-51-49/checkpoints/epoch_88-step_10680.ckpt`
- **Backbone**: 2 layers, 1024 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 2 layers, 1024 d_ff, 8 heads, 0.10 dropout
- **Masking Strategy**: Random Masking, 0.30 ratio
- **Pretrain Clusters**: 500

| Dataset | CER | DER | IER | SER | Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 15.574 | 1.927 | 5.405 | 8.241 | 0.526 |
| **Test** | 17.410 | 2.252 | 5.392 | 9.766 | 0.592 |

#### Ex 3: Widening and Deepening the Backbone (Best Pretrain)
- **Checkpoint**: `logs/2026-02-27/22-41-30/checkpoints/epoch_118-step_14280.ckpt`
- **Backbone**: 3 layers, 2048 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 2 layers, 1024 d_ff, 8 heads, 0.10 dropout
- **Masking Strategy**: Random Masking, 0.35 ratio
- **Pretrain Clusters**: 500

| Dataset | CER | DER | IER | SER | Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 14.311 | 2.614 | 2.725 | 8.972 | 0.520 |
| **Test** | 15.656 | 2.728 | 3.162 | 9.766 | 0.584 |

**Analysis**: Pretraining demonstrates a marked improvement over the baseline Transformer. Expanding the feed-forward dimension to 2048 and increasing the backbone to 3 layers proved beneficial. As a single user typing on a QWERTY keyboard produces 30-40 distinct character intents, reducing the cluster count may prevent the K-means algorithm from separating data points based on physiological noise.

#### Ex 4: Attempting Span Masking and Shallow Refinement Head
- **Checkpoint**: `logs/2026-03-01/00-03-16/checkpoints/epoch_122-step_14760.ckpt`
- **Backbone**: 2 layers, 2048 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 1 layer, 2048 d_ff, 8 heads, 0.15 dropout
- **Masking Strategy**: Span Masking (length 12), 0.30 ratio
- **Pretrain Clusters**: 250

| Dataset | CER | DER | IER | SER | CTC Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 14.821 | 1.573 | 5.738 | 7.510 | 0.511 |
| **Test** | 17.107 | 1.776 | 6.561 | 8.770 | 0.586 |

**Analysis**: Increasing structural complexity while compressing the refinement head into a single layer created a "memorization trap," hindering performance. We consequently reverted the refinement head to its prior 2-layer configuration.

#### Ex 5: Span Masking with Moderate Parameters
- **Checkpoint**: `logs/2026-03-01/22-57-02/checkpoints/epoch=128-step=15480.ckpt`
- **Backbone**: 2 layers, 2048 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 2 layers, 1024 d_ff, 8 heads, 0.10 dropout
- **Masking Strategy**: Span Masking (length 8), 0.15 ratio
- **Pretrain Clusters**: 250

| Dataset | CER | DER | IER | SER | CTC Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 14.289 | 2.260 | 4.143 | 7.887 | 0.502 |
| **Test** | 16.349 | 2.230 | 4.396 | 9.723 | 0.577 |

#### Ex 6: Optimal Span Masking Model (Second Best Pretrain)
- **Checkpoint**: `logs/2026-03-03/01-22-14/checkpoints/epoch_123-step_14880.ckpt`
- **Backbone**: 2 layers, 2048 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 2 layers, 1024 d_ff, 8 heads, 0.10 dropout
- **Masking Strategy**: Span Masking (length 16), 0.15 ratio
- **Pretrain Clusters**: 500

| Dataset | CER | DER | IER | SER | CTC Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 14.665 | 2.459 | 3.655 | 8.551 | 0.520 |
| **Test** | 15.829 | 2.404 | 4.309 | 9.117 | 0.554 |

**Analysis**: Lengthening the span mask to 16 alongside widening the feed-forward layer to 2048 yielded competitive results, yet marginally trailed the 3-layer random masking architecture (Ex 3) by ~0.17 test CER.

#### Ex 7: Dimensionality Reduction Experiment
- **Checkpoint**: `logs/2026-03-04/01-54-01/checkpoints/epoch=116-step=14040.ckpt`
- **Backbone**: 2 layers, d_model 256, 2048 d_ff, 8 heads, 0.15 dropout
- **Refinement**: 2 layers, 512 d_ff, 8 heads, 0.10 dropout
- **Masking Strategy**: Span Masking (length 16), 0.20 ratio
- **Pretrain Clusters**: 300

| Dataset | CER | DER | IER | SER | CTC Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 25.786 | 1.927 | 12.339 | 11.520 | 0.754 |
| **Test** | 25.747 | 2.187 | 9.809 | 13.751 | 0.763 |

**Analysis**: Severe degradation in performance was observed, confirming that drastic reductions to the backbone's hidden dimension (`d_model: 256`) and the refinement layer's feed-forward dimension (512) severely cripple the network's mapping capacity.

---

## 3. Fine-tuning Stage

We selected our top-performing pretrained architectures and subjected them to fine-tuning.

#### Fine-tune 1: Ex 3 (Best Pretrain)
- **Checkpoint**: `logs/2026-03-05/23-01-29/checkpoints/epoch=7-step=960.ckpt`

| Dataset | CER | DER | IER | SER | CTC Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 14.156 | 2.016 | 3.500 | 8.640 | 0.521 |
| **Test** | 15.180 | 1.927 | 3.941 | 9.311 | 0.584 |

#### Fine-tune 2: Ex 6 (Second Best Pretrain)
- **Checkpoint**: `logs/2026-03-06/22-12-23/checkpoints/epoch_12-step_1560.ckpt`

| Dataset | CER | DER | IER | SER | CTC Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 13.957 | 2.127 | 3.877 | 7.953 | 0.514 |
| **Test** | 16.046 | 2.252 | 4.504 | 9.290 | 0.553 |

---

## 4. CTC Beam Search Decoding Results

Utilizing a sequence-level language model via Beam Search decoding considerably boosts our absolute metrics across both validation and test sets.

#### Ex 3 (Best Pretrain) + CTC Beam Search Decoding

| Metric | Validation | Test |
| --- | --- | --- |
| **CER** | 8.817 | 10.113 |
| **DER** | 2.171 | 2.295 |
| **IER** | 2.060 | 2.728 |
| **SER** | 4.586 | 5.089 |
| **CTC Loss** | 0.520 | 0.584 |

#### Ex 6 (Second Best Pretrain) + CTC Beam Search Decoding

| Metric | Validation | Test |
| --- | --- | --- |
| **CER** | 8.861 | 10.264 |
| **DER** | 2.215 | 2.209 |
| **IER** | 2.370 | 3.530 |
| **SER** | 4.276 | 4.526 |
| **CTC Loss** | 0.520 | 0.554 |

#### Fine-tune 1 (Ex 3) + CTC Beam Search Decoding

| Metric | Validation | Test |
| --- | --- | --- |
| **CER** | 9.304 | 10.221 |
| **DER** | 1.994 | 2.079 |
| **IER** | 2.570 | 3.248 |
| **SER** | 4.741 | 4.894 |
| **CTC Loss** | 0.521 | 0.584 |

#### Fine-tune 2 (Ex 6) + CTC Beam Search Decoding

| Metric | Validation | Test |
| --- | --- | --- |
| **CER** | 8.795 | 10.004 |
| **DER** | 2.060 | 2.122 |
| **IER** | 2.503 | 3.486 |
| **SER** | 4.231 | 4.396 |
| **CTC Loss** | 0.514 | 0.553 |

---

## 5. Encoder-Decoder Transformer (CyRo2Formers Seq2Seq)

To explore whether autoregressive decoding can outperform CTC alignment, we extend the CyRo2Formers encoder with a Transformer decoder trained via cross-entropy loss and teacher forcing.

### 5.1 Architecture

```
Raw sEMG (N, T, 2, bands, electrodes)
        │
        ▼
┌──────────────────────┐
│   TDS Conv Frontend  │  (shared with CTC model)
│   + Band Rotation    │
└──────────┬───────────┘
           │  (N, T, D=768)
           ▼
┌──────────────────────┐
│ MultiBandElectrodeMixer │  CyRoPE (2D: time × electrode)
│   4 heads, d=384     │
└──────────┬───────────┘
           │  (N, T, D)
           ▼
┌──────────────────────┐
│ Transformer Backbone │  3 layers, d_ff=2048, 8 heads
│   TemporalRoPE (1D)  │  Pre-LN, GELU
└──────────┬───────────┘
           │  (N, T, D)
           ▼
┌──────────────────────┐
│ Attention Refinement │  2 layers, d_ff=1024, 8 heads
│   TemporalRoPE (1D)  │  Pre-LN, GELU
└──────────┬───────────┘
           │  encoder_out (T, N, D)
           ▼
┌──────────────────────────────────────────┐
│         Transformer Decoder              │
│  2 layers, d_ff=1024, 8 heads            │
│  Causal self-attn (TemporalRoPE)         │
│  + Cross-attn to encoder_out             │
│  Token embedding + sqrt(d_model) scaling │
└──────────┬───────────────────────────────┘
           │  (S, N, D)
           ▼
┌──────────────────────┐
│  Linear → Softmax    │  vocab = 101 (98 chars + blank + SOS + EOS)
└──────────────────────┘
```

### 5.2 Key Differences from CTC Model

| Aspect | CTC (Encoder-Only) | Seq2Seq (Encoder-Decoder) |
| --- | --- | --- |
| **Loss** | CTC Loss | Cross-Entropy (label smoothing 0.1) |
| **Decoding** | Greedy / Beam Search + LM | Autoregressive (greedy or beam) |
| **Output Length** | Determined by encoder | Determined by decoder (max 50) |
| **Special Tokens** | blank (98) | blank (98), SOS (99), EOS (100) |
| **Positional Encoding** | TemporalRoPE (1D) | TemporalRoPE in both encoder & decoder |
| **Training** | End-to-end | Differential LR (encoder 0.1×, decoder 1.0×) |

### 5.3 Training Strategy

1. **Pretrained Encoder**: Load encoder weights from the best CTC checkpoint (Fine-tune 2, Ex 6) via `pretrained_encoder_ckpt`.
2. **Differential Learning Rate**: Encoder parameters use `base_lr × 0.1` to preserve learned representations; decoder trains at full `base_lr`.
3. **Teacher Forcing**: During training, the decoder receives `[SOS, c1, ..., cN]` and predicts `[c1, ..., cN, EOS]`.
4. **Inference**: Autoregressive decoding — greedy (beam_width=1) or beam search (beam_width>1), stopping at EOS or `max_decode_length=50`.

### 5.4 Experimental Results

#### Ex 8: Encoder-Decoder with Pretrained Encoder

- **Checkpoint**: *(pending)*
- **Encoder**: Loaded from Fine-tune 2 (Ex 6)
- **Decoder**: 2 layers, 1024 d_ff, 8 heads, 0.15 dropout
- **Label Smoothing**: 0.1

| Dataset | CER | DER | IER | SER | CE Loss |
| --- | --- | --- | --- | --- | --- |
| **Validation** | — | — | — | — | — |
| **Test** | — | — | — | — | — |

*(Results to be filled after training completes.)*

---

## 6. How to Run

### CNN Baseline Execution
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cnn_ctc
```

### CyRo2Formers Execution

#### Pretraining Setup
Define the number of clusters $k$ and the output path to initialize the k-means model required for generating cluster pseudo-labels.
```shell
python -m scripts.generate_spectre_clusters --k <k> --out <output_path>
```

#### Pretraining Phase
Enable pretraining by verifying `model.pretraining_mode` is set to `True` in `config/model/cyro2formers_ctc.yaml`.
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cyro2formers_ctc ++model.pretraining_mode=True
```

#### Fine-tuning Phase
Ensure `model.pretraining_mode` is set to `False` in the configuration.
Adjust hyperparameters such as the learning rate and total epochs dynamically as needed:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cyro2formers_ctc ++model.pretraining_mode=False ++checkpoint=<checkpoint_path>
```

### CyRo2Formers Seq2Seq (Encoder-Decoder) Execution

#### Training with Pretrained Encoder
Load encoder weights from a CTC checkpoint and train the decoder end-to-end with differential learning rates:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cyro2formers_seq2seq \
  ++model.pretrained_encoder_ckpt=<ctc_checkpoint_path>
```

#### Training from Scratch (No Pretrained Encoder)
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cyro2formers_seq2seq
```

### Evaluation (Testing Without Training)
To evaluate the CTC model over the test set:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train model=cyro2formers_ctc ++train=false ++checkpoint=<checkpoint_path>
```

To evaluate the Seq2Seq model:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train model=cyro2formers_seq2seq ++train=false ++checkpoint=<checkpoint_path>
```

---

## License

`emg2qwerty` is CC-BY-NC-4.0 licensed, as documented in the `LICENSE` file.

## Citing emg2qwerty

```
@misc{sivakumar2024emg2qwertylargedatasetbaselines,
      title={emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
      author={Viswanath Sivakumar and Jeffrey Seely and Alan Du and Sean R Bittner and Adam Berenzweig and Anuoluwapo Bolarinwa and Alexandre Gramfort and Michael I Mandel},
      year={2024},
      eprint={2410.20081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20081},
}
```
