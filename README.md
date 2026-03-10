# C147/247 Final Project — Conformer-CTC for EMG-to-Text
### Winter 2026

---

## Independent Study: Conformer-CTC Encoder (My Work)

> The sections below document my independent study contribution: replacing the baseline TDS convolution encoder with a **Conformer-CTC** architecture, training experiments, and beam search + language model decoding optimization.

### Final Results

| Metric | Greedy Decoding | Beam Search + LM |
|--------|-----------------|-------------------|
| Val CER | 29.07% | **18.72%** |
| Test CER | 48.52% | **21.05%** |

Best checkpoint: `logs/conformer_130epoch/run2/checkpoints/epoch=10-step=5280.ckpt`
Beam search config: `beam_size=50, lm_weight=1.5, insertion_bonus=1.5`

---

### 1. Model Architecture

#### 1.1 Conformer Encoder

The Conformer architecture combines Transformers (global context via self-attention) and CNNs (local feature extraction via depthwise convolution). Each Conformer block follows a "sandwich" structure:

```
FFN (half-step) -> Multi-Head Self-Attention -> Convolution -> FFN (half-step) -> LayerNorm
```

Key design choices:
- **Local attention window** (`window=32`): Limits self-attention to a local window, reducing memory from O(T^2) to O(T*W)
- **Sinusoidal positional encoding**: Position information without learnable parameters
- **Temporal subsampling** (`stride=2`): Reduces sequence length by 2x before Conformer blocks
- **Intermediate CTC loss** (`layer=3, weight=0.3`): Auxiliary CTC loss at an intermediate layer for regularization
- **Macaron-style FFN**: Two half-step feed-forward modules sandwiching the attention and convolution

#### 1.2 Model Configuration

| Parameter | Value |
|-----------|-------|
| Total parameters | ~5.5M |
| Frontend MLP | 528 -> 128 (band-level MLP) |
| Conformer layers | 6 |
| Attention heads | 4 |
| d_model | 128 |
| FFN expansion | 2x |
| Conv kernel size | 15 |
| Local attention window | 32 |
| Time reduction stride | 2 |
| Dropout | 0.1 |
| Precision | FP16 (mixed precision) |

#### 1.3 CTC Training Techniques

- **blank_logit_sub** (`=3.0`): Subtracts a constant from the blank token logit during training, preventing all-blank predictions. Annealed over 6 epochs.
- **Gradient clipping** (`max_norm=1.0`): Essential for training stability.

---

### 2. Training Experiments

#### 2.1 Training Setup

- **Batch size**: 8
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **LR scheduler**: Linear warmup (5 epochs) + cosine annealing
- **Precision**: FP16 (mixed precision)

#### 2.2 Experiment Summary

7+ training runs were conducted. A persistent pattern emerged: **training collapse at epoch 13-14**, where CER spikes from ~30% to 100% within 1-2 epochs.

| Experiment | Epochs | LR | Best Val CER | Collapse Epoch |
|------------|--------|-----|--------------|----------------|
| baseline_40ep | 40 | 1e-4 | ~35% | 13 |
| run1_50ep | 50 | 1e-4 | ~32% | 14 |
| **run2_130ep** | **130** | **1e-4** | **29.07% (ep 10)** | **13** |
| lowlr_100ep | 100 | 5e-5 | 64.67% | 13 |
| depth4 | 40 | 1e-4 | ~38% | 14 |
| depth8 | 40 | 1e-4 | ~33% | 13 |
| global_attn | 40 | 1e-4 | ~40% | 14 |

#### 2.3 Analysis of Training Collapse

The collapse pattern was consistent across all experiments:
1. **Epochs 1-10**: CER steadily decreases, model learns meaningful representations
2. **Epochs 11-12**: CER plateau, blank_logit_sub anneal has completed
3. **Epoch 13-14**: Sudden spike to CER=100%, model predicts near-blank outputs
4. **Epochs 14+**: No recovery

**Root cause**: The `blank_logit_sub` annealing completes at epoch 6. By epoch 13, the model's blank probability increases enough to trigger a phase transition. Once blank dominates, the CTC loss landscape becomes flat and the model cannot recover.

**Strategy**: Instead of solving the collapse, I used the **best pre-collapse checkpoint** (epoch 10, CER=29.07%) and improved results through decoding optimization.

---

### 3. Beam Search + Language Model Decoding

#### 3.1 Approach

The codebase includes a `CTCBeamDecoder` with KenLM language model integration (from the original authors). Instead of greedy CTC decoding, beam search explores multiple candidate sequences and uses a character-level 6-gram language model to rescore them.

- **Language model**: 6-gram character-level KenLM trained on WikiText-103 (`models/lm/wikitext-103-6gram-charlm.bin`, provided by original authors)
- **Decoder**: `CTCBeamDecoder` from `parlance/ctcdecode`

#### 3.2 Hyperparameter Tuning

Two key parameters: **lm_weight** (language model influence) and **insertion_bonus** (counteracts CTC's blank bias).

**Phase 1: Coarse Grid Search** (beam_size=20, faster)

| lm_weight | insertion_bonus | Val CER |
|-----------|----------------|---------|
| 0.5 | 1.0 | 24.18% |
| 0.5 | 1.5 | 23.87% |
| 1.0 | 1.0 | 21.19% |
| 1.0 | 1.5 | 19.49% |
| 1.5 | 1.0 | 20.84% |
| 1.5 | 1.5 | 19.59% |
| 2.0 | 1.5 | 20.62% |
| 2.0 | 2.0 | 20.51% |
| 2.5 | 2.0 | 22.19% |
| 3.0 | 2.0 | 24.82% |

**Phase 2: Fine-Grained Verification** (beam_size=50)

| lm_weight | insertion_bonus | Val CER |
|-----------|----------------|---------|
| 1.2 | 1.5 | 18.90% |
| **1.5** | **1.5** | **18.72%** |
| 2.0 | 1.5 | 20.20% |
| 2.0 | 2.0 | 19.83% |

#### 3.3 Impact of Beam Search + LM

| Decoding Method | Val CER | Relative Improvement |
|-----------------|---------|---------------------|
| Greedy (argmax) | 29.07% | — |
| Beam (default params) | 19.83% | -31.8% |
| Beam (tuned params) | **18.72%** | **-35.6%** |

The LM provides substantial improvement by correcting character-level errors using English character n-gram statistics. The 35.6% relative improvement demonstrates that the Conformer encoder learns meaningful representations even though raw CER appears high.

---

### 4. Final Evaluation

Using the best checkpoint (epoch 10):

| Split | Decoding | CER | DER | IER | SER |
|-------|----------|-----|-----|-----|-----|
| Validation | Greedy | 29.07% | 2.68% | 6.11% | 20.29% |
| Validation | Beam+LM | **18.72%** | 2.28% | 4.39% | 12.05% |
| Test | Greedy | 48.52% | 1.36% | 19.97% | 27.19% |
| Test | Beam+LM | **21.05%** | 2.75% | 4.79% | 13.51% |

- **CER**: Character Error Rate (overall edit distance / reference length)
- **DER**: Deletion Error Rate
- **IER**: Insertion Error Rate
- **SER**: Substitution Error Rate

---

### 5. Conclusions

1. **Conformer architecture** successfully learns EMG-to-text representations, achieving 29.07% greedy CER — attention + convolution captures both local muscle activation patterns and longer-range temporal dependencies.

2. **Training instability** remains a challenge: all experiments collapsed at epoch 13-14. Future work should explore gradient accumulation, longer warmup schedules, or progressive blank_logit_sub annealing.

3. **Beam search + LM** decoding is critical, reducing CER by 35.6% (29.07% -> 18.72%). The encoder captures sufficient information but benefits greatly from linguistic priors during decoding.

4. **Final performance**: Val CER = 18.72%, Test CER = 21.05%.

---

### 6. Files Modified / Added (My Changes)

| File | Status | Description |
|------|--------|-------------|
| `emg2qwerty/modules.py` | Modified | Added ConformerEncoder, ConformerBlock, ConformerFeedForward, ConformerConv, SinusoidalPositionalEncoding, TemporalSubsampling1d |
| `emg2qwerty/lightning.py` | Modified | Added ConformerCTCModule (LightningModule with training/eval/beam search logic) |
| `emg2qwerty/train.py` | Modified | Added resume checkpoint support, dataset size logging, trainer kwargs handling |
| `config/model/conformer_ctc.yaml` | New | Conformer model and datamodule configuration |
| `scripts/run_conformer_independent_study.py` | New | Systematic experiment runner for controlled ablations |
| `scripts/extract_metrics.py` | New | Utility for extracting TensorBoard metrics |

---

### 7. How to Run

#### Training the Conformer
```bash
python -m emg2qwerty.train \
    user=single_user \
    model=conformer_ctc \
    trainer.max_epochs=130 \
    batch_size=8 \
    optimizer.lr=1e-4 \
    +trainer.precision=16 \
    +trainer.gradient_clip_val=1.0 \
    +trainer.gradient_clip_algorithm=norm
```

#### Evaluation with Beam Search + LM (Best Config)
```bash
python -m emg2qwerty.train \
    user=single_user \
    model=conformer_ctc \
    train=false \
    checkpoint=logs/conformer_130epoch/run2/checkpoints/epoch\=10-step\=5280.ckpt \
    decoder=ctc_beam \
    decoder.beam_size=50 \
    decoder.lm_weight=1.5 \
    decoder.insertion_bonus=1.5
```

#### Running Controlled Ablation Studies
```bash
python scripts/run_conformer_independent_study.py \
    --phase screening \
    --group depth
```

---
---

## Course Information (From Instructor)

> The section below is provided by the course instructor.

This course project is built upon the emg2qwerty work from Meta. **Note that the rest of the README is from the original repo and we encourage you to take a look at their work.**

### Guiding Tips + FAQs
_Last updated 2/13/2025_
- Read through the Project Guidelines to ensure that you have a clear understanding of what we expect
- Familiarize yourself with the prediction task and get a high-level understanding of their base architecture (it would be beneficial to read about CTC loss)
- Get comfortable with the codebase
  - `lightning.py` + `modules.py` - where most of your model architecture development will take place
  - `data.py` - defines PyTorch dataset (likely will not need to touch this much)
  - `transforms.py` - implement more data transforms and other preprocessing techniques
  - `config/*.yaml` - modify model hyperparameters and PyTorch Lightning training configuration
    - **Q: How do we update these configuration files?** A: Note the structure of YAML files include basic key-value pairs (i.e. `<key>: <value>`) and hierarchical structure. So, for instance, if we wanted to update the `mlp_features` hyperparameter of the `TDSConvCTCModule`, we would change the value at line 5 of `config/model/tds_conv_ctc.yaml` (under `module`). _Read more details [here](https://pytorch-lightning.readthedocs.io/en/1.3.8/common/lightning_cli.html)._
    - **Q: Where do we configure data splitting?** A: Refer to `config/user/single_user.yaml`. Be careful with your edits, so that you don't accidentally move the test data into your training set.

---
---

## Original README (From Meta / emg2qwerty Authors)

> Everything below is from the [original emg2qwerty repository](https://arxiv.org/abs/2410.20081) by Meta.

# emg2qwerty
[ [`Paper`](https://arxiv.org/abs/2410.20081) ] [ [`Dataset`](https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz) ] [ [`Blog`](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/) ] [ [`BibTeX`](#citing-emg2qwerty) ]

A dataset of surface electromyography (sEMG) recordings while touch typing on a QWERTY keyboard with ground-truth, benchmarks and baselines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/71a9f361-7685-4188-83c3-099a009b6b81" height="80%" width="80%" alt="alt="sEMG recording" >
</p>

### Setup

```shell
# Install git-lfs (for pretrained checkpoints)
git lfs install

# Clone the repo, setup environment, and install local package
git clone git@github.com:joe-lin-tech/emg2qwerty.git ~/emg2qwerty
cd ~/emg2qwerty
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .

# Download the dataset, extract, and symlink to ~/emg2qwerty/data
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

### Data

The dataset consists of 1,136 files in total - 1,135 session files spanning 108 users and 346 hours of recording, and one `metadata.csv` file. Each session file is in a simple HDF5 format and includes the left and right sEMG signal data, prompted text, keylogger ground-truth, and their corresponding timestamps. `emg2qwerty.data.EMGSessionData` offers a programmatic read-only interface into the HDF5 session files.

To load the `metadata.csv` file and print dataset statistics,

```shell
python scripts/print_dataset_stats.py
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012947-66cab4c4-963c-4f1a-af12-47fea1681f09.png" alt="Dataset statistics" height="50%" width="50%">
</p>

To re-generate data splits,

```shell
python scripts/generate_splits.py
```

The following figure visualizes the dataset splits for training, validation and testing of generic and personalized user models. Refer to the paper for details of the benchmark setup and data splits.

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012465-504eccbf-8eac-4432-b8aa-0e453ad85b49.png" alt="Data splits">
</p>

To re-format data in [EEG BIDS format](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html),

```shell
python scripts/convert_to_bids.py
```

### Training (Original Baseline)

Generic user model:

```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=8 \
  --multirun
```

Personalized user models:

```shell
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1
```

If you are using a Slurm cluster, include "cluster=slurm" override in the argument list of above commands to pick up `config/cluster/slurm.yaml`. This overrides the Hydra Launcher to use [Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher). Refer to Hydra documentation for the list of available launcher plugins if you are not using a Slurm cluster.

### Testing (Original Baseline)

Greedy decoding:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Beam-search decoding with 6-gram character-level language model:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

The 6-gram character-level language model, used by the first-pass beam-search decoder above, is generated from [WikiText-103 raw dataset](https://huggingface.co/datasets/wikitext), and built using [KenLM](https://github.com/kpu/kenlm). The LM is available under `models/lm/`, both in the binary format, and the human-readable [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/). These can be regenerated as follows:

1. Build kenlm from source: <https://github.com/kpu/kenlm#compiling>
2. Run `./scripts/lm/build_char_lm.sh <ngram_order>`

### License

emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

### Citing emg2qwerty

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
