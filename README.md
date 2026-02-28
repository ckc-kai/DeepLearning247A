# C147/247 Final Project

### Winter 2026

#### Overview

This course project from ckc branch. The first section of this README provides some proposed ideas, experiemental results, and further investigation. The second section is about how to edit and run my designed models.

### CNN Baseline Model

We use the architecture like this as out baseline model:

**input -> [Conv*2 -> Max Pool]*2 -> [Conv*3 -> Max Pool]\*3 -> Linear(512) -> Linear(128) -> output**

- All Conv modules include batchnorm, activation, and dropout, denote Conv.
- Max Pool is Max Pooling with kernel size 2 and stride 2.
- Gradient Exploding/Vanishing is encountered so set the `gradient_clip_val: 1.0` and `gradient_clip_algorithm: "norm"`.
- See more details in `config/model/cnn_ctc.yaml`

#### Results

##### Validation Metrics

| Metric   | DataLoader 0 |
| -------- | ------------ |
| val/CER  | 85.27        |
| val/DER  | 0.399        |
| val/IER  | 71.53        |
| val/SER  | 13.34        |
| val/loss | 0.623        |

##### Test Metrics

| Metric    | DataLoader 0 |
| --------- | ------------ |
| test/CER  | 76.46        |
| test/DER  | 0.0216       |
| test/IER  | 63.11        |
| test/SER  | 13.33        |
| test/loss | 0.0          |

The results indicate that the CNN baseline model is not well suited for this task. The Character Error Rate (CER) remains extremely high on both validation (85.27%) and test (76.46%) sets, meaning most predicted characters are incorrect. The Insertion Error Rate (IER) is also very large (over 60%), suggesting the model frequently inserts incorrect characters, while the Deletion Error Rate (DER) is near zero on the test set, implying the model rarely deletes but instead predicts incorrect or redundant outputs. This pattern is consistent with a model that lacks sufficient temporal context to correctly align EMG signal patterns with character sequences. Because the CNN uses only stride-1 convolutions, its receptive field covers only a short temporal window $1+13⋅4=53$ receptive fields. This is equivalent to $53/2000=0.0265$, 26.5 ms sippet of time within 4 seconds time window. As a result, the model cannot capture the full structure of each gesture and struggles to form stable character predictions, leading to high error rates despite low loss values. This is an architectural error-the model’s temporal receptive field is too small to model long-duration dependencies required for accurate sequence transcription-and we will investigate more models to improve the performance.

## How to Run

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cnn_ctc
```

## Transformer Model

1. CyRoPE Positional Encoding on time + electrode position: learning consistent spatial relations, used in both MultibandElectrodeMixer and Transformer.
2. MultiBandElectrodeMixer: Learning global spatial relationships with 4 attention heads with CyRope
3. Transformer: a tightly constrained 2-layer Pre-LN Transformer with 8 attention heads, a feed-forward dimension of 1024, and 0.15 dropout
4. Spectral Branch (currently set optional): parallel to raw waveform path, compute time-frequency representation and encode with a lightweight temporal encoder.
5. Attention Refinement Head: fuse two branches and polish step
6. Decoding with a character-level 6-gram language model, same as original paper.

## Results

### Transformer + CyRoPE + MultiBandElectrodeMixer

| Metric   | DataLoader 0       |
| -------- | ------------------ |
| val/CER  | 18.07709312438965  |
| val/DER  | 2.326096534729004  |
| val/IER  | 6.357997417449951  |
| val/SER  | 9.392999649047852  |
| val/loss | 0.6290462613105774 |

| Metric    | DataLoader 0       |
| --------- | ------------------ |
| test/CER  | 19.423992156982422 |
| test/DER  | 3.2698137760162354 |
| test/IER  | 4.785621643066406  |
| test/SER  | 11.368557929992676 |
| test/loss | 0.7223228812217712 |

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train model=cyro2formers_ctc ++train=false ++checkpoint=./logs/2026-02-26/01-30-51/checkpoints/last.ckpt
```

### Transformer + CyRoPE + MultiBandElectrodeMixer + Pretrain

#### Pretrain

| Metric   | DataLoader 0       |
| -------- | ------------------ |
| val/CER  | 15.573770523071289 |
| val/DER  | 1.9273371696472168 |
| val/IER  | 5.405405521392822  |
| val/SER  | 8.24102783203125   |
| val/loss | 0.5256414413452148 |

| Metric    | DataLoader 0       |
| --------- | ------------------ |
| test/CER  | 17.410133361816406 |
| test/DER  | 2.2520570755004883 |
| test/IER  | 5.391944408416748  |
| test/SER  | 9.766132354736328  |
| test/loss | 0.5921993851661682 |

#### Fine-tune

| Metric   | DataLoader 0       |
| -------- | ------------------ |
| val/CER  | 14.421798706054688 |
| val/DER  | 1.7501107454299927 |
| val/IER  | 4.674346446990967  |
| val/SER  | 7.997341632843018  |
| val/loss | 0.5664355754852295 |

| Metric    | DataLoader 0       |
| --------- | ------------------ |
| test/CER  | 16.370723724365234 |
| test/DER  | 2.3603291511535645 |
| test/IER  | 4.850584506988525  |
| test/SER  | 9.159809112548828  |
| test/loss | 0.6432779431343079 |

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train model=cyro2formers_ctc ++model.pretraining_mode=False ++checkpoint=/Users/kaichengchu/Desktop/ucla/247/DeepLearning247A/logs/2026-02-27/01-51-49/checkpoints/epoch_88-step_10680.ckpt
```

## License

emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

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
