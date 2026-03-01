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
3. Self-supervised Pretraining: Use K-means to cluster the input data into 500 clusters, and mask 35% of the input data and training the model to predict the masked cluster labels.
4. Transformer: a tightly constrained 2-layer Pre-LN Transformer with 8 attention heads, a feed-forward dimension of 1024, and 0.15 dropout
5. Attention Refinement Head
6. Fine-tuning on the pretraining model. We use a smaller learning rate for fine-tuning and smaller epochs.
7. Decoding with a character-level 6-gram language model, same as original paper.

## How to Run

### Pretraining Setup

- You need to define the number of clusters $k$ and the output path for the kmeans model.

```shell
python -m scripts.generate_spectre_clusters --k <k> --out <output_path>
```

### Pretraining

- Set the `model.pretraining_mode` to `True` in `config/model/cyro2formers_ctc.yaml`.

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cyro2formers_ctc ++model.pretraining_mode=True
```

### Fine-tuning

- Set the `model.pretraining_mode` to `False` in `config/model/cyro2formers_ctc.yaml`.
- Adjust the learning rate and number of epochs.

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train user=single_user model=cyro2formers_ctc ++model.pretraining_mode=False ++checkpoint=<checkpoint_path>
```

## Results

### Transformer + CyRoPE + MultiBandElectrodeMixer

checkpoint file location: logs/2026-02-26/01-30-51/checkpoints/epoch=134-step=16200.ckpt

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

### Transformer + CyRoPE + MultiBandElectrodeMixer + Pretrain

#### Pretrain

checkpoint file location: logs/2026-02-27/01-51-49/checkpoints/epoch_88-step_10680.ckpt

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

checkpoint file location: logs/2026-02-27/10-07-07/checkpoints/epoch=57-step=6960.ckpt

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

From here, we can see that the Our Transformer model with pretraining and fine-tuning outperforms the baseline CNN model and baseline Transformer model. We will follow some other researchers' idea to tune the hidden dimension and layer depth to further improve the performance. Specifically, we increase the FFN hidden dimension from 1024 to 2048 and the number of layers from 2 to 3.

### Pretrain Results

| Metric   | DataLoader 0       |
| -------- | ------------------ |
| val/CER  | 14.31103229522705  |
| val/DER  | 2.6140894889831543 |
| val/IER  | 2.724855899810791  |
| val/SER  | 8.97198486328125   |
| val/loss | 0.5199993848800659 |

| Metric    | DataLoader 0       |
| --------- | ------------------ |
| test/CER  | 15.6561279296875   |
| test/DER  | 2.7284538745880127 |
| test/IER  | 3.161541700363159  |
| test/SER  | 9.766132354736328  |
| test/loss | 0.5835206508636475 |

### Fine-tune Restuls:

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 14.133806228637695 |
| val/DER      | 1.8387240171432495 |
| val/IER      | 3.65529465675354   |
| val/SER      | 8.639787673950195  |
| val/ctc_loss | 0.5220286250114441 |
| val/loss     | 0.5220286250114441 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 15.461238861083984 |
| test/DER      | 1.8839324712753296 |
| test/IER      | 4.37418794631958   |
| test/SER      | 9.203118324279785  |
| test/ctc_loss | 0.5856450200080872 |
| test/loss     | 0.5856450200080872 |

## Verify any checkpoint results:

For testing without training, use the following command:

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m emg2qwerty.train model=cyro2formers_ctc ++train=false ++checkpoint=<checkpoint_path>
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
