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

From here, we can see that the Our Transformer model with pretraining and fine-tuning outperforms the baseline CNN model and baseline Transformer model. We will follow some other researchers' idea to tune the hidden dimension and layer depth to further improve the performance. Specifically, we increase the FFN hidden dimension from 1024 to 2048 and the number of layers from 2 to 3.

### Pretrain Results

checkpoint file: logs/2026-02-27/22-41-30/checkpoints/epoch_118-step_14280.ckpt

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

There is improvement in pretraining. I will propose the following changes:

1. Decrease the value of K. A single user typing on a QWERTY keyboard is essentially producing combinations of ~30-40 distinct character intents. While the biomechanical transitions between keys create more unique states (co-articulations), 500 clusters might be forcing the K-means algorithm to separate data points based on random physiological noise, sensor shift, or minor velocity differences rather than meaningful gesture differences. If the pre-training task is forcing the model to predict structural noise, it limits fine-tuning performance.
2. Back Transformer layers to 2, we prefer simple models.Paper suggest widen Feed Forward Network to 2048 and shallow layer to 1, I would also apply this to refinement head.
3. I increase the dropout to control the widen feedforward layer.
4. Try to use span masking instead of random masking, with length of 12.

1 raw sample = 0.5 ms
1 spectrogram frame with hope 16 = 16 _ 0.5 ms = 8 ms
mask length = 12 frames = 12 _ 8 ms = 96 ms

The input data is 2kHz, with hope 16 = 125 frames/second
The data window length is 4 seconds = 500 frames

## Results

### Pretrain

logs/2026-03-01/00-03-16/checkpoints/epoch_122-step_14760.ckpt

### Validate

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 14.820558547973633 |
| val/DER      | 1.5728843212127686 |
| val/IER      | 5.737704753875732  |
| val/SER      | 7.5099687576293945 |
| val/ctc_loss | 0.5112477540969849 |
| val/loss     | 0.5112477540969849 |

### Test

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 17.10697364807129  |
| test/DER      | 1.775660514831543  |
| test/IER      | 6.561282157897949  |
| test/SER      | 8.77003002166748   |
| test/ctc_loss | 0.5857426524162292 |
| test/loss     | 0.5857426524162292 |

It seems that adding all these tricks does not help much.

Further improvement:

1. Back the refinement head to original settings. The paper suggests widen FFN to 2048 and shallow layer to 1 does not help in my architecture, probably because the I use refinement head as decoder. By squashing your Refinement Head to 1 layer but exploding its width to 2048, you created a shallow "memorization trap."
2. Slightly increase the dropout. Training and validation accuracy is higher than test accuracy, which means the model is overfitting.

Interesting finding here:
The loss osciliate at early epochs and stops learning. There seems to be a mismatch between either lower the ffn hidden dimension and increase the dropout, or increase the model layer and increase the dropout.

## Results:

### Pretrain

Checkpoint: logs/2026-03-01/22-57-02/checkpoints/epoch=128-step=15480.ckpt
| Metric | DataLoader 0 |
| ------------ | ------------------ |
| val/CER | 14.28887939453125 |
| val/DER | 2.259636640548706 |
| val/IER | 4.142667293548584 |
| val/SER | 7.886575222015381 |
| val/ctc_loss | 0.5020869374275208 |
| val/loss | 0.5020869374275208 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 16.349069595336914 |
| test/DER      | 2.230402708053589  |
| test/IER      | 4.395842552185059  |
| test/SER      | 9.722824096679688  |
| test/ctc_loss | 0.5769004821777344 |
| test/loss     | 0.5769004821777344 |

### Results

The following results are using 2 layer backbone with 2048 ffn hidden layer and 0.15 dropout. and 2 layer refinement head with 1024 ffn hidden layer and 0.10 dropout. Additionally, span masking with length 16 and 500 clusters is applied.

Span_masking + wide ffn theoretically works better than random masking + moderate ffn. However, the performance is still not as good as the 3 layer random masking architecture, with a little 0.3 in test CER.

## Pretrain

Checkpoint: logs/2026-03-03/01-22-14/checkpoints/epoch_123-step_14880.ckpt

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 14.665485382080078 |
| val/DER      | 2.4590163230895996 |
| val/IER      | 3.65529465675354   |
| val/SER      | 8.55117416381836   |
| val/ctc_loss | 0.5199381113052368 |
| val/loss     | 0.5199381113052368 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 15.829363822937012 |
| test/DER      | 2.4036378860473633 |
| test/IER      | 4.309224605560303  |
| test/SER      | 9.116500854492188  |
| test/ctc_loss | 0.5543370246887207 |
| test/loss     | 0.5543370246887207 |

Changes:

1. Increase the span mask length from 8 to 16
2. Increase the numnum_clusters from 250 to 500

## Results

### Pretrain

Checkpoint file: logs/2026-03-04/01-54-01/checkpoints/epoch=116-step=14040.ckpt

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 25.786441802978516 |
| val/DER      | 1.9273371696472168 |
| val/IER      | 12.339388847351074 |
| val/SER      | 11.519716262817383 |
| val/ctc_loss | 0.7541974782943726 |
| val/loss     | 0.7541974782943726 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 25.7470760345459   |
| test/DER      | 2.18709397315979   |
| test/IER      | 9.809441566467285  |
| test/SER      | 13.750541687011719 |
| test/ctc_loss | 0.7633078098297119 |
| test/loss     | 0.7633078098297119 |

1. dimensionality reduce from MultiBandElectrodeMixer to Transformer backbone, 768 -> 256
2. refinement head ffn hidden layer from 1024 -> 512
3. num_clusters from 500 -> 300
   The dimension reduction in ffn hidden layer from 1024 -> 512, backbone hidden dimension layer from 768 -> 256 worsen the performance.

## Result

### Pretrain

Checkpoint file: logs/2026-03-05/09-40-10/checkpoints/epoch=123-step=14880.ckpt

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 14.510412216186523 |
| val/DER      | 1.8830305337905884 |
| val/IER      | 3.4780681133270264 |
| val/SER      | 9.149312973022461  |
| val/ctc_loss | 0.5020440816879272 |
| val/loss     | 0.5020440816879272 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 16.262451171875    |
| test/DER      | 2.381983518600464  |
| test/IER      | 3.8328280448913574 |
| test/SER      | 10.047639846801758 |
| test/ctc_loss | 0.5392957925796509 |
| test/loss     | 0.5392957925796509 |

## Fine-tune Stage

Pick the best pre-train model and perform fine-tune. The best pre-train model is logs/2026-02-27/22-41-30/checkpoints/epoch_118-step_14280.ckpt, which consists of 3 layer transformer backbone with 2048 hidden dimension and 0.15 dropout, and 2 layer refinement head with 1024 hidden dimension and 0.10 dropout. Additionally, random masking ratio of 0.3 and cluster=500 is applied.

### Results

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 14.155959129333496 |
| val/DER      | 2.0159504413604736 |
| val/IER      | 3.5002214908599854 |
| val/SER      | 8.639787673950195  |
| val/ctc_loss | 0.5205371975898743 |
| val/loss     | 0.5205371975898743 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 15.179731369018555 |
| test/DER      | 1.9272412061691284 |
| test/IER      | 3.9411001205444336 |
| test/SER      | 9.311389923095703  |
| test/ctc_loss | 0.5838025212287903 |
| test/loss     | 0.5838025212287903 |

The second top architecture is checkpoint file: logs/2026-03-03/01-22-14/checkpoints/epoch_123-step_14880.ckpt. It has test CER of 15.829363822937012, which is 0.173 worse than the best architecture,15.6561279296875. This architecture uses 2 layer transformer backbone with 2048 hidden dimension and 0.15 dropout, and 2 layer refinement head with 1024 hidden dimension and 0.10 dropout. Additionally, random masking ratio of 0.15 and cluster=500 is applied.

| Metric       | DataLoader 0       |
| ------------ | ------------------ |
| val/CER      | 13.956579208374023 |
| val/DER      | 2.1267168521881104 |
| val/IER      | 3.8768277168273926 |
| val/SER      | 7.9530348777771    |
| val/ctc_loss | 0.5138890147209167 |
| val/loss     | 0.5138890147209167 |

| Metric        | DataLoader 0       |
| ------------- | ------------------ |
| test/CER      | 16.045907974243164 |
| test/DER      | 2.2520570755004883 |
| test/IER      | 4.504114151000977  |
| test/SER      | 9.289735794067383  |
| test/ctc_loss | 0.5525649785995483 |
| test/loss     | 0.5525649785995483 |

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
