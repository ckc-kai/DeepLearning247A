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
