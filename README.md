# Joint Classification of Emotional Tone and Hateful Content in Turkish Speech

Hate speech detection usually relies on text alone, but in speech *how* something is said matters as much as *what* is said. This project classifies Turkish speech along two axes at once and compares acoustic and text-based models:

- **Tone:** aggressive · neutral · happy
- **Content:** hateful · neutral · positive

## Dataset

- **360 recordings:** 5 speakers × 24 sentences × 3 tones
- **120-sentence text set** (40 hateful, 40 neutral, 40 positive)
- **Split:** 288 train / 72 test, grouped by speaker × sentence to avoid leakage
- **Augmentation (deep models only):** 10 noise and pitch-shift variants per training recording (3,168 samples); the test set is kept raw

> The recordings contain volunteers' voices and are not included. They are available from the authors on request.

## Models

| Type | Models | Input |
|---|---|---|
| Classical (audio) | SVM (RBF), Gaussian Naive Bayes, kNN | Handcrafted acoustic features |
| Deep, multi-task (audio) | 1D-CNN, BiGRU, BiLSTM | Log-Mel spectrogram + 18-dim prosodic vector |
| Text | BERTurk, TF-IDF + linear SVM | Sentence text |

The deep models predict tone and content jointly (loss weights 0.1 / 0.9) and were each trained 5 times.

## Results

| Model | Tone acc. (%) | Content acc. (%) |
|---|---|---|
| SVM | **88.89** | 61.11 |
| Gaussian NB | 83.33 | 55.56 |
| kNN | 79.17 | 58.33 |
| 1D-CNN | 75.00 ± 7.54 | **63.33 ± 5.25** |
| BiGRU | 75.28 ± 11.60 | 55.83 ± 2.48 |
| BiLSTM | 66.11 ± 10.79 | 54.17 ± 3.26 |

| Text model | Accuracy (%) | Macro F1 |
|---|---|---|
| BERTurk (5-fold CV) | **78.3 ± 6.1** | **0.766 ± 0.077** |
| TF-IDF + linear SVM | 60.0 | 0.578 |

**Takeaway:** Acoustic features capture tone well but hateful content only weakly; text models capture content much better, which motivates multimodal text–audio fusion.

## Usage

```bash
pip install librosa numpy pandas scikit-learn imbalanced-learn tensorflow torch transformers emoji matplotlib seaborn
```

Run the notebooks in `notebooks/` (classical, deep and text models). A GPU is recommended for BERTurk.

## Limitations

The dataset is small and read in a controlled setting. For the deep models, the test set was also used to monitor early stopping, so their accuracies may be slightly optimistic.

## Authors

Elif Avcı · Simge Buse Kaplan · Tesnim Strazimiri · Emin Borandağ · Bahadır Karasulu · Sibel Dinç Aydemir

## License

Code: MIT. The speech recordings are not covered by this license.
