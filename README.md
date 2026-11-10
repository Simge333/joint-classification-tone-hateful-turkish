# Joint Classification of Emotional Tone and Hateful Content in Turkish Speech

**A comparison of acoustic and text-based models for Turkish hate speech detection**

This repository contains the code and data for the study *"Joint Classification of Emotional Tone and Hateful Content in Turkish Speech: A Comparison of Acoustic and Text-Based Models"*.

## Table of contents

1. [Description](#description)
2. [Repository structure](#repository-structure)
3. [Dataset information](#dataset-information)
4. [Code information](#code-information)
5. [Requirements](#requirements)
6. [Usage instructions](#usage-instructions)
7. [Methodology](#methodology)
8. [Results](#results)
9. [Limitations](#limitations)
10. [Citations](#citations)
11. [License and contribution guidelines](#license-and-contribution-guidelines)
12. [Authors](#authors)

---

## Description

Hate speech detection is usually treated as a text classification problem. In speech, however, *how* something is said (tone, prosody) carries as much information as *what* is said. This project classifies Turkish speech along two axes at the same time:

| Task | Classes |
|---|---|
| **Tone** of the speaker | aggressive, neutral, happy |
| **Content** of the sentence | hateful, neutral, positive |

Six audio models are compared: three classical machine learning baselines (SVM, Gaussian Naive Bayes, kNN) and three multi-task deep learning models (1D-CNN, BiGRU, BiLSTM) that predict tone and content jointly. For the text side, a fine-tuned BERTurk model is compared with a TF-IDF + linear SVM baseline.

---

## Repository structure

```
joint-classification-tone-hateful-turkish/
├── README.md
└── joint-classification-tone-hateful-turkish/
    ├── labels.csv                  # metadata of the 360 audio recordings
    ├── sesler.rar                  # the 360 audio recordings (WAV), "sesler" = "recordings"
    ├── yuzYirmitext.xlsx           # 120-sentence text dataset
    ├── SVM/
    │   └── svm2.ipynb              # RBF-SVM baseline
    ├── Naive Bayes/
    │   └── complementNB1.ipynb     # Gaussian Naive Bayes baseline (two-stage)
    ├── KNN/
    │   └── knn.ipynb               # kNN baseline
    ├── CNN/
    │   └── CNN_deney1–5.ipynb      # multi-task 1D-CNN, runs 1–5 ("deney" = "experiment")
    ├── GRU/
    │   └── GRU_deney1–5.ipynb      # multi-task BiGRU, runs 1–5
    ├── LSTM/
    │   └── LSTM_deney1–5.ipynb     # multi-task BiLSTM, runs 1–5
    └── TURKBERT.ipynb              # BERTurk and TF-IDF text models
```

All notebooks are stored **with their outputs**, so the reported results and figures can be inspected without re-running them.

---

## Dataset information

The study uses three data sources.

### 1. Turkish Offensive Language Dataset (third-party)

- **Source:** Tanyel, T., Alkurdi, B., & Ayvaz, S. (2022). *Linguistic-based data augmentation approach for offensive language detection.* 7th International Conference on Computer Science and Engineering (UBMK), pp. 1–6. https://doi.org/10.1109/UBMK55850.2022.9919562
- **Repository:** https://github.com/toygarr/lingda
- **Use in this study:** Hateful and neutral sentences were selected from this corpus and re-annotated into the target categories.

### 2. Positive sentences from Çiçeksepeti.com (third-party)

- **Source:** Customer reviews on the Turkish e-commerce platform [çiçeksepeti.com](https://www.ciceksepeti.com)
- **Collection method:** Positive customer reviews were manually selected from publicly visible product review pages. They were added because the offensive language corpus contains few positive expressions.

### Text dataset: `yuzYirmitext.xlsx`

120 Turkish sentences ("yüz yirmi" = "one hundred twenty"), 40 per class.

| Column | Description |
|---|---|
| `text` | Turkish sentence |
| `label` | `0` = neutral, `1` = hateful, `2` = positive |
| `kategori` | Sentence index (1–120) |

### 3. Newly collected audio recordings (this study): `sesler.rar`

| Property | Value |
|---|---|
| Speakers | 5 adult volunteers, native Turkish speakers |
| Sentences | 24 per speaker (8 hateful, 8 neutral, 8 positive), taken from the text dataset |
| Tones | Each sentence read in 3 tones: aggressive, neutral, happy |
| Total recordings | **360** WAV files (5 speakers × 24 sentences × 3 tones) |
| Sampling rate used | 22,050 Hz, mono (resampled with librosa) |
| Annotation | Tone and content labels validated by three independent annotators |
| Ethics | Recordings were collected with the informed consent of all volunteers. Ethics approval: *[name of the ethics committee, approval number]* |

**Metadata: `labels.csv`**

| Column | Description |
|---|---|
| `name` | Audio file name inside `sesler.rar`; the prefix (`hateful…`, `neutral…`, `positive…`) is the name of the recording folder |
| `content` | Content label: `hateful`, `neutral`, `positive` |
| `tone` | Tone label: `aggresive`, `neutral`, `happy` (the spelling `aggresive` is used in the file) |
| `sentence_id` | Identifier of a speaker–sentence pair; its three tonal renditions share the same ID and are kept in the same train/test partition |

Each content–tone combination contains exactly 40 recordings.

---

## Code information

| Notebook | Model | Input | Output |
|---|---|---|---|
| `SVM/svm2.ipynb` | RBF-SVM; standardization, PCA (98% variance), grid search over C ∈ {100, 300, 500} and γ ∈ {10⁻⁴, 10⁻³} | 651-dim handcrafted acoustic features | Tone and content reports, confusion matrices, ROC curves |
| `Naive Bayes/complementNB1.ipynb` | Two-stage **Gaussian** Naive Bayes: tone model, then content model using out-of-fold tone probabilities; PCA, SMOTE/ADASYN oversampling, isotonic calibration, sentence-level majority vote | 422-dim acoustic features (425-dim for content) | Tone and content reports, confusion matrices |
| `KNN/knn.ipynb` | kNN, k ∈ {3, 5, 7, 9, 11} selected by 5-fold CV | 18-dim prosodic vector | Tone and content reports, confusion matrices |
| `CNN/CNN_deney1–5.ipynb` | Multi-task 1D-CNN (5 independent runs, identical code) | Mel-spectrogram (250 × 128) + 18-dim prosodic vector | Per-run tone and content results |
| `GRU/GRU_deney1–5.ipynb` | Multi-task BiGRU (5 runs) | same as above | same as above |
| `LSTM/LSTM_deney1–5.ipynb` | Multi-task BiLSTM (5 runs) | same as above | same as above |
| `TURKBERT.ipynb` | BERTurk (`savasy/bert-base-turkish-sentiment-cased`), stratified 5-fold CV; TF-IDF + linear SVM baseline | Sentence text | Per-fold and mean accuracy / macro F1, 5-fold confusion matrix |

The file name `complementNB1.ipynb` is historical; the reported model is **Gaussian** Naive Bayes.

---

## Requirements

- **Python** 3.12 (Google Colab)
- **GPU** recommended for the deep learning models and BERTurk

| Library | Version | Used for |
|---|---|---|
| librosa | *version* | Audio loading, Mel-spectrograms, MFCC, pitch shifting |
| numpy | *version* | Numerical operations |
| pandas | 2.2.2 | Data handling |
| scikit-learn | 1.6.1 | Classical models, PCA, grid search, metrics |
| imbalanced-learn | *version* | SMOTE, ADASYN |
| tensorflow (Keras) | *version* | 1D-CNN, BiGRU, BiLSTM |
| torch | 2.9.0 | BERTurk backend |
| transformers | *version* | BERTurk |
| emoji | 2.15.0 | Text cleaning |
| openpyxl | *version* | Reading `yuzYirmitext.xlsx` |
| matplotlib, seaborn | *version* | Figures |

```bash
pip install librosa numpy pandas scikit-learn imbalanced-learn tensorflow torch transformers emoji openpyxl matplotlib seaborn joblib
```

---

## Usage instructions

### 1. Get the code and data

```bash
git clone https://github.com/Simge333/joint-classification-tone-hateful-turkish.git
cd joint-classification-tone-hateful-turkish/joint-classification-tone-hateful-turkish
unrar x sesler.rar          # or extract with 7-Zip / WinRAR; creates the folder sesler/
```

### 2. Set the data paths

The notebooks were run in Google Colab with the data on Google Drive. Before running an audio notebook, edit the two path variables in its first cell:

```python
CSV_PATH  = "labels.csv"    # path to labels.csv
AUDIO_DIR = "sesler"        # folder with the extracted WAV files
```

- **Google Colab:** upload the repository folder to your Google Drive, keep the `drive.mount('/content/drive')` line and point both paths to that folder, e.g. `/content/drive/MyDrive/joint-classification-tone-hateful-turkish/labels.csv`.
- **Local machine:** delete the `drive.mount(...)` line and use the relative paths shown above.

### 3. Run the audio models

- **Classical baselines:** run `SVM/svm2.ipynb`, `Naive Bayes/complementNB1.ipynb` and `KNN/knn.ipynb`. Each trains one tone and one content model and prints the test results.
- **Deep learning models:** run the five notebooks of an architecture (e.g. `CNN_deney1` to `CNN_deney5`). Each run prints the tone and content results on the 72 test recordings; the paper reports the mean ± SD over the five runs.

### 4. Run the text models

In `TURKBERT.ipynb`, load the text dataset in the data-loading cell:

```python
df = pd.read_excel("yuzYirmitext.xlsx")
```

Select a GPU runtime and run all cells. The notebook prints the per-fold and mean accuracy and macro F1, the averaged 5-fold confusion matrix, and the TF-IDF baseline results.

---

## Methodology

### Preprocessing and features

1. Audio is loaded as mono at 22,050 Hz.
2. **Mel-spectrogram** (deep models): 128 Mel bands (f_max = 8,000 Hz), converted to dB, padded or truncated to 250 frames, giving a 250 × 128 input.
3. **18-dim prosodic vector** (deep models and kNN): mean of 13 MFCCs, RMS energy (mean, std), zero-crossing rate (mean, std) and tempo; standardized with statistics of the training split only.
4. **Handcrafted features** for SVM (651-dim) and GNB (422-dim): MFCC and delta statistics, spectral, chroma and Mel-spectrogram statistics; GNB also uses F0 statistics.

### Data split and augmentation

- The 360 recordings are split into **288 training and 72 test** recordings with `GroupShuffleSplit(test_size=0.2, random_state=42)`, grouped by `sentence_id`. All notebooks use the same split.
- **Deep models only:** each training recording receives 10 augmented copies: 5 with white noise (factor 0.002–0.010) and 5 with pitch shifting (−3, −2, −1, +1, +2 semitones), giving 3,168 training samples. The test set is never augmented.
- Classical models use the 288 original training recordings.

### Models and training

- **Multi-task deep models:** a 1D-CNN (64, 128, 256 filters, global average pooling) or a three-layer BiGRU/BiLSTM (128, 64, 32 units) encodes the Mel-spectrogram; the prosodic vector passes through a 32-unit dense layer. Both are concatenated, passed through shared dense layers (256, 128) and split into two softmax heads for tone and content.
- **Loss:** weighted sparse categorical cross-entropy (tone 0.1, content 0.9).
- **Training:** Adam (learning rate 5 × 10⁻⁴ for CNN and BiLSTM, 10⁻³ for BiGRU; batch size 64 for CNN, 32 for BiGRU and BiLSTM), up to 200 epochs, early stopping (patience 30, best weights restored), ReduceLROnPlateau (factor 0.5, patience 15).
- **Repetitions:** each deep architecture is trained five times with the same split; results are reported as mean ± SD.
- **BERTurk:** classification head re-initialized for three classes; stratified 5-fold CV; learning rate 2 × 10⁻⁵, batch size 8, 3 epochs, maximum sequence length 128; re-initialized from the pre-trained checkpoint in every fold.
- **TF-IDF baseline:** unigrams and bigrams (up to 10,000 features) with a linear SVM, 5-fold CV.

---

## Results

### Audio models (72-recording test set)

| Model | Tone accuracy (%) | Content accuracy (%) |
|---|---|---|
| SVM (RBF) | **88.89** | 61.11 |
| Gaussian Naive Bayes | 83.33 | 55.56 |
| kNN | 79.17 | 58.33 |
| 1D-CNN (5 runs) | 75.00 ± 7.54 | **63.33 ± 5.25** |
| BiGRU (5 runs) | 75.28 ± 11.60 | 55.83 ± 2.48 |
| BiLSTM (5 runs) | 66.11 ± 10.79 | 54.17 ± 3.26 |

### Text models (120 sentences, 5-fold CV)

| Model | Accuracy (%) | Macro F1 |
|---|---|---|
| BERTurk | **78.3 ± 6.1** | **0.766 ± 0.077** |
| TF-IDF + linear SVM | 60.0 | 0.578 |

Acoustic features capture tone well but hateful content only weakly; the text model captures content much better.

---

## Limitations

- The dataset is small (360 recordings, 5 speakers) and consists of read speech recorded in a controlled setting.
- For the deep models, no separate validation set was used: the test partition also served as the monitoring set for early stopping and learning-rate scheduling, so the deep-model accuracies may be slightly optimistic.
- The split groups recordings by speaker × sentence; the same sentence text read by a different speaker can occur in both partitions.

---

## Citations

Third-party data and models:

- Tanyel, T., Alkurdi, B., & Ayvaz, S. (2022). Linguistic-based data augmentation approach for offensive language detection. *UBMK 2022*, 1–6. https://doi.org/10.1109/UBMK55850.2022.9919562
- Schweter, S. (2020). BERTurk – BERT models for Turkish. *Zenodo*. https://doi.org/10.5281/zenodo.3770924
- McFee, B. et al. (2015). librosa: Audio and music signal analysis in Python. *Proc. SciPy*, 18–25.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

---

## License and contribution guidelines

- **Code:** MIT License.
- **Audio recordings (`sesler.rar`) and `labels.csv`:** provided for research purposes only; the recordings may not be used to identify the speakers or for commercial purposes.
- **Third-party text:** sentences from the Turkish Offensive Language Dataset and from Çiçeksepeti.com remain subject to the terms of their original sources.
- **Contributions:** bug reports and suggestions are welcome through GitHub Issues. For larger changes, please open an issue first to discuss the proposed change.

---

## Authors

Elif Avcı · Simge Buse Kaplan · Tesnim Strazimiri · Bahadır Karasulu · Sibel Dinç Aydemir · Emin Borandağ (corresponding author)
