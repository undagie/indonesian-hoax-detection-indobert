# Indonesian Hoax Detection: Data-Centric Pipeline

Reproducibility code for the published study on Indonesian news hoax detection
with IndoBERT and a hybrid class-imbalance objective (class weighting + focal
loss).

**Published article.** Muttaqin, R., Rosadi, M. E., Firdaus, M. I., and
Agustini, D. (2026). Deteksi Hoaks Berita Berbahasa Indonesia Menggunakan
IndoBERT dengan Penanganan Ketidakseimbangan Kelas Berbasis Gabungan Class
Weighting dan Focal Loss. *Infotek: Jurnal Informatika dan Teknologi*, 9(2),
786-797. https://doi.org/10.29408/jit.v9i2.35214

This repository contains the experiment pipeline (code, configs, aggregated
result tables). The journal PDF is archived separately.

## How to cite

If you use this code or build on the method, please cite the journal article.

**APA**

```text
Muttaqin, R., Rosadi, M. E., Firdaus, M. I., & Agustini, D. (2026). Deteksi
hoaks berita berbahasa Indonesia menggunakan IndoBERT dengan penanganan
ketidakseimbangan kelas berbasis gabungan class weighting dan focal loss.
Infotek: Jurnal Informatika dan Teknologi, 9(2), 786-797.
https://doi.org/10.29408/jit.v9i2.35214
```

**BibTeX**

```bibtex
@article{muttaqin2026indobert,
  title   = {Deteksi Hoaks Berita Berbahasa Indonesia Menggunakan IndoBERT
             dengan Penanganan Ketidakseimbangan Kelas Berbasis Gabungan
             Class Weighting dan Focal Loss},
  author  = {Muttaqin, Riadhul and Rosadi, Muhammad Edya and
             Firdaus, Muhammad Iqbal and Agustini, Dian},
  journal = {Infotek: Jurnal Informatika dan Teknologi},
  volume  = {9},
  number  = {2},
  pages   = {786--797},
  year    = {2026},
  doi     = {10.29408/jit.v9i2.35214},
  url     = {https://doi.org/10.29408/jit.v9i2.35214}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).

## Abstract

The spread of hoaxes in Indonesian-language digital media has risen sharply in
the past five years with wide societal harm. Most prior work on Indonesian
hoax detection emphasizes model architecture, while the class-imbalance problem
in field data receives less attention. This study presents a data-centric
approach combining the IndoBERT pretrained language model with a hybrid
class-imbalance objective of class weighting and focal loss. The dataset is the
indonesiafalsenews corpus of 4231 labelled articles with an approximate 4.5 to 1
hoax-to-fact ratio. Evaluation uses five-seed runs with McNemar and
paired-bootstrap significance testing. The proposed configuration achieves an
F1-macro of 0.7240, accuracy of 0.8392, ROC-AUC of 0.8083, and a Matthews
correlation coefficient of 0.4510, significantly outperforming every TF-IDF
baseline at the 0.05 level. Text augmentation does not provide consistent
gains, which implies that imbalance handling is the more effective lever for
Indonesian hoax detection.

Keywords: class imbalance, focal loss, hoax detection, IndoBERT, natural
language processing

## Key results

Proposed model: IndoBERT-base (`indolem/indobert-base-uncased`) with
`imbalance=combined` (class-weighted focal loss) and no text augmentation.

| Model | F1-macro | Accuracy | MCC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Proposed (IndoBERT + combined) | 0.7240 +/- 0.0118 | 0.8392 | 0.4510 | 0.8083 |
| IndoBERT + contextual aug | 0.7164 +/- 0.0311 | 0.8364 | 0.4378 | 0.7986 |
| IndoBERT (vanilla) | 0.7124 +/- 0.0133 | 0.8449 | 0.4323 | 0.7936 |
| TF-IDF + Logistic Regression | 0.6878 +/- 0.0135 | 0.8118 | 0.3764 | 0.7836 |
| TF-IDF + Linear SVM | 0.6708 +/- 0.0107 | 0.8210 | 0.3471 | 0.7660 |
| TF-IDF + XGBoost | 0.6695 +/- 0.0073 | 0.8478 | 0.3865 | 0.7462 |
| TF-IDF + Random Forest | 0.5897 +/- 0.0230 | 0.8364 | 0.2879 | 0.7624 |

Means +/- std over seeds `[7, 42, 101, 1234, 1137]`. Aggregated tables are under
[`results/tables/`](results/tables/).

Main findings:

1. Hybrid class weighting and focal loss is the strongest imbalance strategy
   among the eight tested.
2. Text augmentation is not consistently helpful on this corpus; imbalance
   handling is the more effective lever.
3. Gains versus vanilla IndoBERT are not statistically significant; the
   practical benefit is higher fact-class recall and MCC under heavy imbalance.

## Method summary

| Component | Setting |
| --- | --- |
| Task | Binary classification: `hoax` vs `fakta` |
| Encoder | IndoBERT-base (IndoLEM), max length 384 |
| Proposed loss | Class-weighted focal loss with gamma = 2 |
| Split | Stratified 70% / 10% / 20% |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, warmup 10% |
| Early stopping | Validation F1-macro, patience 3 |
| Ablations | 7 augmentation methods, 8 imbalance strategies, 5 seeds |
| Significance | McNemar and paired bootstrap (1000 resamples) on F1-macro |

## Repository layout

```text
config.py            Hyperparameters (HOAX_* env overrides)
src/                 Data, augment, imbalance, models, training, eval, viz
experiments/         exp01-exp06, run_all, build_report
scripts/             Dataset download, server bootstrap, pack/unpack
results/tables/      Aggregated metrics (CSV, Markdown, LaTeX)
results/report.md    Master experiment summary
requirements.txt
```

Per-seed checkpoints (`best_model.pt`, about 420 MB each) are not included.
Re-run the experiments on a GPU to regenerate them.

## Setup

Python 3.10 or newer is recommended. A GPU with 12-16 GB VRAM (for example
RTX 3060 or T4) is needed for full Transformer runs.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch for your CUDA build first, for example:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On a Linux server without sudo:

```bash
bash scripts/setup_server.sh --python 3.11
source .venv/bin/activate
```

### Dataset

[Kaggle indonesiafalsenews](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews):
4231 labelled rows (3465 hoax / 766 fakta, about 4.5:1). Text is
`judul + " " + narasi`.

```powershell
# Credentials: %USERPROFILE%\.kaggle\kaggle.json  (or ~/.kaggle/kaggle.json)
python scripts/download_data.py
```

Do not commit `kaggle.json` into this repository.

## Running experiments

```powershell
# Full pipeline (about 6-10 h on a 12-16 GB GPU)
python -m experiments.run_all

# Smoke test
python -m experiments.run_all --quick

# Individual stages
python -m experiments.exp01_eda
python -m experiments.exp02_baselines
python -m experiments.exp03_augmentation_ablation
python -m experiments.exp04_imbalance_ablation
python -m experiments.exp05_full_comparison
python -m experiments.exp06_error_analysis
python -m experiments.build_report
```

Finished seeds with `test_metrics.json` are skipped on relaunch. Hyperparameters
can be overridden via environment variables defined in `config.py`
(for example `$env:HOAX_BATCH_SIZE=8`).

## Licence and contact

Article licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
(*Infotek: Jurnal Informatika dan Teknologi*, Universitas Hamzanwadi).

Authors: Riadhul Muttaqin, Muhammad Edya Rosadi, Muhammad Iqbal Firdaus,
Dian Agustini (Universitas Islam Kalimantan Muhammad Arsyad Al Banjari
Banjarmasin).

Article page:
https://e-journal.hamzanwadi.ac.id/index.php/infotek/article/view/35214
