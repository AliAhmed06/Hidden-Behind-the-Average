# Cross-Resource Conformal Calibration Audit for Fetal Ultrasound Plane Classification

Code accompanying the paper *"Hidden Behind the Average: Class- and Country-Conditional Conformal Calibration Reveals Coverage Failures Masked by Marginal Guarantees in Cross-Resource Fetal Ultrasound Plane Classification."*

This repository contains the full pipeline used to train a fetal ultrasound standard-plane classifier on a high-resource clinical dataset (Spain) and audit its reliability, using split conformal prediction, on a five-country low-resource cohort (Algeria, Egypt, Ghana, Malawi, Uganda), with no target-country labels used at any stage of training.

## What this code does

1. Loads and harmonises two public fetal ultrasound datasets into a shared four-class label space (abdomen, brain, femur, thorax).
2. Trains an EfficientNet-B0 classifier (ImageNet-pretrained) on the Spanish source dataset only, across three independent random seeds.
3. Applies temperature scaling for post-hoc calibration.
4. Applies the trained, frozen model zero-shot to the five-country target cohort.
5. Runs split conformal prediction under two regimes, pooled (marginal) and class-conditional (Mondrian), across 50 repeated random calibration/test splits per seed (150 total repetitions).
6. Reports coverage, prediction-set size, and expected calibration error, disaggregated by country, by anatomical class, and by seed, with bootstrap confidence intervals.

## Datasets

Both datasets are public and must be downloaded separately; they are not included in this repository.

- **Source domain (Spain):** Burgos-Artizzu et al., *Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes*, Scientific Reports, 2020. Available at [Zenodo record 3904280](https://zenodo.org/records/3904280).
- **Target domain (five countries):** Sendra-Balcells et al., *Generalisability of fetal ultrasound deep learning models to low-resource imaging settings in five African countries*, Scientific Reports, 2023. Available at [Zenodo record 7540448](https://zenodo.org/records/7540448).

Please cite both original papers if you use these datasets.

## Requirements

```
python >= 3.9
torch
torchvision
numpy
pandas
scikit-learn
Pillow
matplotlib
```

Install with:

```bash
pip install torch torchvision numpy pandas scikit-learn Pillow matplotlib
```

A single consumer-grade GPU (e.g. a free-tier Kaggle or Colab T4) is sufficient. The full run (3 seeds x 12 epochs, plus 150 conformal repetitions) takes under an hour.

## How to run

1. Download both datasets from the Zenodo links above.
2. Edit the two path variables at the top of `train_and_evaluate.py` to point to your local copies:

```python
SPAIN_ROOT  = "path/to/fetal-planes-zenodo"
AFRICA_ROOT = "path/to/five-country-zenodo"
```

3. Run the script:

```bash
python train_and_evaluate.py
```

If you are running this on Kaggle, enable Internet and GPU in the notebook settings, and attach both datasets before running.

## Outputs

Running the script produces, in the working directory:

- `raw_conformal_results.csv` — coverage and set-size results for every seed x repetition x country x class combination
- `coverage_summary_by_group.csv` — aggregated coverage statistics with bootstrap confidence intervals
- `coverage_by_group.png` — bar chart comparing pooled vs. class-conditional coverage
- `model_seed0.pt`, `model_seed1.pt`, `model_seed2.pt` — trained model weights for each seed

These are the exact files used to produce Tables 2 to 4 and Figures 1 and 2 in the paper.

## Reproducibility notes

- All three random seeds (0, 1, 2) are fixed in the script. Minor numerical differences between runs are possible due to non-deterministic GPU operations in PyTorch even with fixed seeds; this is a known limitation discussed in the paper.
- No target-country (five-country cohort) labels are used at any point during training or model selection. The five-country cohort is used exclusively for post-hoc calibration and evaluation.

## Citation

If you use this code, please cite the paper:

```
[Citation to be added once the paper is published or available as a preprint]
```

## License

This code is released under the MIT License. See `LICENSE` for details. The datasets themselves are governed by the licenses of their original publishers (Zenodo records linked above) and are not redistributed here.

## Contact

Questions or issues can be raised via the GitHub Issues tab on this repository.
