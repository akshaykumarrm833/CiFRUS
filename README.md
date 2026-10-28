# COMP8240 Project — Reproduction and Generalisability of CiFRUS

This repository contains the code, data preparation scripts and experimental results for my COMP8240 Applications of Data Science project:

**Evaluating the Reproducibility and Generalisability of CiFRUS for Tabular Classification**

The project reproduces part of the experiments from:

Saimon, S. H., Najnin, T., & Ruan, J. (2024). *A Novel Feature Space Augmentation Method to Improve Classification Performance and Evaluation Reliability*. Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2512–2523. https://doi.org/10.1145/3637528.3671736

This repository is a fork of the authors' official CiFRUS implementation.

Original repository:  
https://github.com/sakhawatsaimon/CiFRUS

---

## Project Overview

The project has three experimental components:

1. **Targeted reproduction of original work**
   - Original dataset: `breast-cancer-wisc-diag`
   - Classifier: Gaussian Naive Bayes
   - 10-fold stratified cross-validation repeated 10 times
   - Baseline, conventional resampling methods and three CiFRUS configurations
   - Five evaluation metrics: AUC, F1-score, Kappa, balanced accuracy and MCC

2. **Evaluation on a new existing dataset**
   - UCI Rice Cammeo and Osmancik dataset
   - 3,810 observations
   - 7 numerical features
   - Gaussian Naive Bayes
   - Same repeated stratified evaluation framework used for the targeted reproduction

3. **Evaluation on a newly constructed dataset**
   - Australian hourly historical weather data obtained from Open-Meteo
   - Sydney, Melbourne, Brisbane, Adelaide and Perth
   - 2023–2025
   - 131,400 observations
   - Target: whether at least 1 mm of precipitation occurs within the following 24 hourly observations
   - Chronological evaluation is used instead of random cross-validation to prevent future information from entering the training process

---

## Environment

The project experiments were run using:

- Python 3.12.6
- NumPy 2.5.4
- SciPy 1.18.1
- pandas 3.1.0
- scikit-learn 1.9.1
- imbalanced-learn 0.14.2
- imbalanced-learn-extra 0.2.10
- requests 2.34.2
- ucimlrepo 0.0.7

### Installation

From the repository root:

```bash
python3 -m venv .venvCIFRUS
source .venvCIFRUS/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

---

## Repository Structure

### Original CiFRUS implementation

The original source code and benchmark result files from the authors are retained in the repository.

Important original material includes:

```text
cifrus/
results/performance/
cross_validate.py
```

### Project scripts

```text
reproduce_subset.py
```

Runs the targeted original-data reproduction using the breast cancer diagnostic dataset and Gaussian Naive Bayes.

```text
compare_reproduction.py
```

Compares the reproduced repeat-averaged metrics with the authors' stored results.

```text
prepare_rice_dataset.py
```

Downloads and prepares the UCI Rice dataset.

```text
evaluate_rice.py
```

Evaluates the baseline, conventional resampling methods and the three CiFRUS configurations on the Rice dataset.

```text
build_weather_dataset_final.py
```

Constructs the final Australian weather dataset from Open-Meteo historical hourly data and generates the future-rain target.

```text
validate_weather_target.py
```

Recalculates and checks the future 24-hour precipitation target on eligible observations.

```text
evaluate_weather.py
```

Performs the leakage-aware chronological weather evaluation.

```text
compare_all_datasets.py
```

Produces a cross-dataset comparison of the targeted reproduction, Rice experiment and weather experiment.

---

## Data

### Rice

```text
data_new/rice/rice_cammeo_osmancik.csv
```

The prepared dataset contains:

- 3,810 rows
- 7 numerical morphological features
- two classes: Cammeo and Osmancik
- no missing values
- no duplicate rows

### Australian Weather

Final dataset:

```text
data_new/weather/australian_weather_rain_final.csv
```

The dataset contains:

- 131,400 hourly observations
- five Australian cities
- 26,280 observations per city
- data from January 2023 to December 2025
- no missing values
- no duplicate rows

The prediction target is:

```text
rain_next_24h
```

It is equal to 1 when at least 1 mm of precipitation occurs during the following 24 hourly observations and 0 otherwise.

The column:

```text
future_24h_precipitation
```

is retained only for verification of target construction and is **not used as a predictor**.

The predictors used in the weather experiment are:

```text
temperature_2m
relative_humidity_2m
pressure_msl
cloud_cover
wind_speed_10m
precipitation
hour
month
```

---

## Experimental Protocols

### Original reproduction

Run:

```bash
python reproduce_subset.py
python compare_reproduction.py
```

The targeted experiment uses:

- `breast-cancer-wisc-diag`
- Gaussian Naive Bayes
- 10-fold stratified cross-validation
- 10 repetitions
- 11 augmentation/baseline configurations
- AUC, F1-score, Kappa, balanced accuracy and MCC

Outputs:

```text
results/reproduction/breast-cancer-wisc-diag.csv
results/reproduction/breast-cancer-wisc-diag_comparison.csv
```

The reproduced repeat-averaged metrics closely match the authors' stored results, with a maximum absolute difference of approximately `0.000005`.

---

## Rice Evaluation

Run:

```bash
python evaluate_rice.py
```

Output:

```text
results/new_data/rice_gnb.csv
```

The evaluation contains 110 result rows:

- 11 methods
- 10 repetitions per method
- no missing metric values
- no duplicate method-repeat combinations

The train-only CiFRUS configuration performs almost identically to the baseline and produces small improvements in several metrics. It achieves the highest F1-score, Kappa and MCC in this experiment, while Borderline-SMOTE achieves the highest AUC.

---

## Weather Evaluation

Run:

```bash
python evaluate_weather.py
```

Output:

```text
results/new_data/weather_gnb_chronological.csv
```

The weather evaluation uses a chronological split:

- **Training:** 1 January 2023 to 30 December 2024
- **Excluded boundary period:** 31 December 2024
- **Testing:** 1 January 2025 to 30 December 2025

This gives:

- 87,600 training rows
- 120 excluded boundary rows
- 43,680 test rows

The 24-hour boundary period prevents training targets whose future precipitation windows would overlap the 2025 test period.

The weather experiment uses nine methods: baseline, ROS, RUS, SMOTE, Borderline-SMOTE, ADASYN and the three CiFRUS configurations.

SMOTE provides the strongest overall threshold-based performance on this dataset. Train-only CiFRUS remains close to the baseline, while balanced train/test CiFRUS substantially improves F1-score, Kappa, balanced accuracy and MCC relative to the baseline but slightly reduces AUC.

---

## Cross-Dataset Comparison

Run:

```bash
python compare_all_datasets.py
```

Output:

```text
results/final_comparison/cross_dataset_comparison.csv
```

The comparison shows that CiFRUS does not produce a uniform improvement across all datasets.

Its effectiveness varies according to:

- dataset characteristics;
- augmentation configuration; and
- evaluation metric.

The results therefore provide evidence of **partial and configuration-dependent generalisability**, rather than consistent superiority over conventional resampling methods.

---

## Notes on Reproducibility

This project performs a targeted reproduction rather than reproducing every dataset-classifier combination from the original paper.

The new-data experiments use Gaussian Naive Bayes to maintain a clear connection with the targeted original-data reproduction.

The weather evaluation intentionally differs from the repeated random cross-validation protocol because the observations are temporally ordered and the target depends on future observations. A chronological holdout is therefore used to reduce temporal information leakage.

---

## Original CiFRUS Project

CiFRUS stands for **Class-invariant Feature Range Uniform Sampling**.

The official implementation is compatible with scikit-learn and imbalanced-learn. The original authors provide the core augmentation implementation together with benchmark datasets and experimental results.

For details of the original method, implementation and citation, refer to:

https://github.com/sakhawatsaimon/CiFRUS

and:

https://doi.org/10.1145/3637528.3671736

---

## License and Attribution

The original CiFRUS implementation was developed by:

- Sakhawat Hossain Saimon
- Tanzira Najnin
- Jianhua Ruan

The original project is licensed under the Creative Commons Attribution 4.0 International License. See `LICENSE` for details.

Changes and additional scripts in this fork were created as part of a COMP8240 academic reproducibility and generalisability project.