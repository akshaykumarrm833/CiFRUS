import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.naive_bayes import GaussianNB

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    ADASYN,
)

from cifrus.cifrus import CiFRUS


# =========================================================
# Configuration
# =========================================================

SEED = 2023

R = 10
R_MAJORITY = 5
K_FUNC = None
P_THRESHOLD = 0.5

INPUT_FILE = Path(
    "data_new/weather/australian_weather_rain_final.csv"
)

OUTPUT_DIR = Path("results/new_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "weather_gnb_chronological.csv"


FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
    "hour",
    "month",
]


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["time"],
)

train_df = df[
    df["time"] <= "2024-12-30 23:00:00"
].copy()

gap_df = df[
    (df["time"] >= "2024-12-31 00:00:00")
    & (df["time"] < "2025-01-01 00:00:00")
].copy()

test_df = df[
    df["time"] >= "2025-01-01 00:00:00"
].copy()


X_train = train_df[
    FEATURE_COLUMNS
].to_numpy(dtype=float)

y_train = train_df[
    "rain_next_24h"
].to_numpy(dtype=int)

X_test = test_df[
    FEATURE_COLUMNS
].to_numpy(dtype=float)

y_test = test_df[
    "rain_next_24h"
].to_numpy(dtype=int)


print("==============================================")
print("AUSTRALIAN WEATHER — CHRONOLOGICAL EVALUATION")
print("==============================================")

print("\nTraining rows:", len(train_df))
print(
    "Training range:",
    train_df["time"].min(),
    "->",
    train_df["time"].max(),
)

print("\nGap rows:", len(gap_df))
print(
    "Gap range:",
    gap_df["time"].min(),
    "->",
    gap_df["time"].max(),
)

print("\nTest rows:", len(test_df))
print(
    "Test range:",
    test_df["time"].min(),
    "->",
    test_df["time"].max(),
)

print("\nFeatures:")
for column in FEATURE_COLUMNS:
    print(" -", column)

print("\nTrain class counts:")
print(train_df["rain_next_24h"].value_counts())

print("\nTest class counts:")
print(test_df["rain_next_24h"].value_counts())


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(y_true, scores):

    predictions = (
        scores >= P_THRESHOLD
    ).astype(int)

    return {
        "AUC": roc_auc_score(
            y_true,
            scores,
        ),
        "F1-score": f1_score(
            y_true,
            predictions,
        ),
        "Kappa": cohen_kappa_score(
            y_true,
            predictions,
        ),
        "balanced-accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),
        "MCC": matthews_corrcoef(
            y_true,
            predictions,
        ),
    }


# =========================================================
# Helper
# =========================================================

def evaluate_model(
    method_name,
    classifier,
    X_eval,
    y_eval,
):

    scores = classifier.predict_proba(
        X_eval
    )[:, 1]

    metrics = calculate_metrics(
        y_eval,
        scores,
    )

    return {
        "augmentation": method_name,
        **metrics,
    }


# =========================================================
# Evaluation
# =========================================================

start_time = time.time()

results = []

base_classifier = GaussianNB()


# ---------------------------------------------------------
# Baseline
# ---------------------------------------------------------

print("\nRunning Baseline...")

clf = clone(base_classifier)

clf.fit(
    X_train,
    y_train,
)

results.append(
    evaluate_model(
        "Baseline",
        clf,
        X_test,
        y_test,
    )
)


# ---------------------------------------------------------
# Conventional resampling
# ---------------------------------------------------------

augmenters = {
    "ROS": RandomOverSampler(
        random_state=SEED,
    ),
    "RUS": RandomUnderSampler(
        random_state=SEED,
    ),
    "SMOTE": SMOTE(
        random_state=SEED,
    ),
    "Borderline-SMOTE": BorderlineSMOTE(
        random_state=SEED,
    ),
    "ADASYN": ADASYN(
        random_state=SEED,
    ),
}


for method_name, augmenter in augmenters.items():

    print(f"Running {method_name}...")

    method_start = time.time()

    X_resampled, y_resampled = (
        augmenter.fit_resample(
            X_train,
            y_train,
        )
    )

    clf = clone(base_classifier)

    clf.fit(
        X_resampled,
        y_resampled,
    )

    results.append(
        evaluate_model(
            method_name,
            clf,
            X_test,
            y_test,
        )
    )

    print(
        f"  Completed in "
        f"{time.time() - method_start:.1f}s"
    )


# ---------------------------------------------------------
# CiFRUS
# ---------------------------------------------------------

print("Running CiFRUS fit...")

cifrus_start = time.time()

cifrus = CiFRUS(
    k=K_FUNC,
    random_state=SEED,
)

cifrus.fit(
    X_train
)

print(
    f"  CiFRUS fit completed in "
    f"{time.time() - cifrus_start:.1f}s"
)


# ---------------------------------------------------------
# CiFRUS train
# ---------------------------------------------------------

print("Running CiFRUS (train)...")

method_start = time.time()

X_train_aug, y_train_aug = (
    cifrus.resample(
        X_train,
        Y=y_train,
        r=R,
    )
)

clf_train = clone(
    base_classifier
)

clf_train.fit(
    X_train_aug,
    y_train_aug,
)

results.append(
    evaluate_model(
        "CiFRUS (train)",
        clf_train,
        X_test,
        y_test,
    )
)

print(
    f"  Completed in "
    f"{time.time() - method_start:.1f}s"
)


# ---------------------------------------------------------
# CiFRUS train/test
# ---------------------------------------------------------

print("Running CiFRUS (train/test)...")

method_start = time.time()

scores = (
    cifrus
    .resample_predict_proba(
        clf_train.predict_proba,
        X_test,
        r=R,
    )[:, 1]
)

results.append({
    "augmentation":
        "CiFRUS (train/test)",
    **calculate_metrics(
        y_test,
        scores,
    ),
})

print(
    f"  Completed in "
    f"{time.time() - method_start:.1f}s"
)


# ---------------------------------------------------------
# CiFRUS balanced train/test
# ---------------------------------------------------------

print(
    "Running CiFRUS "
    "(balanced-train/test)..."
)

method_start = time.time()

X_train_bal, y_train_bal = (
    cifrus.resample_balanced(
        X_train,
        Y=y_train,
        r_majority=R_MAJORITY,
    )
)

clf_balanced = clone(
    base_classifier
)

clf_balanced.fit(
    X_train_bal,
    y_train_bal,
)

scores = (
    cifrus
    .resample_predict_proba(
        clf_balanced.predict_proba,
        X_test,
        r=R,
    )[:, 1]
)

results.append({
    "augmentation":
        "CiFRUS (balanced-train/test)",
    **calculate_metrics(
        y_test,
        scores,
    ),
})

print(
    f"  Completed in "
    f"{time.time() - method_start:.1f}s"
)


# =========================================================
# Save
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df.insert(
    0,
    "classifier",
    "GNB",
)

results_df.insert(
    1,
    "train_rows",
    len(train_df),
)

results_df.insert(
    2,
    "test_rows",
    len(test_df),
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f",
)


# =========================================================
# Display
# =========================================================

elapsed = time.time() - start_time

display_columns = [
    "augmentation",
    "AUC",
    "F1-score",
    "Kappa",
    "balanced-accuracy",
    "MCC",
]

print("\n==============================================")
print("WEATHER RESULTS")
print("==============================================\n")

print(
    results_df[
        display_columns
    ]
    .sort_values(
        "AUC",
        ascending=False,
    )
    .round(6)
    .to_string(index=False)
)

print("\n==============================================")

print(
    "Methods completed:",
    len(results_df),
)

print(
    f"Total elapsed time: "
    f"{elapsed:.1f} seconds"
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("==============================================")