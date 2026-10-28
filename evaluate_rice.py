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
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
)
from imblearn_extra.gsmote import GeometricSMOTE

from cifrus.cifrus import CiFRUS


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 2023

N_SPLITS = 10
N_REPEATS = 10

R = 10
R_MAJORITY = 5
K_FUNC = None

P_THRESHOLD = 0.5

INPUT_FILE = Path(
    "data_new/rice/rice_cammeo_osmancik.csv"
)

OUTPUT_DIR = Path("results/new_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "rice_gnb.csv"


# ---------------------------------------------------------
# Load Rice dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

feature_columns = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]

X = df[feature_columns].to_numpy(dtype=float)

# Cammeo = 0, Osmancik = 1
Y = (
    df["Class"]
    .map({
        "Cammeo": 0,
        "Osmancik": 1,
    })
    .to_numpy(dtype=int)
)

print("==============================================")
print("RICE DATASET — CiFRUS EVALUATION")
print("==============================================")

print(f"Samples: {len(df):,}")
print(f"Features: {len(feature_columns)}")

print("\nClass counts:")
print(df["Class"].value_counts())

print("\nMissing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------
# Classifier
# ---------------------------------------------------------

base_classifier = GaussianNB()


# ---------------------------------------------------------
# Traditional resampling methods
# ---------------------------------------------------------

def make_augmenters():
    return {
        "ROS": RandomOverSampler(
            random_state=SEED
        ),
        "RUS": RandomUnderSampler(
            random_state=SEED
        ),
        "SMOTE": SMOTE(
            random_state=SEED
        ),
        "Borderline-SMOTE": BorderlineSMOTE(
            random_state=SEED
        ),
        "SVM-SMOTE": SVMSMOTE(
            random_state=SEED
        ),
        "ADASYN": ADASYN(
            random_state=SEED
        ),
        "G-SMOTE": GeometricSMOTE(
            random_state=SEED
        ),
    }


CIFRUS_NAMES = [
    "CiFRUS (train)",
    "CiFRUS (train/test)",
    "CiFRUS (balanced-train/test)",
]


# ---------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------

def calculate_metrics(y_true, scores):

    y_pred = (
        scores >= P_THRESHOLD
    ).astype(int)

    return {
        "AUC": roc_auc_score(
            y_true,
            scores
        ),
        "F1-score": f1_score(
            y_true,
            y_pred
        ),
        "Kappa": cohen_kappa_score(
            y_true,
            y_pred
        ),
        "balanced-accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),
        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        ),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

start_time = time.time()

all_results = []

for repeat in range(N_REPEATS):

    print(
        f"\nRepeat "
        f"{repeat + 1}/{N_REPEATS}"
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=repeat,
    )

    repeat_scores = {
        "Baseline": [],
        "ROS": [],
        "RUS": [],
        "SMOTE": [],
        "Borderline-SMOTE": [],
        "SVM-SMOTE": [],
        "ADASYN": [],
        "G-SMOTE": [],
        "CiFRUS (train)": [],
        "CiFRUS (train/test)": [],
        "CiFRUS (balanced-train/test)": [],
    }

    repeat_y_true = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, Y),
        start=1,
    ):

        X_train = X[train_idx]
        Y_train = Y[train_idx]

        X_test = X[test_idx]
        Y_test = Y[test_idx]

        repeat_y_true.extend(
            Y_test.tolist()
        )

        # ---------------------------------------------
        # Baseline
        # ---------------------------------------------

        clf = clone(base_classifier)

        clf.fit(
            X_train,
            Y_train
        )

        baseline_scores = (
            clf.predict_proba(
                X_test
            )[:, 1]
        )

        repeat_scores[
            "Baseline"
        ].extend(
            baseline_scores.tolist()
        )

        # ---------------------------------------------
        # Traditional resampling
        # ---------------------------------------------

        augmenters = make_augmenters()

        for (
            augmenter_name,
            augmenter
        ) in augmenters.items():

            try:

                X_train_aug, Y_train_aug = (
                    augmenter.fit_resample(
                        X_train,
                        Y_train
                    )
                )

                clf = clone(
                    base_classifier
                )

                clf.fit(
                    X_train_aug,
                    Y_train_aug
                )

                scores = (
                    clf.predict_proba(
                        X_test
                    )[:, 1]
                )

                repeat_scores[
                    augmenter_name
                ].extend(
                    scores.tolist()
                )

            except Exception as e:

                print(
                    f"  Fold {fold}: "
                    f"{augmenter_name} failed: "
                    f"{e}"
                )

                repeat_scores[
                    augmenter_name
                ].extend(
                    [np.nan] * len(Y_test)
                )

        # ---------------------------------------------
        # CiFRUS
        # ---------------------------------------------

        cifrus = CiFRUS(
            k=K_FUNC,
            random_state=SEED
        )

        cifrus.fit(
            X_train
        )

        # CiFRUS: train augmentation
        X_train_aug, Y_train_aug = (
            cifrus.resample(
                X_train,
                Y=Y_train,
                r=R
            )
        )

        clf = clone(
            base_classifier
        )

        clf.fit(
            X_train_aug,
            Y_train_aug
        )

        scores = (
            clf.predict_proba(
                X_test
            )[:, 1]
        )

        repeat_scores[
            CIFRUS_NAMES[0]
        ].extend(
            scores.tolist()
        )

        # CiFRUS: train/test augmentation
        scores = (
            cifrus
            .resample_predict_proba(
                clf.predict_proba,
                X_test,
                r=R
            )[:, 1]
        )

        repeat_scores[
            CIFRUS_NAMES[1]
        ].extend(
            scores.tolist()
        )

        # CiFRUS: balanced train/test
        X_train_bal, Y_train_bal = (
            cifrus.resample_balanced(
                X_train,
                Y=Y_train,
                r_majority=R_MAJORITY
            )
        )

        clf = clone(
            base_classifier
        )

        clf.fit(
            X_train_bal,
            Y_train_bal
        )

        scores = (
            cifrus
            .resample_predict_proba(
                clf.predict_proba,
                X_test,
                r=R
            )[:, 1]
        )

        repeat_scores[
            CIFRUS_NAMES[2]
        ].extend(
            scores.tolist()
        )

    # -------------------------------------------------
    # Metrics for this repeat
    # -------------------------------------------------

    y_true = np.array(
        repeat_y_true
    )

    for (
        method,
        scores
    ) in repeat_scores.items():

        scores = np.array(
            scores,
            dtype=float
        )

        if np.isnan(scores).any():

            print(
                f"  WARNING: {method} "
                f"contains missing predictions."
            )

            continue

        metric_values = (
            calculate_metrics(
                y_true,
                scores
            )
        )

        all_results.append({
            "classifier": "GNB",
            "iter": repeat,
            "augmentation": method,
            **metric_values,
        })


# ---------------------------------------------------------
# Save full results
# ---------------------------------------------------------

results = pd.DataFrame(
    all_results
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f"
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

summary = (
    results
    .groupby("augmentation")[
        [
            "AUC",
            "F1-score",
            "Kappa",
            "balanced-accuracy",
            "MCC",
        ]
    ]
    .mean()
    .sort_values(
        "AUC",
        ascending=False
    )
)

elapsed = time.time() - start_time

print("\n==============================================")
print("MEAN RESULTS ACROSS 10 REPEATS")
print("==============================================\n")

print(
    summary.round(6).to_string()
)

print("\n==============================================")

print(
    f"Rows saved: {len(results)}"
)

print(
    f"Elapsed time: "
    f"{elapsed:.1f} seconds"
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)

print("==============================================")