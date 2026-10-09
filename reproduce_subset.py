#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.io import loadmat

# Scoring metrics
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)

# Classifier
from sklearn.naive_bayes import GaussianNB

# Augmentation algorithms
from cifrus.cifrus import CiFRUS
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN
)
from imblearn_extra.gsmote import GeometricSMOTE

from sklearn.model_selection import StratifiedKFold

from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):,}h {int(minutes)}m {int(seconds)}s"


# ---------------------------------------------------------
# Reproduction setup
# ---------------------------------------------------------

SEED = 2023

# We are running the experiment now
DRY_RUN = False

# Keep our reproduced results separate from authors' results
OUTPUT_DIR = Path("./results/reproduction")

# Original CiFRUS settings
k_func = None
r = 10
r_majority = 5

# Same evaluation protocol as original study
n_splits = 10
n_repeats = 10

p_threshold = 0.5

# First reproduction:
# one classifier only
classifiers = {
    'GNB': GaussianNB()
}

augmenters = {
    'ROS': RandomOverSampler(random_state=SEED),
    'RUS': RandomUnderSampler(random_state=SEED),
    'SMOTE': SMOTE(random_state=SEED),
    'Borderline-SMOTE': BorderlineSMOTE(random_state=SEED),
    'SVM-SMOTE': SVMSMOTE(random_state=SEED),
    'ADASYN': ADASYN(random_state=SEED),
    'G-SMOTE': GeometricSMOTE(random_state=SEED)
}

cifrus_names = [
    'CiFRUS (train)',
    'CiFRUS (train/test)',
    'CiFRUS (balanced-train/test)'
]

metrics = {
    'AUC': lambda y_true, y_pred, scores:
        roc_auc_score(y_true, scores),

    'F1-score': lambda y_true, y_pred, scores:
        f1_score(y_true, y_pred),

    'Kappa': lambda y_true, y_pred, scores:
        cohen_kappa_score(y_true, y_pred),

    'balanced-accuracy': lambda y_true, y_pred, scores:
        balanced_accuracy_score(y_true, y_pred),

    'MCC': lambda y_true, y_pred, scores:
        matthews_corrcoef(y_true, y_pred)
}


# ---------------------------------------------------------
# Load original CiFRUS datasets
# ---------------------------------------------------------

basepath = Path('./data/')

dataset_names = pd.read_csv(
    Path(basepath, 'dataset_names.txt'),
    header=None,
    quotechar="'"
).squeeze()

dataset_info = pd.DataFrame(
    index=range(1, len(dataset_names) + 1),
    columns=['name', 'samples', 'features']
)

subdirs = sorted(
    [d for d in basepath.iterdir() if d.is_dir()]
)

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

for d in subdirs:

    dataset_idx = int(d.name)

    dataset_name = dataset_names[
        dataset_idx - 1
    ]

    dataset_info.loc[
        dataset_idx, :
    ] = (
        dataset_name,
        *loadmat(
            d.joinpath('data.mat')
        )['X'].shape
    )

dataset_info_arr = (
    dataset_info
    .sort_values('samples')
    .reset_index()
    .to_numpy()
)


# ---------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------

def metrics_from_scores(scores):

    metric_df = pd.DataFrame(
        index=scores.columns,
        columns=metrics.keys(),
        dtype=float
    )

    y_true = scores['Y_true']

    for metric_name, metric_func in metrics.items():

        for augmenter_name in metric_df.index:

            scores_augmenter = scores[
                augmenter_name
            ]

            y_pred = (
                scores_augmenter >= p_threshold
            ).astype(int)

            metric_df.loc[
                augmenter_name,
                metric_name
            ] = metric_func(
                y_true,
                y_pred,
                scores_augmenter
            )

    metric_df = metric_df.drop('Y_true')

    return metric_df


# ---------------------------------------------------------
# Reproduction experiment
# ---------------------------------------------------------

starttime = time.time()

TARGET_DATASET = 'breast-cancer-wisc-diag'

for (
    dataset_idx,
    dataset_name,
    n,
    m
) in dataset_info_arr:

    # Run only our selected original dataset
    if dataset_name != TARGET_DATASET:
        continue

    dataset_starttime = time.time()

    print(
        f"\nReproducing: "
        f"{dataset_name} "
        f"(n={n}, m={m})"
    )

    mat = loadmat(
        basepath.joinpath(
            f'{dataset_idx}/data.mat'
        )
    )

    X = mat['X']
    Y = mat['y'].ravel()

    Y = pd.factorize(
        Y,
        sort=True
    )[0]

    out_path = Path(
        f"{OUTPUT_DIR}/{dataset_name}.csv"
    )

    if DRY_RUN:
        continue

    results = {}

    for it in range(n_repeats):

        print(
            f"Repeat {it + 1}/{n_repeats}"
        )

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=it
        )

        results_clf = {}

        for classifier_name, clf in classifiers.items():

            print(
                f"  Classifier: "
                f"{classifier_name}"
            )

            scores = []

            for (
                train_index,
                test_index
            ) in cv.split(X, Y):

                X_train = X[train_index]
                Y_train = Y[train_index]

                X_test = X[test_index]
                Y_test = Y[test_index]

                scores_fold = pd.DataFrame(
                    Y_test,
                    columns=['Y_true']
                )

                # -----------------------------
                # Baseline
                # -----------------------------

                clf.fit(
                    X_train,
                    Y_train
                )

                scores_fold[
                    'Baseline'
                ] = clf.predict_proba(
                    X_test
                )[:, -1]

                # -----------------------------
                # Traditional resampling
                # -----------------------------

                for (
                    augmenter_name,
                    augmenter
                ) in augmenters.items():

                    try:

                        X_train_v, y_train_v = (
                            augmenter.fit_resample(
                                X_train,
                                Y_train
                            )
                        )

                        clf.fit(
                            X_train_v,
                            y_train_v
                        )

                        scores_fold[
                            augmenter_name
                        ] = clf.predict_proba(
                            X_test
                        )[:, 1]

                    except Exception as e:

                        if augmenter_name == 'ADASYN':

                            scores_fold[
                                augmenter_name
                            ] = scores_fold[
                                'Baseline'
                            ]

                        else:

                            print(
                                f"    "
                                f"{augmenter_name} "
                                f"error: {e}"
                            )

                            scores_fold[
                                augmenter_name
                            ] = np.nan

                # -----------------------------
                # CiFRUS
                # -----------------------------

                cifrus = CiFRUS(
                    k=k_func,
                    random_state=SEED
                )

                cifrus.fit(
                    X_train
                )

                # CiFRUS train augmentation
                X_train_v, Y_train_v = (
                    cifrus.resample(
                        X_train,
                        Y=Y_train,
                        r=r
                    )
                )

                clf.fit(
                    X_train_v,
                    Y_train_v
                )

                scores_fold[
                    cifrus_names[0]
                ] = clf.predict_proba(
                    X_test
                )[:, 1]

                # CiFRUS train/test
                scores_fold[
                    cifrus_names[1]
                ] = (
                    cifrus
                    .resample_predict_proba(
                        clf.predict_proba,
                        X_test,
                        r=r
                    )[:, 1]
                )

                # CiFRUS balanced train/test
                X_train_v, Y_train_v = (
                    cifrus.resample_balanced(
                        X_train,
                        Y=Y_train,
                        r_majority=r_majority
                    )
                )

                clf.fit(
                    X_train_v,
                    Y_train_v
                )

                scores_fold[
                    cifrus_names[2]
                ] = (
                    cifrus
                    .resample_predict_proba(
                        clf.predict_proba,
                        X_test,
                        r=r
                    )[:, 1]
                )

                scores.append(
                    scores_fold
                )

            scores = pd.concat(
                scores
            ).reset_index(
                drop=True
            )

            metric_df = (
                metrics_from_scores(
                    scores
                )
            )

            results_clf[
                classifier_name
            ] = metric_df

        results_clf = pd.concat(
            results_clf
        )

        results[it] = results_clf

    # -----------------------------------------------------
    # Save reproduced results
    # -----------------------------------------------------

    results = pd.concat(
        results,
        names=[
            'iter',
            'classifier',
            'augmentation'
        ]
    )

    results = results.reorder_levels(
        [1, 0, 2]
    )

    results = results.loc[
        classifiers.keys(), :
    ]

    results.to_csv(
        out_path,
        float_format='%.6f'
    )

    print(
        "\nMean reproduced AUC:"
    )

    print(
        results
        .groupby([
            'classifier',
            'augmentation'
        ])
        .mean()['AUC']
        .unstack()
        .T
    )

    dataset_endtime = time.time()

    print(
        "\nDataset time:",
        format_time(
            dataset_endtime -
            dataset_starttime
        )
    )

    print(
        "\nSaved reproduced results to:"
    )

    print(out_path)


endtime = time.time()

print(
    "\nTotal elapsed time:",
    format_time(
        endtime - starttime
    )
)