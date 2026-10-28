from pathlib import Path
import pandas as pd


# =========================================================
# Paths
# =========================================================

ORIGINAL_FILE = Path(
    "results/reproduction/breast-cancer-wisc-diag.csv"
)

RICE_FILE = Path(
    "results/new_data/rice_gnb.csv"
)

WEATHER_FILE = Path(
    "results/new_data/weather_gnb_chronological.csv"
)

OUTPUT_DIR = Path("results/final_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cross_dataset_comparison.csv"


METRICS = [
    "AUC",
    "F1-score",
    "Kappa",
    "balanced-accuracy",
    "MCC",
]


# =========================================================
# Load
# =========================================================

original = pd.read_csv(ORIGINAL_FILE)
rice = pd.read_csv(RICE_FILE)
weather = pd.read_csv(WEATHER_FILE)


# =========================================================
# Mean repeated results
# =========================================================

original_mean = (
    original
    .groupby("augmentation")[METRICS]
    .mean()
)

rice_mean = (
    rice
    .groupby("augmentation")[METRICS]
    .mean()
)

weather_mean = (
    weather
    .set_index("augmentation")[METRICS]
)


# =========================================================
# Methods shared across all three datasets
# =========================================================

shared_methods = sorted(
    set(original_mean.index)
    & set(rice_mean.index)
    & set(weather_mean.index)
)

print("Shared methods across all datasets:")
for method in shared_methods:
    print(" -", method)


# =========================================================
# Combine
# =========================================================

rows = []

datasets = {
    "Original breast cancer": original_mean,
    "Rice": rice_mean,
    "Weather": weather_mean,
}

for dataset_name, results in datasets.items():

    baseline = results.loc["Baseline"]

    for method in shared_methods:

        values = results.loc[method]

        row = {
            "dataset": dataset_name,
            "method": method,
        }

        for metric in METRICS:

            row[metric] = values[metric]

            row[f"{metric}_vs_baseline"] = (
                values[metric]
                - baseline[metric]
            )

        rows.append(row)


combined = pd.DataFrame(rows)


# =========================================================
# Save detailed comparison
# =========================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f",
)


# =========================================================
# CiFRUS-focused summary
# =========================================================

cifrus_methods = [
    "CiFRUS (train)",
    "CiFRUS (train/test)",
    "CiFRUS (balanced-train/test)",
]

cifrus_summary = combined[
    combined["method"].isin(cifrus_methods)
].copy()


print("\n==============================================")
print("CIFRUS — CHANGE RELATIVE TO BASELINE")
print("==============================================\n")

display_cols = [
    "dataset",
    "method",
    "AUC_vs_baseline",
    "F1-score_vs_baseline",
    "Kappa_vs_baseline",
    "balanced-accuracy_vs_baseline",
    "MCC_vs_baseline",
]

print(
    cifrus_summary[
        display_cols
    ]
    .round(6)
    .to_string(index=False)
)


# =========================================================
# Best method per dataset / metric
# =========================================================

print("\n==============================================")
print("BEST METHOD BY DATASET AND METRIC")
print("==============================================")

for dataset_name, results in datasets.items():

    print(f"\n{dataset_name}")

    for metric in METRICS:

        best_method = (
            results[metric]
            .idxmax()
        )

        best_value = (
            results.loc[
                best_method,
                metric
            ]
        )

        print(
            f"{metric:20s} "
            f"{best_method:32s} "
            f"{best_value:.6f}"
        )


print("\n==============================================")
print("Saved to:")
print(OUTPUT_FILE)
print("==============================================")