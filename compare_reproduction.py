import pandas as pd
from pathlib import Path

DATASET = "breast-cancer-wisc-diag"
CLASSIFIER = "GNB"

original_path = Path(f"results/performance/{DATASET}.csv")
reproduced_path = Path(f"results/reproduction/{DATASET}.csv")
output_path = Path(f"results/reproduction/{DATASET}_comparison.csv")

metrics = [
    "AUC",
    "F1-score",
    "Kappa",
    "balanced-accuracy",
    "MCC"
]

# Load authors' original results and our independently reproduced results
original = pd.read_csv(original_path)
reproduced = pd.read_csv(reproduced_path)

# Compare the same classifier only
original = original[original["classifier"] == CLASSIFIER]
reproduced = reproduced[reproduced["classifier"] == CLASSIFIER]

# Mean across the 10 repeated CV runs
original_mean = original.groupby("augmentation")[metrics].mean()
reproduced_mean = reproduced.groupby("augmentation")[metrics].mean()

# Keep only augmentation methods present in both
common = original_mean.index.intersection(reproduced_mean.index)

original_mean = original_mean.loc[common]
reproduced_mean = reproduced_mean.loc[common]

# Build full comparison table
comparison = pd.DataFrame(index=common)

for metric in metrics:
    comparison[f"Original_{metric}"] = original_mean[metric]
    comparison[f"Reproduced_{metric}"] = reproduced_mean[metric]
    comparison[f"Difference_{metric}"] = (
        reproduced_mean[metric] - original_mean[metric]
    )
    comparison[f"AbsDifference_{metric}"] = (
        comparison[f"Difference_{metric}"].abs()
    )

comparison.index.name = "augmentation"
comparison.to_csv(output_path, float_format="%.6f")

# Compact AUC table for terminal / screenshot
auc_table = pd.DataFrame({
    "Authors": original_mean["AUC"],
    "Reproduced": reproduced_mean["AUC"],
})

auc_table["Difference"] = (
    auc_table["Reproduced"] - auc_table["Authors"]
)

print("\n==============================================================")
print(f"REPRODUCTION COMPARISON: {DATASET} | {CLASSIFIER}")
print("Means across 10 repeated 10-fold cross-validation runs")
print("==============================================================\n")

print("AUC comparison:\n")
print(auc_table.round(6).to_string())

print("\n--------------------------------------------------------------")
print("Mean absolute difference across ALL five metrics:")
print(f"{comparison.filter(like='AbsDifference_').to_numpy().mean():.6f}")

print("Maximum absolute difference across ALL five metrics:")
print(f"{comparison.filter(like='AbsDifference_').to_numpy().max():.6f}")

print("\nFull five-metric comparison saved to:")
print(output_path)
print("==============================================================")
