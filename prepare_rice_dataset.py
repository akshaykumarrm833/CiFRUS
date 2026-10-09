from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------
# Load Rice dataset from UCI
# ---------------------------------------------------------

rice = fetch_ucirepo(id=545)

X = rice.data.features.copy()
y = rice.data.targets.copy()

df = X.copy()
df["Class"] = y.iloc[:, 0]

# ---------------------------------------------------------
# Save locally
# ---------------------------------------------------------

output_dir = Path("data_new/rice")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "rice_cammeo_osmancik.csv"
df.to_csv(output_file, index=False)

# ---------------------------------------------------------
# Check whether dataset appears in original CiFRUS list
# ---------------------------------------------------------

original_names = pd.read_csv(
    "data/dataset_names.txt",
    header=None,
    quotechar="'"
).iloc[:, 0].astype(str)

rice_name_matches = original_names[
    original_names.str.contains(
        "rice",
        case=False,
        na=False
    )
]

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("========== NEW EXISTING DATASET: RICE ==========")

print("Source: UCI Machine Learning Repository")
print("UCI dataset ID: 545")
print("Task: Binary classification")

print("\nRows:", len(df))
print("Features:", X.shape[1])
print("Stored columns:", df.shape[1])

print("\nFeature names:")
for col in X.columns:
    print(" -", col)

print("\nClass counts:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print(
    (
        df["Class"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nMissing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nOriginal CiFRUS dataset-name matches containing 'rice':")
if len(rice_name_matches) == 0:
    print("None")
else:
    print(rice_name_matches.to_string(index=False))

print("\nSaved to:")
print(output_file)