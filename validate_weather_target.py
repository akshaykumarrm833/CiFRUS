import pandas as pd
import numpy as np

FILE = "data_new/weather/australian_weather_rain_final.csv"
RAIN_THRESHOLD_MM = 1.0

df = pd.read_csv(FILE)
df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["location", "time"]
).reset_index(drop=True)

# Recalculate next-24-hour precipitation independently
recalculated = (
    df.groupby("location")["precipitation"]
      .transform(
          lambda s:
          s.shift(-1)
           .rolling(window=24, min_periods=24)
           .sum()
           .shift(-23)
      )
)

# Only compare rows where a complete future 24h window exists
valid = recalculated.notna()

precip_difference = (
    df.loc[valid, "future_24h_precipitation"]
    - recalculated.loc[valid]
).abs()

recalculated_target = (
    recalculated.loc[valid] >= RAIN_THRESHOLD_MM
).astype(int)

target_matches = (
    df.loc[valid, "rain_next_24h"].astype(int)
    == recalculated_target
)

print("========== TARGET VALIDATION ==========")

print("\nRows checked:", valid.sum())

print(
    "Maximum precipitation-sum difference:",
    precip_difference.max()
)

print(
    "Precipitation sums matching:",
    np.isclose(
        precip_difference,
        0,
        atol=1e-9
    ).sum(),
    "/",
    valid.sum()
)

print(
    "Binary targets matching:",
    target_matches.sum(),
    "/",
    valid.sum()
)

print(
    "Target mismatches:",
    (~target_matches).sum()
)

print("\n========== LEAKAGE CHECK ==========")

predictor_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
    "hour",
    "month",
]

print("Intended predictor columns:")
for col in predictor_columns:
    print(" -", col)

print(
    "\nFuture precipitation is retained only "
    "for target verification and will NOT be used as a predictor."
)

print("\n========== SAMPLE POSITIVE ROWS ==========")

positive = df[df["rain_next_24h"] == 1][
    [
        "time",
        "location",
        "precipitation",
        "future_24h_precipitation",
        "rain_next_24h"
    ]
].head(10)

print(positive.to_string(index=False))