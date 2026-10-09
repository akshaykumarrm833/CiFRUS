import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

LOCATIONS = {
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Adelaide": (-34.9285, 138.6007),
    "Perth": (-31.9523, 115.8613),
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
]

OUTPUT_DIR = Path("data_new/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------
# Collection
# ---------------------------------------------------------

all_locations = []

for location, (lat, lon) in LOCATIONS.items():

    print(f"Collecting {location}...")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Australia/Sydney",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    df["time"] = pd.to_datetime(df["time"])

    df["location"] = location

    all_locations.append(df)


# ---------------------------------------------------------
# Combine locations
# ---------------------------------------------------------

weather = pd.concat(
    all_locations,
    ignore_index=True
)

weather = weather.sort_values(
    ["location", "time"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

weather["hour"] = weather["time"].dt.hour
weather["month"] = weather["time"].dt.month

# Rain target:
# total precipitation during the NEXT 24 hours
weather["future_24h_precipitation"] = (
    weather
    .groupby("location")["precipitation"]
    .transform(
        lambda s:
        s.shift(-1)
         .rolling(
             window=24,
             min_periods=24
         )
         .sum()
         .shift(-23)
    )
)

# Binary target:
# 1 if at least 1 mm falls in the next 24 hours
RAIN_THRESHOLD_MM = 1.0

weather["rain_next_24h"] = (
    weather["future_24h_precipitation"]
    >= RAIN_THRESHOLD_MM
).astype("Int64")


# ---------------------------------------------------------
# Remove rows without complete future window
# ---------------------------------------------------------

weather = weather.dropna(
    subset=["future_24h_precipitation"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

output_file = OUTPUT_DIR / "australian_weather_rain.csv"

weather.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("WEATHER DATASET CREATED")
print("========================================")

print(f"Rows: {len(weather):,}")
print(f"Columns: {weather.shape[1]}")
print(f"Locations: {weather['location'].nunique()}")
print(f"Start: {weather['time'].min()}")
print(f"End: {weather['time'].max()}")

print("\nClass distribution:")
print(
    weather["rain_next_24h"]
    .value_counts()
    .sort_index()
)

print("\nClass percentages:")
print(
    (
        weather["rain_next_24h"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)
)

print("\nMissing values:")
print(
    weather.isna()
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

print("\nSaved to:")
print(output_file)

print("\nFirst 5 rows:")
print(weather.head().to_string(index=False))
