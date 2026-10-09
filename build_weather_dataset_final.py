import time
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
RAIN_THRESHOLD_MM = 1.0

LOCATIONS = {
    "Sydney": (
        -33.8688,
        151.2093,
        "Australia/Sydney"
    ),
    "Melbourne": (
        -37.8136,
        144.9631,
        "Australia/Melbourne"
    ),
    "Brisbane": (
        -27.4698,
        153.0251,
        "Australia/Brisbane"
    ),
    "Adelaide": (
        -34.9285,
        138.6007,
        "Australia/Adelaide"
    ),
    "Perth": (
        -31.9523,
        115.8613,
        "Australia/Perth"
    ),
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path("data_new/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "australian_weather_rain_final.csv"
)


# ---------------------------------------------------------
# Robust API request
# ---------------------------------------------------------

def fetch_location(name, lat, lon, timezone):

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
    }

    for attempt in range(1, 4):

        try:
            print(
                f"Collecting {name} "
                f"(attempt {attempt}/3)..."
            )

            response = requests.get(
                API_URL,
                params=params,
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            df = pd.DataFrame(
                data["hourly"]
            )

            df["time"] = pd.to_datetime(
                df["time"]
            )

            df["location"] = name
            df["timezone"] = timezone

            print(
                f"  {name}: "
                f"{len(df):,} hourly rows"
            )

            return df

        except requests.RequestException as e:

            print(
                f"  Request failed: {e}"
            )

            if attempt == 3:
                raise

            time.sleep(5)


# ---------------------------------------------------------
# Collect all locations
# ---------------------------------------------------------

all_locations = []

for (
    location,
    (lat, lon, timezone)
) in LOCATIONS.items():

    df = fetch_location(
        location,
        lat,
        lon,
        timezone
    )

    all_locations.append(df)


# ---------------------------------------------------------
# Combine
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

weather["hour"] = (
    weather["time"].dt.hour
)

weather["month"] = (
    weather["time"].dt.month
)


# ---------------------------------------------------------
# Construct future-rain target
# ---------------------------------------------------------

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

weather["rain_next_24h"] = (
    weather["future_24h_precipitation"]
    >= RAIN_THRESHOLD_MM
).astype("Int64")


# ---------------------------------------------------------
# Remove incomplete future windows
# ---------------------------------------------------------

weather = weather.dropna(
    subset=["future_24h_precipitation"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

weather.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n==========================================")
print("FINAL WEATHER DATASET CREATED")
print("==========================================")

print(
    f"Rows: {len(weather):,}"
)

print(
    f"Columns: {weather.shape[1]}"
)

print(
    f"Locations: "
    f"{weather['location'].nunique()}"
)

print(
    f"Start: {weather['time'].min()}"
)

print(
    f"End: {weather['time'].max()}"
)

print("\nTimezone by location:")

print(
    weather[
        ["location", "timezone"]
    ]
    .drop_duplicates()
    .sort_values("location")
    .to_string(index=False)
)

print("\nClass counts:")

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

print(
    "\nMissing values:",
    weather.isna().sum().sum()
)

print("\nSaved to:")
print(OUTPUT_FILE)