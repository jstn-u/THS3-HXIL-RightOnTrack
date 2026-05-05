import pandas as pd
import json
import numpy as np
from geopy.distance import geodesic


df = pd.read_csv("/Users/joehannacansino/Desktop/THS3/data/train_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")

df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

samples = []

# Group by trip_id AND date (so each trip instance is per day)
for (trip_id, trip_date), g in df.groupby(["trip_id", df["timestamp"].dt.date]):
    g = g.sort_values("timestamp").copy()

    # Keep only valid coordinate rows
    g = g[g["latitude"].notna() & g["longitude"].notna()].copy()
    if len(g) == 0:
        continue

    driver_id = int(g["vehicleID"].iloc[0]) if pd.notna(g["vehicleID"].iloc[0]) else -1
    start_ts = g["timestamp"].iloc[0]

    date_id = int(pd.to_datetime(trip_date).day - 1)   # 0-30
    week_id = int(pd.to_datetime(trip_date).weekday())  # Mon=0 ... Sun=6
    time_id = int(start_ts.hour * 60 + start_ts.minute)

    # Trip totals (row-level incremental values)
    dist_km = g["distance_m"].sum() / 1000.0            # km
    time_min = g["travel_time_sec"].sum() / 60.0       # minutes

    # Trajectory sequences
    lngs = g["longitude"].astype(float).tolist()
    lats = g["latitude"].astype(float).tolist()

    # Use currentStatus as states
    states = g["currentStatus"].fillna(0).astype(int).tolist()

    # time_gap: elapsed time from first point
    time_gap = (g["travel_time_sec"].fillna(0).clip(lower=0).astype(float) / 60.0).tolist()
    if len(time_gap) > 0:
        time_gap[0] = 0.0

    # dist_gap: cumulative distance from first point
    dist_gap = (g["distance_m"].fillna(0).clip(lower=0).astype(float).cumsum() / 1000.0).tolist()
    if len(dist_gap) > 0:
        dist_gap[0] = 0.0

    
    arrival_delays = g["arrivalDelay"].fillna(0).astype(float).tolist()
    departure_delays = g["departureDelay"].fillna(0).astype(float).tolist()
    speeds = g["speed_kph"].fillna(0).astype(float).tolist()
    is_peak_hours = g["is_peak_hour"].fillna(0).astype(int).tolist()

    temperature_2m = g["temperature_2m"].fillna(0).astype(float).tolist()
    apparent_temperature = g["apparent_temperature"].fillna(0).astype(float).tolist()
    precipitation = g["precipitation"].fillna(0).astype(float).tolist()
    rain = g["rain"].fillna(0).astype(float).tolist()
    snowfall = g["snowfall"].fillna(0).astype(float).tolist()
    windspeed_10m = g["windspeed_10m"].fillna(0).astype(float).tolist()
    windgusts_10m = g["windgusts_10m"].fillna(0).astype(float).tolist()
    winddirection_10m = g["winddirection_10m"].fillna(0).astype(float).tolist()

    sample = {
    "driverID": driver_id,
    "tripID": str(trip_id),
    "dateID": date_id,
    "weekID": week_id,
    "timeID": time_id,

    "dist": dist_km,
    "time": time_min,

    "lngs": lngs,
    "lats": lats,
    "states": states,
    "time_gap": time_gap,
    "dist_gap": dist_gap,

    "arrival_delay": arrival_delays,
    "departure_delay": departure_delays,
    "speed": speeds,
    "is_peak_hour": is_peak_hours,

    "temperature": temperature_2m,
    "apparent_temperature": apparent_temperature,
    "precipitation": precipitation,
    "rain": rain,
    "snowfall": snowfall,
    "windspeed": windspeed_10m,
    "windgust": windgusts_10m,
    "winddirection": winddirection_10m
}

    samples.append(sample)

with open("train_data.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
