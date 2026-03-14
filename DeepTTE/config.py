import json
import numpy as np

dist = []
time = []
lngs = []
lats = []
time_gap = []
dist_gap = []

arrival_delay = []
departure_delay = []
speed = []
is_peak_hour = []

temperature = []
apparent_temperature = []
precipitation = []
rain = []
snowfall = []
windspeed = []
windgust = []
winddirection = []

with open("/Users/joehannacansino/Desktop/THS3/DeepTTE_addt_feat/data/train_data.jsonl") as f:
    for line in f:
        d = json.loads(line)

        dist.append(d["dist"])
        time.append(d["time"])

        lngs.extend(d["lngs"])
        lats.extend(d["lats"])
        time_gap.extend(d["time_gap"])
        dist_gap.extend(d["dist_gap"])

        arrival_delay.extend(d["arrival_delay"])
        departure_delay.extend(d["departure_delay"])
        speed.extend(d["speed"])
        is_peak_hour.extend(d["is_peak_hour"])

        temperature.extend(d["temperature"])
        apparent_temperature.extend(d["apparent_temperature"])
        precipitation.extend(d["precipitation"])
        rain.extend(d["rain"])
        snowfall.extend(d["snowfall"])
        windspeed.extend(d["windspeed"])
        windgust.extend(d["windgust"])
        winddirection.extend(d["winddirection"])


config = {

    "dist_gap_mean": np.mean(dist_gap),
    "dist_gap_std": np.std(dist_gap),

    "time_gap_mean": np.mean(time_gap),
    "time_gap_std": np.std(time_gap),

    "lngs_mean": np.mean(lngs),
    "lngs_std": np.std(lngs),

    "lats_mean": np.mean(lats),
    "lats_std": np.std(lats),

    "dist_mean": np.mean(dist),
    "dist_std": np.std(dist),

    "time_mean": np.mean(time),
    "time_std": np.std(time),

    # operational features
    "arrival_delay_mean": np.mean(arrival_delay),
    "arrival_delay_std": np.std(arrival_delay),

    "departure_delay_mean": np.mean(departure_delay),
    "departure_delay_std": np.std(departure_delay),

    "speed_mean": np.mean(speed),
    "speed_std": np.std(speed),

    "is_peak_hour_mean": np.mean(is_peak_hour),
    "is_peak_hour_std": np.std(is_peak_hour),

    # weather features
    "temperature_mean": np.mean(temperature),
    "temperature_std": np.std(temperature),

    "apparent_temperature_mean": np.mean(apparent_temperature),
    "apparent_temperature_std": np.std(apparent_temperature),

    "precipitation_mean": np.mean(precipitation),
    "precipitation_std": np.std(precipitation),

    "rain_mean": np.mean(rain),
    "rain_std": np.std(rain),

    "snowfall_mean": np.mean(snowfall),
    "snowfall_std": np.std(snowfall),

    "windspeed_mean": np.mean(windspeed),
    "windspeed_std": np.std(windspeed),

    "windgust_mean": np.mean(windgust),
    "windgust_std": np.std(windgust),

    "winddirection_mean": np.mean(winddirection),
    "winddirection_std": np.std(winddirection)
}

with open("/Users/joehannacansino/Desktop/THS3/DeepTTE_addt_feat/config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Config file updated successfully.")
print(config)