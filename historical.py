import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. LOAD BOTH DATASETS
# ============================

# Historical cleaned dataset
df = pd.read_csv("combined_features_with_location_clean.csv")

# Forecast dataset
forecast = pd.read_csv("data/processed/meteor_forecasts.csv")

# ============================
# 2. FIX DATE FORMATS
# ============================

df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors='coerce')
forecast["Start_Date"] = pd.to_datetime(forecast["Start_Date"], errors='coerce')

# ============================
# 3. CLEAN HISTORICAL DATA
# ============================

# Create meteor count per hour
df["Meteor_Count_per_hour"] = df["Meteor_Count"] / df["Duration_hrs"]

# Remove invalid values
df = df[
    (df["Meteor_Count_per_hour"] > 0) &
    (df["Meteor_Count_per_hour"] < 200)  # realistic meteor rates
]

# ============================
# 4. SPLIT HISTORICAL DATA BY SHOWER
# ============================

gem_hist = df[df["Shower_Name"].str.contains("Geminid", case=False, na=False)]
lyr_hist = df[df["Shower_Name"].str.contains("Lyrid", case=False, na=False)]

# ============================
# 5. SPLIT FORECAST DATA BY SHOWER
# ============================

gem_fore = forecast[forecast["Shower_Name"] == "Geminids"]
lyr_fore = forecast[forecast["Shower_Name"] == "Lyrids"]

# ============================
# 6. PLOTTING — GEMINIDS
# ============================

plt.figure(figsize=(14, 7))

# Historical
plt.plot(
    gem_hist["Start_Date"],
    gem_hist["Meteor_Count_per_hour"],
    color="gold",
    linewidth=2,
    label="Geminids – Historical"
)

# Forecast
plt.plot(
    gem_fore["Start_Date"],
    gem_fore["Forecast_Meteor_Count_meteors_per_hr"],
    color="gold",
    linestyle="--",
    linewidth=2,
    label="Geminids – Forecast"
)

plt.title("Geminids – Meteor Count per Hour (Historical vs Forecast)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Meteor Count (per hour)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ============================
# 7. PLOTTING — LYRIDS
# ============================

plt.figure(figsize=(14, 7))

# Historical
plt.plot(
    lyr_hist["Start_Date"],
    lyr_hist["Meteor_Count_per_hour"],
    color="cyan",
    linewidth=2,
    label="Lyrids – Historical"
)

# Forecast
plt.plot(
    lyr_fore["Start_Date"],
    lyr_fore["Forecast_Meteor_Count_meteors_per_hr"],
    color="cyan",
    linestyle="--",
    linewidth=2,
    label="Lyrids – Forecast"
)

plt.title("Lyrids – Meteor Count per Hour (Historical vs Forecast)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Meteor Count (per hour)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
