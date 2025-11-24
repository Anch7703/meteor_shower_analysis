import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# LOAD YOUR FILES
# -------------------------------
hist = pd.read_csv("combined_features_with_location_clean.csv")
forecast = pd.read_csv("data/processed/meteor_forecasts.csv")

# -------------------------------
# FIX DATE COLUMNS
# -------------------------------
hist["Start_Date"] = pd.to_datetime(hist["Start_Date"])
forecast["Start_Date"] = pd.to_datetime(forecast["Start_Date"])

# -------------------------------
# CLEAN HISTORICAL DATA
# -------------------------------
# Keep only valid meteor activity rates (avoid corrupted large numbers)
hist = hist[(hist["Activity_Rate"] > 0) & (hist["Activity_Rate"] < 500)]

# Use Activity_Rate as actual meteor/hour value
hist["Historical_per_hr"] = hist["Activity_Rate"]

# Only Geminids & Lyrids
hist = hist[hist["Shower_Name"].isin(["Geminids", "Lyrids"])]
forecast = forecast[forecast["Shower_Name"].isin(["Geminids", "Lyrids"])]

# Rename forecast column
forecast = forecast.rename(columns={
    "Forecast_Meteor_Count_meteors_per_hr": "Forecast_per_hr"
})

# -------------------------------
# SPLIT BY SHOWER
# -------------------------------
hist_gem = hist[hist["Shower_Name"] == "Geminids"]
hist_lyr = hist[hist["Shower_Name"] == "Lyrids"]

fore_gem = forecast[forecast["Shower_Name"] == "Geminids"]
fore_lyr = forecast[forecast["Shower_Name"] == "Lyrids"]

# -------------------------------
# FUNCTION TO PLOT EACH SHOWER
# -------------------------------
def plot_shower(hist, fore, name, color):
    plt.figure(figsize=(15,8))

    # Historical line
    plt.plot(
        hist["Start_Date"], hist["Historical_per_hr"],
        color=color, linewidth=2.5, label=f"{name} – Historical"
    )

    # Forecast line (dashed)
    plt.plot(
        fore["Start_Date"], fore["Forecast_per_hr"],
        color=color, linestyle="--", linewidth=2.5, label=f"{name} – Forecast"
    )

    plt.title(f"{name} – Meteor Count per Hour (Historical vs Forecast)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Meteor Count (per hour)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f"{name}_Historical_vs_Forecast.png"
    plt.savefig(filename, dpi=350)
    plt.show()

    print(f"✨ Saved: {filename}")


# -------------------------------
# GENERATE BOTH GRAPHS
# -------------------------------
plot_shower(hist_gem, fore_gem, "Geminids", "gold")
plot_shower(hist_lyr, fore_lyr, "Lyrids", "cyan")
