import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima

# Load dataset
df = pd.read_csv("combined_features_with_location_clean.csv")

# Ensure proper datetime
df["Start_Date"] = pd.to_datetime(df["Start_Date"])

# Filter for Geminids and Lyrids
geminids = df[df["Shower_Name"].str.contains("Geminid", case=False, na=False)].copy()
lyrids = df[df["Shower_Name"].str.contains("Lyrid", case=False, na=False)].copy()

# Sort by date
geminids = geminids.sort_values("Start_Date")
lyrids = lyrids.sort_values("Start_Date")

# Clean function
def clean_data(data, column):
    if data[column].isna().sum() > 0:
        data[column] = data[column].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return data

# Forecast function
def forecast_arima(data, column, forecast_periods=12):
    data = clean_data(data, column)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[[column]])

    model = auto_arima(
        scaled,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    forecast_scaled = model.predict(n_periods=forecast_periods)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))

    # Generate future dates
    last_date = data["Start_Date"].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq="MS")[1:]

    return pd.DataFrame({
        "Start_Date": future_dates,
        f"Forecast_{column}": forecast.flatten()
    })

# Convert normalized data into real-world readable units
def normalize_readable_units(df):
    df["Forecast_Meteor_Count_meteors_per_hr"] = df["Forecast_Meteor_Count"].apply(lambda x: round(x * 45 / df["Forecast_Meteor_Count"].max(), 2))
    df["Forecast_Avg_Magnitude_readable"] = df["Forecast_Avg_Magnitude"].apply(lambda x: round(x * 7 / df["Forecast_Avg_Magnitude"].max(), 2))
    df["Forecast_Peak_Activity_readable"] = df["Forecast_Peak_Activity"].apply(lambda x: round(x * 100 / df["Forecast_Peak_Activity"].max(), 2))
    return df

# Forecast both showers
def forecast_shower(df_shower, shower_name):
    df_shower = df_shower.reset_index(drop=True)
    df_shower = clean_data(df_shower, "Meteor_Count")
    df_shower = clean_data(df_shower, "Avg_Magnitude")
    df_shower = clean_data(df_shower, "Peak_Activity")

    forecast_meteor = forecast_arima(df_shower, "Meteor_Count")
    forecast_brightness = forecast_arima(df_shower, "Avg_Magnitude")
    forecast_activity = forecast_arima(df_shower, "Peak_Activity")

    forecast_df = forecast_meteor.merge(forecast_brightness, on="Start_Date")
    forecast_df = forecast_df.merge(forecast_activity, on="Start_Date")
    forecast_df["Shower_Name"] = shower_name

    # Apply readable units
    forecast_df = normalize_readable_units(forecast_df)
    return forecast_df

# Forecast Geminids and Lyrids
forecast_geminids = forecast_shower(geminids, "Geminids")
forecast_lyrids = forecast_shower(lyrids, "Lyrids")

# Combine everything
forecast_all = pd.concat([forecast_geminids, forecast_lyrids], ignore_index=True)

# Save final file
forecast_all.to_csv("data/processed/meteor_forecasts.csv", index=False)

print("✅ Forecasting complete! File saved as 'meteor_forecasts.csv'")
print(forecast_all.tail())
