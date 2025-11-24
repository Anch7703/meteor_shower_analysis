import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("combined_features_with_location_clean.csv")

print("✅ File loaded successfully!")
print("Total rows:", len(df))

# Fill missing numeric values with median or mean
df["Peak_Activity"].fillna(df["Peak_Activity"].median(), inplace=True)
df["Avg_Magnitude"].fillna(df["Avg_Magnitude"].median(), inplace=True)
df["Meteor_Count"].fillna(df["Meteor_Count"].median(), inplace=True)
df["Latitude"].fillna(df["Latitude"].mean(), inplace=True)
df["Longitude"].fillna(df["Longitude"].mean(), inplace=True)

# Check if columns exist
required_cols = ["Meteor_Count", "Peak_Activity", "Avg_Magnitude", "Latitude", "Longitude"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Define features (X) and targets (y)
X = df[["Meteor_Count", "Peak_Activity", "Avg_Magnitude"]]
y_lat = df["Latitude"]
y_lon = df["Longitude"]

# Split dataset
X_train, X_test, y_lat_train, y_lat_test = train_test_split(X, y_lat, test_size=0.2, random_state=42)
_, _, y_lon_train, y_lon_test = train_test_split(X, y_lon, test_size=0.2, random_state=42)

# Train regression models for latitude and longitude
lat_model = RandomForestRegressor(n_estimators=100, random_state=42)
lon_model = RandomForestRegressor(n_estimators=100, random_state=42)

lat_model.fit(X_train, y_lat_train)
lon_model.fit(X_train, y_lon_train)

# Predictions
y_lat_pred = lat_model.predict(X_test)
y_lon_pred = lon_model.predict(X_test)

# Evaluate performance
lat_mae = mean_absolute_error(y_lat_test, y_lat_pred)
lon_mae = mean_absolute_error(y_lon_test, y_lon_pred)

lat_r2 = r2_score(y_lat_test, y_lat_pred)
lon_r2 = r2_score(y_lon_test, y_lon_pred)

print("\n🌍 Model Evaluation:")
print(f"Latitude - MAE: {lat_mae:.2f}, R²: {lat_r2:.2f}")
print(f"Longitude - MAE: {lon_mae:.2f}, R²: {lon_r2:.2f}")

# Store predictions
pred_df = X_test.copy()
pred_df["Actual_Latitude"] = y_lat_test.values
pred_df["Predicted_Latitude"] = y_lat_pred
pred_df["Actual_Longitude"] = y_lon_test.values
pred_df["Predicted_Longitude"] = y_lon_pred

# Save for Power BI visualization
pred_df.to_csv("predicted_meteor_locations.csv", index=False)

print("\n✅ Predictions complete! File saved as 'predicted_meteor_locations.csv'")
print(pred_df.head(10))
