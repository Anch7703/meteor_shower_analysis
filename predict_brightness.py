# predict_brightness.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/processed/combined_features.csv")

# Select relevant features for brightness prediction
features = ['Meteor_Count', 'Activity_Rate', 'Duration_hrs', 'Peak_Activity']
target = 'Avg_Magnitude'

X = data[features]
y = data[target]

# Handle any missing or infinite values
X = X.replace([float('inf'), -float('inf')], pd.NA)
X = X.fillna(X.mean())
y = y.replace([float('inf'), -float('inf')], pd.NA)
y = y.fillna(y.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train regression model
regressor = RandomForestRegressor(n_estimators=150, random_state=42)
regressor.fit(X_train_scaled, y_train)

# Predict brightness
y_pred = regressor.predict(X_test_scaled)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🌠 Brightness Prediction Results 🌠")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# Plot actual vs predicted brightness
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6, color='orange', edgecolor='k')
plt.xlabel("Actual Brightness (Avg Magnitude)")
plt.ylabel("Predicted Brightness")
plt.title("Actual vs Predicted Meteor Brightness")
plt.grid(True)
plt.show()

# Feature importance visualization
importance = regressor.feature_importances_
plt.figure(figsize=(6,4))
plt.bar(features, importance, color='royalblue')
plt.title("Feature Importance in Brightness Prediction")
plt.xticks(rotation=30)
plt.show()
# Save predictions to CSV
# Save predictions to CSV
results_df = pd.DataFrame({
    "Actual_Brightness": y_test,
    "Predicted_Brightness": y_pred
})
results_df.to_csv("predicted_brightness.csv", index=False)
print("✅ Saved as predicted_brightness.csv")
