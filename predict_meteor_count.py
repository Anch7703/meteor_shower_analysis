# predict_meteor_count.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np 

# Load data
data = pd.read_csv("data/processed/combined_features.csv")

# Features and target
X = data[['Avg_Magnitude', 'Activity_Rate', 'Duration_hrs', 'Peak_Activity']]
y = data['Meteor_Count']

# Handle missing values if any
X = X.replace([np.inf, -np.inf], np.nan)   # replace infinities with NaN
X = X.fillna(X.mean())                     # fill NaNs with column means
y = y.fillna(y.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Double-check after split (safety measure)
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())     # use train’s mean for consistency

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🌠 Meteor Count Prediction Results 🌠")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='royalblue', alpha=0.7)
plt.xlabel("Actual Meteor Count")
plt.ylabel("Predicted Meteor Count")
plt.title("Actual vs Predicted Meteor Count")
plt.grid(True)
plt.show()
# Save predictions to CSV
results_df = pd.DataFrame({
    "Actual_Meteor_Count": y_test,
    "Predicted_Meteor_Count": y_pred
})
results_df.to_csv("predicted_meteor_count.csv", index=False)
print("✅ Saved as predicted_meteor_count.csv")
