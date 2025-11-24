import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
df = pd.read_csv('data/processed/combined_features.csv')

# Select features and target
# Select features and target
features = ['Meteor_Count', 'Avg_Magnitude', 'Max_Magnitude', 
            'Min_Magnitude', 'Duration_hrs', 'Activity_Rate']
target = 'Peak_Activity'

X = df[features]
y = df[target]

# Handle infinities and NaNs
X.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
X.fillna(0, inplace=True)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf.predict(X_test_scaled)

# Evaluate
print("🌌 MODEL PERFORMANCE 🌌")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
import matplotlib.pyplot as plt
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.title("Feature Importance in Meteor Peak Prediction")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
# Save predictions to CSV
results_df = pd.DataFrame({
    "Actual_Peak_Activity": y_test,
    "Predicted_Peak_Activity": y_pred
})
results_df.to_csv("predicted_peak_activity.csv", index=False)
print("✅ Saved as predicted_peak_activity.csv")

