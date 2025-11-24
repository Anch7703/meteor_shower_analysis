import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned data
df = pd.read_csv("combined_features_with_location_clean.csv")

print("✅ Data Loaded Successfully!")
print("Shape:", df.shape)
print("\n🔹 First look:")
print(df.head())

# ---------- 1️⃣ Basic Cleaning ----------
# Replace infinite values and handle missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Activity_Rate', 'Meteor_Count', 'Avg_Magnitude'], inplace=True)

# ---------- 2️⃣ Basic Statistics ----------
print("\n📊 Summary of main numerical columns:")
print(df[['Activity_Rate', 'Meteor_Count', 'Avg_Magnitude', 'Duration_hrs']].describe())

# ---------- 3️⃣ Correlation Heatmap ----------
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Activity_Rate', 'Meteor_Count', 'Avg_Magnitude', 'Duration_hrs', 'Elevation']].corr(),
            annot=True, cmap='magma', fmt='.2f')
plt.title('🔥 Correlation Heatmap — Meteor Features')
plt.show()

# ---------- 4️⃣ Distribution Visuals ----------
plt.figure(figsize=(8, 5))
sns.histplot(df['Avg_Magnitude'], kde=True, bins=30, color='gold')
plt.title("💫 Brightness (Avg Magnitude) Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Meteor_Count'], kde=True, bins=30, color='purple')
plt.title("🌠 Meteor Count Distribution")
plt.show()

# ---------- 5️⃣ Shower Type Analysis ----------
plt.figure(figsize=(10, 5))
top_showers = df['Shower_Name'].value_counts().head(10)
sns.barplot(x=top_showers.index, y=top_showers.values, palette='rocket')
plt.xticks(rotation=45)
plt.title("☄️ Top 10 Most Observed Meteor Showers")
plt.ylabel("Count")
plt.show()

# ---------- 6️⃣ Relationship Between Variables ----------
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Avg_Magnitude', y='Meteor_Count', hue='Shower_Name', data=df, alpha=0.6)
plt.title("✨ Brightness vs Meteor Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Duration_hrs', y='Activity_Rate', data=df, color='teal')
plt.title("🕓 Observation Duration vs Activity Rate")
plt.show()
