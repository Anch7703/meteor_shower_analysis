import pandas as pd

# Load both datasets
features = pd.read_csv("data/processed/combined_features.csv")
sessions = pd.read_csv("data/processed/all_session_data.csv")

# Merge on Session ID (the bridge between the two)
merged = pd.merge(features, sessions, on="Session ID", how="left")

# Save the final enriched dataset
merged.to_csv("data/processed/combined_with_visibility.csv", index=False)

print("🌍 Combined with visibility region data!")
print(f"Final shape: {merged.shape}")
