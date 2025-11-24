import pandas as pd

combined = pd.read_csv("data/processed/combined_features.csv")
session = pd.read_csv("data/processed/all_session_data.csv")

# normalize column names
combined.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
session.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

# fix the datatype mismatch
combined["Session_ID"] = combined["Session_ID"].astype(str).str.replace(".0", "", regex=False)
session["Session_ID"] = session["Session_ID"].astype(str).str.strip()

# merge them on the fixed Session_ID
merged = pd.merge(combined, session, on="Session_ID", how="left", suffixes=("", "_session"))

merged.to_csv("data/processed/combined_features_with_location.csv", index=False)

print("✅ Merge finally fixed! Rows merged:", merged["City_session"].notna().sum())
