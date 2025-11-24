import pandas as pd

# === Step 1: Read CSV files ===
combined = pd.read_csv("data/processed/combined_features.csv")
session = pd.read_csv("data/processed/all_session_data.csv")

# === Step 2: Strip and clean column names ===
combined.columns = combined.columns.str.strip()
session.columns = session.columns.str.strip()

# === Step 3: Clean and standardize Session ID ===
def clean_id(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    val = val.replace('.0', '')
    val = val.replace('\u200b', '')  # remove invisible characters
    return val

combined['Session ID'] = combined['Session ID'].apply(clean_id)
session['Session ID'] = session['Session ID'].apply(clean_id)

# === Step 4: Debug — check intersections ===
common_ids = set(combined['Session ID']).intersection(set(session['Session ID']))
print(f"🔍 Common Session IDs found: {len(common_ids)}")

if len(common_ids) < 5:
    print("⚠️ Few or no matches — here’s what the first few from each file look like:")
    print("\nCombined IDs sample:", combined['Session ID'].head(10).tolist())
    print("\nSession IDs sample:", session['Session ID'].head(10).tolist())

# === Step 5: Merge ===
merged = pd.merge(combined, session, on='Session ID', how='left', suffixes=('', '_session'))

# === Step 6: Merge results ===
print(f"\n✅ Merge done. Non-empty cities found: {merged['City_session'].notna().sum()} / {len(merged)}")

# === Step 7: Cleanup ===
for col in ['City', 'Country', 'Latitude', 'Longitude', 'Elevation']:
    merged[col] = merged[col].combine_first(merged[f"{col}_session"])
    merged.drop(columns=[f"{col}_session"], inplace=True, errors='ignore')

# === Step 8: Save output ===
merged.to_csv("combined_features_with_location.csv", index=False)
print("💾 Saved as 'combined_features_with_location.csv'")
# Remove duplicate columns
merged = merged.loc[:, ~merged.columns.duplicated()]

# Drop columns that are entirely empty
merged = merged.dropna(axis=1, how='all')

# Drop rows that are completely empty
merged = merged.dropna(how='all')

# Reset index if old one got added as a column
if 'Unnamed: 0' in merged.columns:
    merged = merged.drop(columns=['Unnamed: 0'])

# Save cleaned version
merged.to_csv('combined_features_with_location_clean.csv', index=False)
print("✅ Cleaned file saved as combined_features_with_location_clean.csv")
df = pd.read_csv("combined_features_with_location_clean.csv")
print(df.dtypes)
df['Session ID'].duplicated().sum()
df['Session ID'].isna().sum()
