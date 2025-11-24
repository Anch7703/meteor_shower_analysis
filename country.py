import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your already merged dataset
merged_df = pd.read_csv("combined_features_with_location_clean.csv")

# Step 1: Fill missing countries as 'Unknown'
merged_df["Country"] = merged_df["Country"].fillna("Unknown")

# Step 2: Country -> Region mapping
continent_mapping = {
    "Europe": ["Slovakia", "Ukraine", "Netherlands", "Germany", "Belarus", "France", 
               "Italy", "Spain", "Poland", "Russia", "Czechia", "Slovenia", "Denmark",
               "Switzerland", "Hungary", "Greece", "Norway", "Sweden", "Austria", "Estonia",
               "United Kingdom", "Belgium", "Malta", "Croatia", "Cyprus", "Romania"],
    "Asia": ["Philippines", "China", "India", "Japan", "Thailand", "Taiwan", "South Korea", "Iran", "Israel", "United Arab Emirates", "Oman"],
    "North America": ["Mexico", "United States", "Canada", "Cuba"],
    "South America": ["Brazil", "Argentina", "Chile"],
    "Africa": ["South Africa", "Egypt", "Nigeria", "Reunion"],
    "Oceania": ["Australia", "New Zealand"],
}

def get_region(country_name):
    for region, countries in continent_mapping.items():
        if country_name in countries:
            return region
    return "Other"  # fallback

# Apply country mapping
merged_df["region"] = merged_df["Country"].apply(get_region)

# Step 3: Fill missing/Other regions using latitude & longitude
def latlong_to_region(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return "Other"
    if -35 <= lat <= 37 and -20 <= lon <= 55:      # Africa
        return "Africa"
    elif 5 <= lat <= 55 and 60 <= lon <= 150:      # Asia
        return "Asia"
    elif 25 <= lat <= 70 and -170 <= lon <= -50:   # North America
        return "North America"
    elif -55 <= lat <= 15 and -80 <= lon <= -30:   # South America
        return "South America"
    elif 35 <= lat <= 72 and -10 <= lon <= 40:     # Europe
        return "Europe"
    elif -50 <= lat <= 0 and 110 <= lon <= 180:    # Oceania
        return "Oceania"
    else:
        return "Other"

# Fill missing/Other regions
merged_df["region"] = merged_df.apply(
    lambda row: latlong_to_region(row["Latitude"], row["Longitude"]) 
    if row["region"]=="Other" else row["region"], axis=1
)

# Step 4: Encode region for ML
le = LabelEncoder()
merged_df["region_encoded"] = le.fit_transform(merged_df["region"])

# Step 5: Save final dataset
merged_df.to_csv("combined_features_with_regions.csv", index=False)

print("✅ Region mapping complete! Sample output:")
print(merged_df[["Obs Session ID", "Country", "region", "region_encoded"]].head())
