import pandas as pd
import os

# Path setup
base_path = "data/processed"
showers = ["lyrids_master.csv", "geminids_master.csv"]

def extract_features(file_name):
    path = os.path.join(base_path, file_name)
    df = pd.read_csv(path)

    # Convert dates to datetime
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df['End_Date'] = pd.to_datetime(df['End_Date'])

    # Duration in hours
    df['Duration_hrs'] = (df['End_Date'] - df['Start_Date']).dt.total_seconds() / 3600

    # Extract magnitude columns
    mag_cols = [col for col in df.columns if 'Mag' in col]

    # Feature extraction
    df['Avg_Magnitude'] = df[mag_cols].mean(axis=1)
    df['Max_Magnitude'] = df[mag_cols].max(axis=1)
    df['Min_Magnitude'] = df[mag_cols].min(axis=1)
    df['Meteor_Count'] = df[mag_cols].sum(axis=1)

    # Derived features
    df['Activity_Rate'] = df['Meteor_Count'] / df['Duration_hrs']
    threshold = df['Meteor_Count'].mean() + df['Meteor_Count'].std()
    df['Peak_Activity'] = (df['Meteor_Count'] > threshold).astype(int)

    # Add shower name column
    shower_name = file_name.split('_')[0].capitalize()
    df['Shower_Name'] = shower_name

    return df

# Process both datasets
features = [extract_features(f) for f in showers]
combined_df = pd.concat(features, ignore_index=True)

# Save individual + combined
combined_df.to_csv(os.path.join(base_path, "combined_features.csv"), index=False)

print("✅ Feature extraction complete for both Lyrids & Geminids!")
print("Saved combined file as: data/processed/combined_features.csv")
print(combined_df.head())


# Load processed features
data = pd.read_csv("data/processed/combined_features.csv")
