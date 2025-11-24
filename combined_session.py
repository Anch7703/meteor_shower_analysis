import pandas as pd
import glob
import os

# Folders for both showers
folders = ["data/raw/lyrids", "data/raw/geminids"]

all_sessions = []

for folder in folders:
    for file in glob.glob(os.path.join(folder, "*session*.csv")):  # catches any file with 'session' in name
        print(f"Reading session file: {file}")
        df = pd.read_csv(file, sep=';', encoding='utf-8')
        df['Shower_Name'] = os.path.basename(folder).capitalize()  # Tag which shower it belongs to
        all_sessions.append(df)

# Combine into one big DataFrame
combined_sessions = pd.concat(all_sessions, ignore_index=True)

# Clean the columns a bit
combined_sessions.columns = combined_sessions.columns.str.strip().str.replace('"', '')

# Save cleaned version
combined_sessions.to_csv("data/processed/all_session_data.csv", index=False)

print(f"✨ Combined session data saved to: data/processed/all_session_data.csv")
print(f"Total records: {len(combined_sessions)}")
