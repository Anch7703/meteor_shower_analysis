import pandas as pd
import glob

# Load ALL geminids rate files automatically
rate_files = glob.glob("data/raw/geminids/Rate-IMO-VMDB-Year-*-Shower-GEM.csv")

if not rate_files:
    print("No files found. Check your path!")
    exit()

all_rows = []

for f in rate_files:
    print(f"Processing: {f}")
    df = pd.read_csv(f)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Drop unusable rows
    df = df.dropna(subset=['Start Date', 'Number', 'Teff'])

    # Convert datatypes
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['Teff'] = pd.to_numeric(df['Teff'], errors='coerce')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')

    # Filter valid rows
    df = df[(df['Teff'] > 0) & (df['Number'] >= 0)]

    # Calculate hourly meteor rate
    df['Meteor_Count_per_hour'] = df['Number'] / df['Teff']

    # Clean shower name
    df['Shower'] = df['Shower'].str.replace(';', '').str.strip()

    all_rows.append(df)

# Combine EVERYTHING
historical = pd.concat(all_rows, ignore_index=True)

# Save master file
historical.to_csv("cleaned_historical_geminids.csv", index=False)

print("\n✔ DONE! Cleaned dataset created:")
print(historical.head())
print("Total rows:", len(historical))
