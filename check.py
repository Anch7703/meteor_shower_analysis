import pandas as pd
import numpy as np

lyr = pd.read_csv("data/processed/lyrids_master.csv", low_memory=False)
gem = pd.read_csv("data/processed/geminids_master.csv", low_memory=False)

# parse dates
for df in (lyr, gem):
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['End Date']   = pd.to_datetime(df['End Date'], errors='coerce')

# detect mag columns
mag_cols = [c for c in lyr.columns if c.startswith('Mag') or 'Mag' in c]
mag_cols[:10], len(mag_cols)
