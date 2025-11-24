import pandas as pd
import os
from pathlib import Path

# Define paths
base_dir = Path(__file__).resolve().parent
raw_dir = base_dir / "data" / "raw"
processed_dir = base_dir / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

def merge_csvs(input_dir, output_file, sep=";"):
    """
    Merge all CSVs from input_dir into a single CSV.
    """
    all_files = list(Path(input_dir).glob("*.csv"))
    if not all_files:
        print(f"⚠️ No CSV files found in {input_dir}")
        return
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, sep=sep)
            print(f"✅ Loaded {f.name} with shape {df.shape}")
            df_list.append(df)
        except Exception as e:
            print(f"❌ Error reading {f.name}: {e}")

    if df_list:
        merged = pd.concat(df_list, ignore_index=True)
        merged.to_csv(output_file, index=False)
        print(f"📁 Saved merged CSV -> {output_file} ({merged.shape[0]} rows, {merged.shape[1]} cols)")
    else:
        print(f"⚠️ No data merged for {output_file}")

# Merge Geminids
merge_csvs(raw_dir / "geminids", processed_dir / "geminids_master.csv")

# Merge Lyrids
merge_csvs(raw_dir / "lyrids", processed_dir / "lyrids_master.csv")
