import pandas as pd

# Load all CSVs
brightness = pd.read_csv("predicted_brightness.csv")
count = pd.read_csv("predicted_meteor_count.csv")
location = pd.read_csv("predicted_meteor_locations.csv")
peak = pd.read_csv("predicted_peak_activity.csv")

# Merge all horizontally (assuming each has the same number/order of rows)
merged = pd.concat([brightness, count, location, peak], axis=1)

# Save final combined file
merged.to_csv("final_meteor_predictions.csv", index=False)

print("✅ All prediction files merged into final_meteor_predictions.csv")
