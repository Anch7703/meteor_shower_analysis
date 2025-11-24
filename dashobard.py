import pandas as pd 
df = pd.read_csv("combined_features_with_location_clean.csv")
print(df["Shower_Name"].unique())

gem_rows = df[df["Shower_Name"] == "Geminids"]
print("Geminid rows:", gem_rows.shape)


gem_rows = df[df["Shower_Name"] == "Geminids"]
print(gem_rows[["Start_Date", "Meteor_Count", "Duration_hrs"]].head(20))

print(df["Activity_Rate"].describe())
print(df["Activity_Rate"].head(20))
