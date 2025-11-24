import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load data
merged_df = pd.read_csv("combined_features_with_regions.csv")

# Create geometry column
gdf = gpd.GeoDataFrame(
    merged_df, 
    geometry=gpd.points_from_xy(merged_df.Longitude, merged_df.Latitude),
    crs="EPSG:4326"
)

# Load world map
url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
world = gpd.read_file(url)

# Assign manual colors
color_dict = {'Geminids': 'blue', 'Lyrids': 'pink'}
gdf['color'] = gdf['Shower_Name'].map(color_dict)

# Plot
fig, ax = plt.subplots(figsize=(20,10))
world.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plot meteors with manual colors
gdf.plot(ax=ax, color=gdf['color'], markersize=50, alpha=0.7, edgecolor='k')

# Add a legend manually
for shower, color in color_dict.items():
    ax.scatter([], [], c=color, alpha=0.7, s=50, label=shower)
ax.legend(title='Meteor Shower')

plt.title("Meteor Observations: Geminids vs Lyrids", fontsize=20)
plt.show()
