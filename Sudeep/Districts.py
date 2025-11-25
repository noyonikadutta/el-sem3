import geopandas as gpd
import pandas as pd
import os

# --- Configuration ---
SHP_FILE_PATH = r"C:\Users\SUDEEP XAVIER ROCHE\Downloads\District\District.shp" 
DISTRICT_NAME_COLUMN = 'KGISDist_1' # Assuming this is the correct name
# ---------------------

def generate_district_centroids_corrected(shp_path, name_col):
    """
    Reads the shapefile, transforms the CRS to WGS84 (Lat/Lon), 
    calculates the centroid, and extracts coordinates in Decimal Degrees.
    """
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found at: {shp_path}")
        
    districts_gdf = gpd.read_file(shp_path)
    print(f"Original CRS: {districts_gdf.crs}")
    
    # 1. 🛑 CRITICAL STEP: Reproject to WGS84 (EPSG:4326) 
    # This converts coordinates from meters to decimal degrees (Lat/Lon).
    districts_gdf_wgs84 = districts_gdf.to_crs(epsg=4326)
    print(f"New CRS: {districts_gdf_wgs84.crs}")
    
    # 2. Calculate the centroid for each district
    # The centroid calculated now will be in Lat/Lon
    districts_gdf_wgs84['centroid'] = districts_gdf_wgs84.geometry.centroid
    
    # 3. Extract X (Longitude) and Y (Latitude) coordinates in decimal degrees
    districts_gdf_wgs84['longitude'] = districts_gdf_wgs84['centroid'].x
    districts_gdf_wgs84['latitude'] = districts_gdf_wgs84['centroid'].y
    
    # 4. Create the final lookup table
    centroid_lookup = districts_gdf_wgs84[[name_col, 'latitude', 'longitude']].copy()
    centroid_lookup.rename(columns={name_col: 'district_name'}, inplace=True)
    
    return centroid_lookup

# Run the corrected function
district_centers_df = generate_district_centroids_corrected(SHP_FILE_PATH, DISTRICT_NAME_COLUMN)

print("\n--- Example of Corrected Coordinates (Should be between -180/180 and -90/90) ---")
print(district_centers_df)

# Save the corrected lookup table
district_centers_df.to_csv('data/district_centroids_wgs84_lookup.csv', index=False)