import pandas as pd

def add_lat_lon_to_shipments(shipments_path, clustered_path, output_path):
    shipments = pd.read_csv(shipments_path)
    clustered = pd.read_csv(clustered_path)
    
    # Rename lat/lon columns to join on suppliers
    suppliers = clustered[['State_Name', 'District_Name', 'lat', 'lon']].copy()
    suppliers.columns = ['Supplier_State', 'Supplier_District', 'Supplier_Lat', 'Supplier_Lon']
    
    consumers = clustered[['State_Name', 'District_Name', 'lat', 'lon']].copy()
    consumers.columns = ['Consumer_State', 'Consumer_District', 'Consumer_Lat', 'Consumer_Lon']
    
    # Merge supplier coordinates
    shipments = shipments.merge(suppliers, on=['Supplier_State', 'Supplier_District'], how='left')
    
    # Merge consumer coordinates
    shipments = shipments.merge(consumers, on=['Consumer_State', 'Consumer_District'], how='left')
    
    shipments.to_csv(output_path, index=False)
    print(f"Saved shipments with lat/lon to {output_path}")
    return shipments

if __name__ == "__main__":
    shipments_path = r"D:\2nd year college\el\food_surplus_deficit\data\optimized_shipments.csv"
    clustered_path = r"D:\2nd year college\el\food_surplus_deficit\data\clustered_data.csv"
    output_path = r"D:\2nd year college\el\food_surplus_deficit\data\optimized_shipments_with_latlon.csv"
    
    add_lat_lon_to_shipments(shipments_path, clustered_path, output_path)
