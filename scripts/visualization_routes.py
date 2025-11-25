import pandas as pd
import folium

def visualize_routes(shipments_path, map_output_path):
    df = pd.read_csv(shipments_path)
    
    # Create a base map centered roughly on India
    m = folium.Map(location=[22, 79], zoom_start=5)
    
    for _, row in df.iterrows():
        # Supplier and consumer coordinates (you may need to merge lat/lon from original data)
        # For now, assume you have those columns; if not, you can merge or simulate them
        
        supplier_coords = (row['Supplier_Lat'], row['Supplier_Lon'])
        consumer_coords = (row['Consumer_Lat'], row['Consumer_Lon'])
        
        # Draw a line for shipment route
        folium.PolyLine(locations=[supplier_coords, consumer_coords],
                        color='green',
                        weight=2,
                        opacity=0.7,
                        popup=f"{row['Shipment_Tons']:.0f} tons").add_to(m)
        
        # Add markers
        folium.CircleMarker(location=supplier_coords, color='blue', radius=4, popup=row['Supplier_District']).add_to(m)
        folium.CircleMarker(location=consumer_coords, color='red', radius=4, popup=row['Consumer_District']).add_to(m)
    
    m.save(map_output_path)
    print(f"Map saved to {map_output_path}")

if __name__ == "__main__":
    shipments_file = r"D:\2nd year college\el\food_surplus_deficit\data\optimized_shipments_with_latlon.csv"
    map_file = r"D:\2nd year college\el\food_surplus_deficit\data\shipment_map.html"
    visualize_routes(shipments_file, map_file)
