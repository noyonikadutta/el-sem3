import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def cluster_and_simulate_coords(data_path, output_path):
    df = pd.read_csv(data_path)
    
    features = df[['crop_yield_tons', 'Population']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Simulate lat/lon coordinates within India’s approximate lat/lon bounds
    np.random.seed(42)
    df['lat'] = np.random.uniform(8, 37, size=len(df))   # India approx latitude range
    df['lon'] = np.random.uniform(68, 97, size=len(df))  # India approx longitude range
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    input_path = r"D:\2nd year college\el\food_surplus_deficit\data\processed_data.csv"
    output_path = r"D:\2nd year college\el\food_surplus_deficit\data\clustered_data.csv"
    df = cluster_and_simulate_coords(input_path, output_path)
    print(df.head())
