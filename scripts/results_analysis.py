import pandas as pd
import numpy as np

def analyze_results(surplus, deficit, allocations_path):
    allocations = np.loadtxt(allocations_path, delimiter=',')
    
    # Total allocated per surplus district
    total_allocated = allocations.sum(axis=1)
    for i, row in surplus.iterrows():
        print(f"Surplus district {row['District_Name']} allocated {total_allocated[i]:.2f} tons")
    
    # Total received per deficit district
    total_received = allocations.sum(axis=0)
    for j, row in deficit.iterrows():
        print(f"Deficit district {row['District_Name']} received {total_received[j]:.2f} tons")
    
if __name__ == "__main__":
    # Load surplus and deficit info
    surplus = pd.read_csv('../data/clustered_data.csv')
    deficit = surplus  # Just placeholders, modify as needed
    
    analyze_results(surplus, deficit, '../data/allocations.csv')
