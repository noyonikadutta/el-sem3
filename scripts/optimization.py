import pandas as pd
import numpy as np
from scipy.optimize import linprog
from geopy.distance import geodesic

def optimize_food_distribution(data_path, output_path):
    df = pd.read_csv(data_path)
    
    # Filter surplus and deficit districts
    suppliers = df[df['Surplus_tons'] > 0].reset_index(drop=True)
    consumers = df[df['Deficit_tons'] > 0].reset_index(drop=True)
    
    n_suppliers = len(suppliers)
    n_consumers = len(consumers)
    
    # Calculate cost matrix (distance in km between each supplier-consumer pair)
    cost_matrix = np.zeros((n_suppliers, n_consumers))
    for i in range(n_suppliers):
        for j in range(n_consumers):
            supplier_coords = (suppliers.loc[i, 'lat'], suppliers.loc[i, 'lon'])
            consumer_coords = (consumers.loc[j, 'lat'], consumers.loc[j, 'lon'])
            cost_matrix[i, j] = geodesic(supplier_coords, consumer_coords).km
    
    # Objective function coefficients (flatten cost matrix)
    c = cost_matrix.flatten()
    
    # Inequality constraints:
    # Supply constraints: each supplier can supply at most its surplus
    A_supply = np.zeros((n_suppliers, n_suppliers * n_consumers))
    for i in range(n_suppliers):
        for j in range(n_consumers):
            A_supply[i, i * n_consumers + j] = 1
    b_supply = suppliers['Surplus_tons'].values
    
    # Equality constraints:
    # Demand constraints: each consumer must receive exactly its deficit
    A_demand = np.zeros((n_consumers, n_suppliers * n_consumers))
    for j in range(n_consumers):
        for i in range(n_suppliers):
            A_demand[j, i * n_consumers + j] = 1
    b_demand = consumers['Deficit_tons'].values
    
    # Bounds: shipments >= 0
    bounds = [(0, None) for _ in range(n_suppliers * n_consumers)]
    
    # Combine constraints:
    A_eq = A_demand
    b_eq = b_demand
    A_ub = A_supply
    b_ub = b_supply
    
    # Solve linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        shipments = result.x.reshape((n_suppliers, n_consumers))
        
        # Prepare output dataframe
        records = []
        for i in range(n_suppliers):
            for j in range(n_consumers):
                if shipments[i, j] > 0:
                    records.append({
                        'Supplier_State': suppliers.loc[i, 'State_Name'],
                        'Supplier_District': suppliers.loc[i, 'District_Name'],
                        'Consumer_State': consumers.loc[j, 'State_Name'],
                        'Consumer_District': consumers.loc[j, 'District_Name'],
                        'Shipment_Tons': shipments[i, j],
                        'Distance_km': cost_matrix[i, j]
                    })
        df_shipments = pd.DataFrame(records)
        df_shipments.to_csv(output_path, index=False)
        print(f"Optimization successful! Results saved to {output_path}")
        return df_shipments
    else:
        print("Optimization failed:", result.message)
        return None

if __name__ == "__main__":
    input_path = r"D:\2nd year college\el\food_surplus_deficit\data\clustered_data.csv"
    output_path = r"D:\2nd year college\el\food_surplus_deficit\data\optimized_shipments.csv"
    shipments_df = optimize_food_distribution(input_path, output_path)
    if shipments_df is not None:
        print(shipments_df.head())
