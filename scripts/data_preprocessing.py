import pandas as pd
import numpy as np

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # your existing processing here...
    # Example:
    df_yield = df.groupby(['State_Name', 'District_Name']).agg({
        'crop_yield_tons': 'sum',
        'Area': 'sum'
    }).reset_index()

    np.random.seed(42)
    df_yield['Population'] = np.random.randint(500000, 3000000, size=len(df_yield))

    df_yield['Demand'] = df_yield['Population'] * 0.5
    df_yield['Surplus'] = df_yield['crop_yield_tons'] - df_yield['Demand']
    df_yield['Surplus_tons'] = df_yield['Surplus'].clip(lower=0)
    df_yield['Deficit_tons'] = (-df_yield['Surplus']).clip(lower=0)

    return df_yield

if __name__ == "__main__":
    filepath = r"D:\2nd year college\el\food_surplus_deficit\data\crop_data.csv"
    df_processed = preprocess_data(filepath)
    print(df_processed.head())
    df_processed.to_csv(r"D:\2nd year college\el\food_surplus_deficit\data\processed_data.csv", index=False)
