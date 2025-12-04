import pandas as pd

# Load your merged dataset
df = pd.read_csv("tourism_dataset_with_trends.csv")

# Drop GBP and month_name columns if they exist
df = df.drop(columns=[col for col in ['GBP', 'month_name'] if col in df.columns])

# Reorder columns if needed (optional)
cols_order = ['year', 'month', 'tourist arrival', 'USD', 'GDP (current US$)',
              'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
              'festival','mean_temp','max_temp','min_temp','total_precip',
              'Sri Lanka tourism','Sri Lanka travel']
# Keep only columns that exist in your dataset
cols_order = [col for col in cols_order if col in df.columns]
df = df[cols_order]

# Sort by year and month
df = df.sort_values(by=['year','month']).reset_index(drop=True)

# Save cleaned dataset
df.to_csv("tourism_dataset_clean.csv", index=False)

print("Cleaned dataset saved as 'tourism_dataset_clean.csv'")
