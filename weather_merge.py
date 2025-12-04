


# import pandas as pd

# # === Load both datasets ===
# tourism = pd.read_csv("tourism_data.csv")
# weather = pd.read_csv("sri_lanka_monthly_weather.csv")

# # === Clean column names ===
# tourism.columns = tourism.columns.str.strip()
# weather.columns = weather.columns.str.strip()

# # === Extract year and month ===
# if 'month' not in tourism.columns or 'year' not in tourism.columns:
#     tourism[['year', 'month_name']] = tourism.iloc[:, 0].astype(str).str.extract(r'(\d{4})\s*([A-Za-z]+)')
#     month_map = {
#         'January': 1, 'February': 2, 'March': 3, 'April': 4,
#         'May': 5, 'June': 6, 'July': 7, 'August': 8,
#         'September': 9, 'October': 10, 'November': 11, 'December': 12
#     }
#     tourism['month'] = tourism['month_name'].map(month_map)
#     tourism['year'] = tourism['year'].astype(int)

# # === Merge ===
# merged = pd.merge(tourism, weather, on=["year", "month"], how="inner")

# # === Reorder columns: year, month first ===
# cols = list(merged.columns)
# # move 'year' and 'month' to front
# ordered_cols = ['year', 'month'] + [c for c in cols if c not in ['year', 'month']]
# merged = merged[ordered_cols]

# # === Save ===
# merged.to_csv("merged_tourism_weather.csv", index=False)
# print("✅ Merged file saved as: merged_tourism_weather.csv")
# print(merged.head())



import pandas as pd

# --- Load both CSV files ---
weather_df = pd.read_csv("sri_lanka_monthly_weather_countrywide.csv")
merged_df = pd.read_csv("merged_tourism_weather_with_festivals.csv")

# --- Clean up column names ---
weather_df.columns = weather_df.columns.str.strip()
merged_df.columns = merged_df.columns.str.strip()

# --- Ensure same data types for merging ---
weather_df["month"] = weather_df["month"].astype(int)
merged_df["month"] = merged_df["month"].astype(int)
weather_df["year"] = weather_df["year"].astype(int)
merged_df["year"] = merged_df["year"].astype(int)

# --- Drop old weather columns in merged dataset ---
cols_to_drop = ["mean_temp", "max_temp", "min_temp", "total_precip"]
merged_df = merged_df.drop(columns=cols_to_drop, errors="ignore")

# --- Merge using 'year' and 'month' ---
updated_df = pd.merge(
    merged_df,
    weather_df,
    on=["year", "month"],
    how="left"
)

# --- Save the result ---
updated_df.to_csv("merged_tourism_weather_updated.csv", index=False)

print("✅ Merge completed successfully! File saved as 'merged_tourism_weather_updated.csv'")
