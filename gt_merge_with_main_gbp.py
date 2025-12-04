import pandas as pd

# Load your main tourism dataset
tourism_df = pd.read_csv("merged_tourism_weather_updated.csv")

# Load your Google Trends data
trends_df = pd.read_csv("google_trends_srilanka.csv")  # columns: year_month, Sri Lanka tourism, Sri Lanka travel

# Convert year_month in trends_df to separate year and month
trends_df['year'] = pd.to_datetime(trends_df['year_month']).dt.year
trends_df['month'] = pd.to_datetime(trends_df['year_month']).dt.month
trends_df.drop('year_month', axis=1, inplace=True)

# Merge Google Trends into tourism data on year & month
merged_df = pd.merge(tourism_df, trends_df, on=['year','month'], how='left')

# Sort by year and month
merged_df = merged_df.sort_values(by=['year','month']).reset_index(drop=True)

# Save to CSV
merged_df.to_csv("tourism_dataset_with_trends.csv", index=False)

print("Merge complete! File saved as 'tourism_dataset_with_trends.csv'")
