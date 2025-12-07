import pandas as pd

# Load your dataset
df = pd.read_csv("merged_tourism_weather_with_festivals.csv")

# Filter only festival months (exclude 'None')
festival_df = df[df["festival"] != "None"]

# Show all festival months
print("🎉 Festival months in dataset:")
print(festival_df[["year", "month", "month_name", "festival", "tourist arrival"]])

# Optional — save to a new file
festival_df.to_csv("festival_months_only.csv", index=False)
print("\n Saved filtered file as festival_months_only.csv")
