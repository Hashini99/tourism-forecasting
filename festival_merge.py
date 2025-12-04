import pandas as pd

# ✅ Define main Sri Lankan festivals by month
festival_dict = {
    4: "Sinhala & Tamil New Year",
    5: "Vesak",
    6: "Poson",
    8: "Esala Perahera (Kandy)",
    12: "Christmas"
}

# ✅ Load your merged dataset
df = pd.read_csv("merged_tourism_weather.csv")

# ✅ Ensure year and month are numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["month"] = pd.to_numeric(df["month"], errors="coerce")

# ✅ Add a 'festival' column based on the month
df["festival"] = df["month"].map(festival_dict).fillna("None")

# ✅ Reorder columns to keep 'year' and 'month' first
columns = ["year", "month"] + [col for col in df.columns if col not in ["year", "month"]]
df = df[columns]

# ✅ Save updated dataset
output_file = "merged_tourism_weather_with_festivals.csv"
df.to_csv(output_file, index=False)

print(f"✅ Added 'festival' column and saved as {output_file}")
print(df.head())
