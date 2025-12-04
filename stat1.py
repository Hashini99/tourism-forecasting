# full_visual_summary.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load cleaned dataset
# -------------------------------
df = pd.read_csv("tourism_dataset_final_clean.csv")

# -------------------------------
# 2. Ensure numeric columns
# -------------------------------
df['tourist arrival'] = pd.to_numeric(df['tourist arrival'], errors='coerce')
df['USD'] = pd.to_numeric(df['USD'], errors='coerce')
df['Sri Lanka tourism'] = pd.to_numeric(df['Sri Lanka tourism'], errors='coerce')
df['Sri Lanka travel'] = pd.to_numeric(df['Sri Lanka travel'], errors='coerce')

# Fill missing values if needed
df['tourist arrival'].interpolate(inplace=True)
df['USD'].interpolate(inplace=True)
df['Sri Lanka tourism'].interpolate(inplace=True)
df['Sri Lanka travel'].interpolate(inplace=True)

# -------------------------------
# 3. Add datetime column for plotting
# -------------------------------
df['date'] = pd.to_datetime(df[['year','month']].assign(DAY=1))

# -------------------------------
# 4. Print time series summary
# -------------------------------
print("=== Tourist Arrivals Over Time ===")
print(df[['date','tourist arrival']].head(10))

print("\n=== Google Trends Over Time ===")
print(df[['date','Sri Lanka tourism','Sri Lanka travel']].head(10))

print("\n=== USD Exchange Rate Over Time ===")
print(df[['date','USD']].head(10))

# -------------------------------
# 5. Line plot: Tourist Arrivals + Festivals
# -------------------------------
plt.figure(figsize=(14,6))
sns.lineplot(data=df, x='date', y='tourist arrival', label='Tourist Arrival', marker='o')
sns.scatterplot(data=df[df['festival'].notnull()], x='date', y='tourist arrival', hue='festival', s=100, palette='tab10', legend='full')
plt.title("Tourist Arrivals Over Time with Festivals")
plt.xlabel("Date")
plt.ylabel("Tourist Arrivals")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Line plot: Tourist Arrivals vs Google Trends
# -------------------------------
plt.figure(figsize=(14,6))
sns.lineplot(data=df, x='date', y='tourist arrival', label='Tourist Arrival', marker='o')
sns.lineplot(data=df, x='date', y='Sri Lanka tourism', label='Google Trends: Tourism', color='orange')
sns.lineplot(data=df, x='date', y='Sri Lanka travel', label='Google Trends: Travel', color='green')
plt.title("Tourist Arrivals vs Google Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Counts / Search Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Line plot: Tourist Arrivals vs USD
# -------------------------------
plt.figure(figsize=(14,6))
sns.lineplot(data=df, x='date', y='tourist arrival', label='Tourist Arrival', marker='o')
sns.lineplot(data=df, x='date', y='USD', label='USD Exchange Rate', color='red')
plt.title("Tourist Arrivals vs USD Exchange Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Counts / USD")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 8. Monthly seasonality
# -------------------------------
monthly_avg = df.groupby('month')['tourist arrival'].mean()
print("\n=== Average Tourist Arrivals by Month ===")
print(monthly_avg)

plt.figure(figsize=(12,5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette='coolwarm')
plt.title("Average Tourist Arrivals by Month (Seasonality)")
plt.xlabel("Month")
plt.ylabel("Average Tourist Arrivals")
plt.show()
