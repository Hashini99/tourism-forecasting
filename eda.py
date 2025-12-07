
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("tourism_dataset_clean.csv")  

# -------------------------------
# 2. Initial exploration
# -------------------------------
print("=== Dataset Info ===")
print(df.info())
print("\n=== First 5 Rows ===")
print(df.head())
print("\n=== Descriptive Stats ===")
print(df.describe(include='all'))

# -------------------------------
# 3. Data Cleaning
# -------------------------------
if 'month_name' in df.columns:
    df.drop('month_name', axis=1, inplace=True)

df['tourist arrival'] = pd.to_numeric(df['tourist arrival'].str.replace(',', ''), errors='coerce')
df['tourist arrival'] = df['tourist arrival'].interpolate()

numeric_cols = ['USD', 'GDP (current US$)',
                'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
                'mean_temp', 'max_temp', 'min_temp', 'total_precip',
                'Sri Lanka tourism', 'Sri Lanka travel']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[numeric_cols] = df[numeric_cols].interpolate()

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# -------------------------------
# 4. Basic Statistics & Distribution
# -------------------------------
for col in ['tourist arrival', 'USD'] + ['Sri Lanka tourism', 'Sri Lanka travel']:
    print(f"\n=== Statistics for {col} ===")
    print(df[col].describe())
    print(f"\nTop 5 values for {col}:\n{df[col].head()}")
    plt.figure(figsize=(10,4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# -------------------------------
# 5. Time Series Plots
# -------------------------------
df['date'] = pd.to_datetime(df[['year','month']].assign(DAY=1))
for col in ['tourist arrival', 'Sri Lanka tourism', 'Sri Lanka travel']:
    print(f"\n=== Time Series Summary for {col} ===")
    print(df.groupby(['year','month'])[col].sum())
    plt.figure(figsize=(12,5))
    sns.lineplot(x='date', y=col, data=df)
    plt.title(f"{col} Over Time")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.show()

# -------------------------------
# 6. Correlation Analysis
# -------------------------------
numeric_df = df.select_dtypes(include='number')
print("\n=== Correlation Matrix ===")
corr_matrix = numeric_df.corr()
print(corr_matrix)
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 7. Boxplots for Outliers
# -------------------------------
if 'festival' in df.columns:
    print("\n=== Tourist Arrivals by Festival ===")
    print(df.groupby('festival')['tourist arrival'].describe())
    plt.figure(figsize=(12,5))
    sns.boxplot(x='festival', y='tourist arrival', data=df)
    plt.xticks(rotation=45)
    plt.title("Tourist Arrivals by Festival")
    plt.show()

print("\n=== Boxplots of Numeric Columns ===")
plt.figure(figsize=(12,6))
sns.boxplot(data=numeric_df)
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Columns")
plt.show()

# -------------------------------
# 8. Weather Impact on Tourist Arrivals
# -------------------------------
weather_cols = ['mean_temp', 'max_temp', 'min_temp', 'total_precip']
print("\n=== Weather Impact Analysis ===")
for col in weather_cols:
    correlation = df['tourist arrival'].corr(df[col])
    print(f"Correlation between tourist arrival and {col}: {correlation:.3f}")
    
    # Scatter plot
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df[col], y=df['tourist arrival'])
    plt.title(f"Tourist Arrivals vs {col} (Correlation: {correlation:.2f})")
    plt.xlabel(col)
    plt.ylabel('Tourist Arrival')
    plt.show()
    
    # Time series comparison
    plt.figure(figsize=(12,5))
    sns.lineplot(x='date', y=col, data=df, label=col)
    sns.lineplot(x='date', y='tourist arrival', data=df, label='Tourist Arrival', color='red')
    plt.title(f"Tourist Arrival vs {col} over Time")
    plt.show()

# -------------------------------
# 9. Save cleaned & processed dataset
# -------------------------------
# df.to_csv("tourism_dataset_final_clean.csv", index=False)


# -------------------------------
# 10. Additional Plots for Report Figures
# -------------------------------



# === Figure 2.3: Tourist Arrivals vs Google Trends Index ===
plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='tourist arrival', data=df, label='Tourist Arrivals', color='blue')
sns.lineplot(x='date', y='Sri Lanka tourism', data=df, label='Google Trends: "Sri Lanka tourism"', color='orange')
sns.lineplot(x='date', y='Sri Lanka travel', data=df, label='Google Trends: "Sri Lanka travel"', color='green')
plt.title(" Time Series of Tourist Arrivals vs Google Trends Index")
plt.xlabel("Date")
plt.ylabel("Value (Scaled)")
plt.legend()
plt.tight_layout()
plt.show()

# === Figure 2.4: Monthly Seasonal Decomposition ===
# Ensure the data is sorted and has a DateTime index
df_sorted = df.sort_values('date').set_index('date')
result = seasonal_decompose(df_sorted['tourist arrival'], model='additive', period=12)

plt.figure(figsize=(12,8))
result.plot()
plt.suptitle(" Monthly Seasonal Decomposition (Trend + Seasonality + Residual)", fontsize=14)
plt.tight_layout()
plt.show()

# === Figure 2.5: Boxplot of Arrivals by Month ===
plt.figure(figsize=(10,6))
sns.boxplot(x='month', y='tourist arrival', data=df, palette='Set3')
plt.title("Figure 2.5: Boxplot of Tourist Arrivals by Month (Seasonal Peaks)")
plt.xlabel("Month")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()

# === Figure 2.6: Pair Plot (Arrivals vs Economic Indicators) ===
econ_vars = ['tourist arrival', 'USD',
             'GDP (current US$)',
             'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))']
sns.pairplot(df[econ_vars], diag_kind='kde', corner=True)
plt.suptitle("Pair Plot – Tourist Arrivals vs Economic Indicators", y=1.02)
plt.show()

# === Line Graph of LKR/USD vs Time ===
plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='USD', data=df, color='purple')
plt.title("Line Graph of Exchange Rate (LKR/USD) Over Time")
plt.xlabel("Date")
plt.ylabel("Exchange Rate (LKR per USD)")
plt.tight_layout()
plt.show()

# === Correlation Plot: Exchange Rate vs Tourist Arrivals ===
corr_usd_arrival = df['tourist arrival'].corr(df['USD'])
print(f"\nCorrelation between Tourist Arrivals and Exchange Rate (LKR/USD): {corr_usd_arrival:.3f}")

plt.figure(figsize=(8,5))
sns.scatterplot(x='USD', y='tourist arrival', data=df, color='teal')
sns.regplot(x='USD', y='tourist arrival', data=df, scatter=False, color='red')
plt.title(f"Correlation between Exchange Rate and Tourist Arrivals (r = {corr_usd_arrival:.2f})")
plt.xlabel("Exchange Rate (LKR per USD)")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()



# -------------------------------
# 8B. Weather Impact Visuals (Rainfall & Temperature)
# -------------------------------

# === Dual-axis Line Chart: Rainfall vs Tourist Arrivals ===
fig, ax1 = plt.subplots(figsize=(12,6))

color1 = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Tourist Arrivals', color=color1)
ax1.plot(df['date'], df['tourist arrival'], color=color1, label='Tourist Arrivals')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Total Rainfall (mm)', color=color2)
ax2.plot(df['date'], df['total_precip'], color=color2, linestyle='--', label='Rainfall (mm)')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Rainfall vs Tourist Arrivals (Dual-Axis Line Chart)', fontsize=12)
fig.tight_layout()
plt.show()

# === Scatter Plot: Rainfall vs Tourist Arrivals ===
corr_rain_arrival = df['tourist arrival'].corr(df['total_precip'])
print(f"\nCorrelation between Tourist Arrivals and Rainfall: {corr_rain_arrival:.3f}")

plt.figure(figsize=(8,5))
sns.scatterplot(x='total_precip', y='tourist arrival', data=df, color='blue')
sns.regplot(x='total_precip', y='tourist arrival', data=df, scatter=False, color='red')
plt.title(f"Scatter Plot: Rainfall vs Tourist Arrivals (r = {corr_rain_arrival:.2f})")
plt.xlabel("Total Rainfall (mm)")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()

# === Scatter Plot: Mean Temperature vs Tourist Arrivals ===
corr_temp_arrival = df['tourist arrival'].corr(df['mean_temp'])
print(f"Correlation between Tourist Arrivals and Mean Temperature: {corr_temp_arrival:.3f}")

plt.figure(figsize=(8,5))
sns.scatterplot(x='mean_temp', y='tourist arrival', data=df, color='orange')
sns.regplot(x='mean_temp', y='tourist arrival', data=df, scatter=False, color='red')
plt.title(f"Scatter Plot: Mean Temperature vs Tourist Arrivals (r = {corr_temp_arrival:.2f})")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()
corr_usd_arrival = df['tourist arrival'].corr(df['USD'])
print(f"Correlation between Tourist Arrivals and Exchange Rate (LKR/USD): {corr_usd_arrival:.3f}")

# : Line Plot of Exchange Rate (LKR/USD) over Time

plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='USD', data=df, color='purple')
plt.title("Exchange Rate (LKR/USD) Over Time")
plt.xlabel("Date")
plt.ylabel("Exchange Rate (LKR per USD)")
plt.tight_layout()
plt.show()


# Correlation Plot (Exchange Rate vs Tourist Arrivals)
plt.figure(figsize=(8,5))
sns.scatterplot(x='USD', y='tourist arrival', data=df, color='teal')
sns.regplot(x='USD', y='tourist arrival', data=df, scatter=False, color='red')
plt.title(f"Exchange Rate vs Tourist Arrivals (r = {corr_usd_arrival:.2f})")
plt.xlabel("Exchange Rate (LKR per USD)")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()


print("Cleaned dataset saved as 'tourism_dataset_final_clean.csv'")
