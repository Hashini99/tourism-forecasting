# sarima_tourist_prediction_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("tourism_dataset_final_clean.csv")

# Ensure clean column names
df.columns = df.columns.str.strip()

# Convert 'tourist arrival' to numeric
if df['tourist arrival'].dtype == 'object':
    df['tourist arrival'] = pd.to_numeric(df['tourist arrival'].str.replace(',', ''), errors='coerce')
else:
    df['tourist arrival'] = pd.to_numeric(df['tourist arrival'], errors='coerce')

df['tourist arrival'] = df['tourist arrival'].interpolate()

# -------------------------------
# 2. Create datetime index
# -------------------------------
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df.set_index('date', inplace=True)

ts = df['tourist arrival']
ts = ts.asfreq('MS')  # ensure monthly start frequency

# -------------------------------
# 3. SARIMA Model
# -------------------------------
model = SARIMAX(ts,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)

# -------------------------------
# 4. Forecast to Sept 2025
# -------------------------------
forecast_date = pd.Timestamp('2025-09-01')
last_date = ts.index[-1]

# Calculate how many months ahead
months_ahead = (forecast_date.year - last_date.year) * 12 + (forecast_date.month - last_date.month)

if months_ahead <= 0:
    # Already have data up to or beyond target month
    predicted_sept2025 = ts.loc[forecast_date] if forecast_date in ts.index else np.nan
    print("\n  September 2025 already exists in the dataset.")
    print(f"Existing recorded tourist arrivals for September 2025: {predicted_sept2025:.0f}")
else:
    forecast = results.get_forecast(steps=months_ahead)
    forecast_mean = forecast.predicted_mean
    predicted_sept2025 = forecast_mean.iloc[-1]
    print("\n============================")
    print(f"Predicted Tourist Arrivals for September 2025 (SARIMA): {int(predicted_sept2025)}")
    print("============================\n")

    # -------------------------------
    # 5. Plot forecast
    # -------------------------------
    forecast_index = pd.date_range(start=last_date + MonthEnd(1), periods=months_ahead, freq='M')

    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Actual')
    plt.plot(forecast_index, forecast_mean, label='Forecast', color='red')
    plt.title("Tourist Arrivals Forecast (SARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Tourist Arrivals")
    plt.legend()
    plt.show()


# ============================================
# 6. Next 6-Month Forecast (after last dataset date)
# ============================================

print("\n=== SARIMA 6-Month Future Forecast ===")

steps = 6  # forecast length

six_month_forecast = results.get_forecast(steps=steps)
six_month_mean = six_month_forecast.predicted_mean

six_month_index = pd.date_range(
    start=ts.index[-1] + MonthEnd(1),
    periods=steps,
    freq='MS'
)

# Print results
for date, value in zip(six_month_index, six_month_mean):
    print(f"{date.strftime('%Y-%m')}: {int(value)} ")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(ts, label="Historical Data")
plt.plot(six_month_index, six_month_mean, label="Next 6-Month Forecast", color='green')

plt.title("SARIMA – Next 6 Months Forecast")
plt.xlabel("Date")
plt.ylabel("Tourist Arrivals")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================
# 4. MODEL EVALUATION (Train–Test Split)
# ============================================



# Use last 12 months for testing
test_size = 12
train = ts[:-test_size]
test = ts[-test_size:]

# Fit SARIMA on training data
eval_model = SARIMAX(train,
                     order=(1,1,1),
                     seasonal_order=(1,1,1,12),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)

# Forecast for test period
test_forecast = eval_model.get_forecast(steps=test_size).predicted_mean
test_forecast.index = test.index  # align index

# Calculate metrics
mae = mean_absolute_error(test, test_forecast)
rmse = np.sqrt(mean_squared_error(test, test_forecast))
r2 = r2_score(test, test_forecast)

print("\n===== SARIMA MODEL EVALUATION =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print("===================================\n")


# Predicted Tourist Arrivals for September 2025 (SARIMA): 171889
# ============================