# sarima_tourist_prediction_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd

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



# Predicted Tourist Arrivals for September 2025 (SARIMA): 171889
# ============================