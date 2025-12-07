

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------
#  Reproducibility (Seed)
# -------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Load & Clean Data
# -------------------------
data = pd.read_csv("tourism_dataset_final_clean.csv")
data.columns = [c.strip() for c in data.columns]

numeric_cols = [
    'tourist arrival','USD','GDP (current US$)',
    'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
    'mean_temp','max_temp','min_temp','total_precip',
    'Sri Lanka tourism','Sri Lanka travel'
]
count = len(data)
print(count)

count = data.shape[0]
print(count)


for col in numeric_cols:
    if col in data.columns:
        data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

# Festival  One-Hot Encode
if 'festival' in data.columns:
    data['festival'] = data['festival'].fillna('None')
    data = pd.get_dummies(data, columns=['festival'], drop_first=True)

# Create Date column
data['date'] = pd.to_datetime(data[['year','month']].assign(day=1))
data.sort_values('date', inplace=True)
data.set_index('date', inplace=True)

# -------------------------
#  Feature & Target Split
# -------------------------
target_col = "tourist arrival"
y = data[target_col].values.reshape(-1,1)
X = data.drop(columns=['year','month',target_col], errors='ignore')

# -------------------------
#  Train / Test Split
# -------------------------
train_size = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
y_train_raw, y_test_raw = y[:train_size], y[train_size:]

# -------------------------
# Scaling
# -------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# -------------------------
#  Sequence Conversion
# -------------------------
time_steps = 12  # 1 year history

def create_sequences(X_data, y_data):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:i+time_steps])
        ys.append(y_data[i+time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled)

# -------------------------
#  LSTM Model
# -------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# -------------------------
#  Train
# -------------------------
history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# -------------------------
#  Predictions
# -------------------------
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# -------------------------
# 9 Evaluation Metrics
# -------------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n=== LSTM Performance (Test Set) ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# -------------------------
# Plot Actual vs Predicted
# -------------------------
test_index = data.index[train_size+time_steps:]
plt.figure(figsize=(12,6))
plt.plot(test_index, y_true, label="Actual")
plt.plot(test_index, y_pred, label="Predicted")
plt.title("Actual vs Predicted Tourist Arrivals")
plt.xlabel("Date")
plt.ylabel("Tourist Arrivals")
plt.legend()
plt.show()

# -------------------------
# 1️⃣1️⃣ Forecast 6 Months Ahead
# -------------------------
# 11. Forecast Next 6 Months (FIXED)
future_steps = 6
last_seq = X_test_scaled[-time_steps:]

forecast_scaled = []
num_features = X_train.shape[2]  # total input features

for _ in range(future_steps):
    pred = model.predict(last_seq.reshape(1, time_steps, num_features), verbose=0)

    # Create placeholder feature vector (copy last state)
    new_row = last_seq[-1].copy()
    new_row[0] = pred  # replace the target index (0) with prediction

    forecast_scaled.append(pred[0])
    last_seq = np.vstack([last_seq[1:], new_row])

future = scaler_y.inverse_transform(np.array(forecast_scaled))


future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1),
                             periods=future_steps, freq='MS')

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Arrivals": future.flatten()
})

print("\n=== Future Forecast (Next 6 Months) ===")
print(forecast_df)

plt.figure(figsize= (10,5))
plt.plot(forecast_df["Date"], forecast_df["Forecast_Arrivals"], marker="o")
plt.title("6-Month Forecast: Tourist Arrivals")
plt.xlabel("Date")
plt.ylabel("Forecasted Arrivals")
plt.show()


# -------------------------
#  Actual vs Predicted Plot (Clean)
# -------------------------

plt.figure(figsize=(12,6))
plt.plot(test_index, y_true, label="Actual Arrivals", linewidth=2)
plt.plot(test_index, y_pred, label="Predicted Arrivals (LSTM)", linestyle="--", linewidth=2)

plt.title("Actual vs Predicted Tourist Arrivals (Test Set)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Tourist Arrivals", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 12. Seasonality Decomposition (STL)
# =========================
from statsmodels.tsa.seasonal import STL

stl = STL(data[target_col], period=12)
result = stl.fit()

plt.figure(figsize=(10,8))
plt.subplot(411); plt.plot(result.observed); plt.title("Observed")
plt.subplot(412); plt.plot(result.trend); plt.title("Trend")
plt.subplot(413); plt.plot(result.seasonal); plt.title("Seasonality")
plt.subplot(414); plt.plot(result.resid); plt.title("Residual")
plt.tight_layout()
plt.show()



# =========================
# 13. Feature Importance using Random Forest
# =========================
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_raw, y_train_raw.ravel())

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(10,6))
importances.plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.show()



# =========================
# 14. Scenario-Based Forecasting
# =========================
def run_scenario(multiplier_USD=1.0, multiplier_travel=1.0):
    seq = X_test_scaled[-time_steps:].copy()
    future_scaled = []
    
    for _ in range(6):
        pred = model.predict(seq.reshape(1, time_steps, X_train.shape[2]), verbose=0)
        
        new_row = seq[-1].copy()
        new_row[X.columns.get_loc("USD")] *= multiplier_USD
        new_row[X.columns.get_loc("Sri Lanka travel")] *= multiplier_travel
        
        seq = np.vstack([seq[1:], new_row])
        future_scaled.append(pred[0])

    return scaler_y.inverse_transform(np.array(future_scaled)).flatten()

future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1),
                             periods=6, freq='MS')

baseline = forecast_df["Forecast_Arrivals"].values
best_case = run_scenario(multiplier_USD=0.95, multiplier_travel=1.10)
worst_case = run_scenario(multiplier_USD=1.10, multiplier_travel=0.90)

plt.figure(figsize=(10,6))
plt.plot(future_dates, baseline, label="Baseline")
plt.plot(future_dates, best_case, label="Best Case (+ Tourism Trend, ↓ USD)")
plt.plot(future_dates, worst_case, label="Worst Case (- Trend, ↑ USD)")
plt.title("Scenario-Based Forecasting")
plt.ylabel("Tourist Arrivals")
plt.xlabel("Future Months")
plt.legend()
plt.show()
   

   