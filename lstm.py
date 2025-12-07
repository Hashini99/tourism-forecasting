
# # lstm_tourism_forecast_full.py

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from datetime import datetime

# # === 1. Load dataset ===
# data = pd.read_csv("tourism_dataset_final_clean.csv")

# # Strip column names
# data.columns = [c.strip() for c in data.columns]

# # Rename target column
# if 'tourist arrival' in data.columns:
#     data.rename(columns={'tourist arrival': 'arrivals'}, inplace=True)
# elif 'tourist_arrivals' in data.columns:
#     data.rename(columns={'tourist_arrivals': 'arrivals'}, inplace=True)

# # === 2. Create datetime column ===
# if 'date' not in data.columns:
#     data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
# else:
#     data['date'] = pd.to_datetime(data['date'])

# data.sort_values('date', inplace=True)
# data.set_index('date', inplace=True)

# # === 3. Handle missing values ===
# for col in ['arrivals', 'USD', 'GDP (current US$)',
#             'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
#             'festival', 'mean_temp', 'max_temp', 'min_temp', 'total_precip',
#             'Sri Lanka tourism', 'Sri Lanka travel']:
#     if col in data.columns:
#         if data[col].dtype == object:
#             data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
#         data[col].interpolate(inplace=True)

# # === 4. Encode festival as numeric (optional) ===
# if 'festival' in data.columns:
#     data['festival'] = data['festival'].fillna('None')
#     data['festival'] = data['festival'].astype('category').cat.codes

# # === 5. Features and target ===
# feature_cols = ['arrivals', 'USD', 'GDP (current US$)',
#                 'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
#                 'festival', 'mean_temp', 'max_temp', 'min_temp', 'total_precip',
#                 'Sri Lanka tourism', 'Sri Lanka travel']

# X_all = data[feature_cols].values

# # === 6. Normalize data ===
# scaler = MinMaxScaler()
# scaled_X = scaler.fit_transform(X_all)

# # === 7. Create sequences ===
# time_steps = 12

# def create_sequences(data, time_steps):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i+time_steps, :])
#         y.append(data[i + time_steps, 0])  # predict 'arrivals'
#     return np.array(X), np.array(y)

# X, y = create_sequences(scaled_X, time_steps)

# # === 8. Build LSTM model ===
# model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
#     Dropout(0.2),
#     LSTM(64, return_sequences=False),
#     Dropout(0.2),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# # === 9. Forecast next 9 months (up to Sep 2025) ===
# forecast_steps = 9
# current_sequence = scaled_X[-time_steps:].copy()
# predictions = []

# for _ in range(forecast_steps):
#     pred = model.predict(current_sequence.reshape(1, time_steps, X.shape[2]), verbose=0)[0,0]
#     predictions.append(pred)
    
#     # Prepare next input row: predicted arrivals + last known feature values
#     next_row = current_sequence[-1, :].copy()
#     next_row[0] = pred  # replace arrivals with predicted value
#     current_sequence = np.vstack((current_sequence[1:], next_row))

# # === 10. Inverse scale only the 'arrivals' column ===
# predicted_values = np.array(predictions).reshape(-1, 1)
# arrivals_scaler = MinMaxScaler()
# arrivals_scaler.min_, arrivals_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
# predicted_arrivals = arrivals_scaler.inverse_transform(predicted_values)

# # === 11. Build forecast DataFrame ===
# future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1),
#                              periods=forecast_steps, freq='MS')

# forecast_df = pd.DataFrame({
#     'Date': future_dates,
#     'Predicted_Arrivals': predicted_arrivals.flatten()
# })

# print("\n=== Forecasted Tourist Arrivals ===")
# print(forecast_df)

# # === 12. Prediction for September 2025 ===
# sept_2025 = forecast_df[forecast_df['Date'].dt.month == 9]
# if not sept_2025.empty:
#     print(f"\n📈 Predicted Tourist Arrivals for September 2025: {int(sept_2025['Predicted_Arrivals'].values[0])}")

# # === 13. Plot results ===
# plt.figure(figsize=(10, 5))
# plt.plot(data.index, data['arrivals'], label='Historical Data')
# plt.plot(forecast_df['Date'], forecast_df['Predicted_Arrivals'], label='Forecast', marker='o')
# plt.title("Tourist Arrivals Forecast (LSTM with All Features)")
# plt.xlabel("Date")
# plt.ylabel("Number of Tourists")
# plt.legend()
# plt.tight_layout()
# plt.show()


# lstm_tourism_forecast_full.py



# lstm_tourism_forecast_full_fixed.py

# lstm_tourism_forecast_full.py
# lstm_tourism_forecast_deterministic.py



# version 2


# import os
# import random
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# # =========================
# # 0. Set seeds for reproducibility
# # =========================
# SEED = 42
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# # Force TensorFlow to use deterministic operations
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# # =========================
# # 1. Load and prepare data
# # =========================
# data = pd.read_csv("tourism_dataset_final_clean.csv")
# data.columns = [c.strip() for c in data.columns]

# # Fill numeric columns
# numeric_cols = [
#     'tourist arrival','USD','GDP (current US$)',
#     'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
#     'mean_temp','max_temp','min_temp','total_precip',
#     'Sri Lanka tourism','Sri Lanka travel'
# ]
# for col in numeric_cols:
#     if col in data.columns:
#         data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

# # Handle festival column as categorical
# if 'festival' in data.columns:
#     data['festival'] = data['festival'].fillna('None')
#     data = pd.get_dummies(data, columns=['festival'], drop_first=True)

# # Create datetime column
# if 'date' not in data.columns:
#     data['date'] = pd.to_datetime(data[['year','month']].assign(day=1))
# else:
#     data['date'] = pd.to_datetime(data['date'])

# data.sort_values('date', inplace=True)
# data.set_index('date', inplace=True)

# # =========================
# # 2. Prepare features & target
# # =========================
# target_col = 'tourist arrival'
# y = data[target_col].values.reshape(-1,1)

# drop_cols = ['year','month','date',target_col]
# X = data.drop(columns=[c for c in drop_cols if c in data.columns])

# # =========================
# # 3. Normalize features & target
# # =========================
# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y)

# # =========================
# # 4. Create sequences for LSTM
# # =========================
# time_steps = 12

# def create_sequences(X, y, time_steps=12):
#     X_seq, y_seq = [], []
#     for i in range(len(X) - time_steps):
#         X_seq.append(X[i:i+time_steps])
#         y_seq.append(y[i+time_steps])
#     return np.array(X_seq), np.array(y_seq)

# X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# # =========================
# # 5. Build LSTM model
# # =========================
# model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
#     Dropout(0.2, seed=SEED),
#     LSTM(64, return_sequences=False),
#     Dropout(0.2, seed=SEED),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error')

# # =========================
# # 6. Train the model
# # =========================
# model.fit(X_seq, y_seq, epochs=100, batch_size=16, verbose=1)

# # =========================
# # 7. Forecast next 9 months
# # =========================
# forecast_steps = 9
# last_sequence = X_scaled[-time_steps:].copy()
# predictions = []

# future_features = np.tile(X_scaled[-1:], (forecast_steps, 1))
# current_sequence = last_sequence.copy()

# for i in range(forecast_steps):
#     pred = model.predict(current_sequence.reshape(1, time_steps, X_scaled.shape[1]), verbose=0)
#     predictions.append(pred[0,0])
#     current_sequence = np.vstack([current_sequence[1:], future_features[i:i+1]])

# predicted_values = scaler_y.inverse_transform(np.array(predictions).reshape(-1,1))

# # =========================
# # 8. Build forecast DataFrame
# # =========================
# future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1),
#                              periods=forecast_steps, freq='MS')

# forecast_df = pd.DataFrame({
#     'Date': future_dates,
#     'Predicted_Arrivals': predicted_values.flatten()
# })

# print("\n=== Forecasted Tourist Arrivals ===")
# print(forecast_df)

# # =========================
# # 9. Predicted September 2025
# # =========================
# # sept_2025 = forecast_df[forecast_df['Date'].dt.month==9]
# # if not sept_2025.empty:
# #     print(f"\n📈 Predicted Tourist Arrivals for September 2025: {int(sept_2025['Predicted_Arrivals'].values[0])}")

# aug_2025 = forecast_df[forecast_df['Date'].dt.month==8]
# if not aug_2025.empty:
#     print(f"\n📈 Predicted Tourist Arrivals for September 2025: {int(aug_2025['Predicted_Arrivals'].values[0])}")

# # =========================
# # 10. Plot results
# # =========================
# plt.figure(figsize=(12,6))
# plt.plot(data.index, y, label='Historical Data')
# plt.plot(forecast_df['Date'], forecast_df['Predicted_Arrivals'], label='Forecast', marker='o')
# plt.title("Tourist Arrivals Forecast (Deterministic LSTM with All Features)")
# plt.xlabel("Date")
# plt.ylabel("Number of Tourists")
# plt.legend()
# plt.tight_layout()
# plt.show()




# import os
# import random
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# # =========================
# # 0. Set seeds for reproducibility
# # =========================
# SEED = 42
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# # =========================
# # 1. Load and prepare data
# # =========================
# data = pd.read_csv("tourism_dataset_final_clean.csv")
# data.columns = [c.strip() for c in data.columns]

# numeric_cols = [
#     'tourist arrival','USD','GDP (current US$)',
#     'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
#     'mean_temp','max_temp','min_temp','total_precip',
#     'Sri Lanka tourism','Sri Lanka travel'
# ]
# for col in numeric_cols:
#     if col in data.columns:
#         data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

# if 'festival' in data.columns:
#     data['festival'] = data['festival'].fillna('None')
#     data = pd.get_dummies(data, columns=['festival'], drop_first=True)

# if 'date' not in data.columns:
#     data['date'] = pd.to_datetime(data[['year','month']].assign(day=1))
# else:
#     data['date'] = pd.to_datetime(data['date'])

# data.sort_values('date', inplace=True)
# data.set_index('date', inplace=True)

# # =========================
# # 2. Prepare features & target
# # =========================
# target_col = 'tourist arrival'
# y = data[target_col].values.reshape(-1,1)

# drop_cols = ['year','month','date',target_col]
# X = data.drop(columns=[c for c in drop_cols if c in data.columns])

# # =========================
# # 3. Normalize
# # =========================
# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y)

# # =========================
# # 4. Create sequences
# # =========================
# time_steps = 12
# def create_sequences(X, y, time_steps=12):
#     X_seq, y_seq = [], []
#     for i in range(len(X) - time_steps):
#         X_seq.append(X[i:i+time_steps])
#         y_seq.append(y[i+time_steps])
#     return np.array(X_seq), np.array(y_seq)

# X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# # =========================
# # 5. Build model
# # =========================
# model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
#     Dropout(0.2, seed=SEED),
#     LSTM(64, return_sequences=False),
#     Dropout(0.2, seed=SEED),
#     Dense(1)
# ])
# model.compile(optimizer='adam', loss='mean_squared_error')

# # =========================
# # 6. Train & Plot Loss Curve
# # =========================
# history = model.fit(X_seq, y_seq, epochs=100, batch_size=16, verbose=1)

# # === Figure 3.1: Model Loss Curve ===
# plt.figure(figsize=(8,5))
# plt.plot(history.history['loss'], label='Training Loss', color='blue')
# plt.title("Figure 3.1: LSTM Model Training Loss Curve")
# plt.xlabel("Epochs")
# plt.ylabel("Loss (MSE)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # =========================
# # 7. Forecast next 9 months
# # =========================
# forecast_steps = 9
# last_sequence = X_scaled[-time_steps:].copy()
# predictions = []

# future_features = np.tile(X_scaled[-1:], (forecast_steps, 1))
# current_sequence = last_sequence.copy()

# for i in range(forecast_steps):
#     pred = model.predict(current_sequence.reshape(1, time_steps, X_scaled.shape[1]), verbose=0)
#     predictions.append(pred[0,0])
#     current_sequence = np.vstack([current_sequence[1:], future_features[i:i+1]])

# predicted_values = scaler_y.inverse_transform(np.array(predictions).reshape(-1,1))

# # =========================
# # 8. Forecast DataFrame
# # =========================
# future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1),
#                              periods=forecast_steps, freq='MS')
# forecast_df = pd.DataFrame({
#     'Date': future_dates,
#     'Predicted_Arrivals': predicted_values.flatten()
# })

# print("\n=== Forecasted Tourist Arrivals ===")
# print(forecast_df)

# # =========================
# # 9. Evaluate LSTM on training data
# # =========================
# y_pred_train = model.predict(X_seq, verbose=0)
# y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
# y_true_train_inv = scaler_y.inverse_transform(y_seq)

# mae = mean_absolute_error(y_true_train_inv, y_pred_train_inv)
# rmse = np.sqrt(mean_squared_error(y_true_train_inv, y_pred_train_inv))
# r2 = r2_score(y_true_train_inv, y_pred_train_inv)

# print("\n=== LSTM Model Performance (on training data) ===")
# print(f"MAE:  {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R²:   {r2:.3f}")

# # =========================
# # 10. Plot Forecast
# # =========================
# plt.figure(figsize=(12,6))
# plt.plot(data.index, y, label='Historical Data')
# plt.plot(forecast_df['Date'], forecast_df['Predicted_Arrivals'], label='LSTM Forecast', marker='o')
# plt.title("Tourist Arrivals Forecast (Deterministic LSTM with All Features)")
# plt.xlabel("Date")
# plt.ylabel("Number of Tourists")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # =========================
# # 11. Overlay LSTM vs SARIMA (Figure 3.2)
# # =========================
# # Assume you have SARIMA predictions saved as 'sarima_forecast.csv' with columns ['Date','SARIMA_Pred']
# if os.path.exists("sarima_forecast.csv"):
#     sarima_df = pd.read_csv("sarima_forecast.csv")
#     sarima_df['Date'] = pd.to_datetime(sarima_df['Date'])

#     plt.figure(figsize=(12,6))
#     plt.plot(data.index, y, label='Actual Tourist Arrivals', color='black')
#     plt.plot(sarima_df['Date'], sarima_df['SARIMA_Pred'], label='SARIMA Forecast', color='orange', linestyle='--')
#     plt.plot(forecast_df['Date'], forecast_df['Predicted_Arrivals'], label='LSTM Forecast', color='blue', marker='o')
#     plt.title("Figure 3.2: Predicted vs Actual Arrivals (SARIMA vs LSTM)")
#     plt.xlabel("Date")
#     plt.ylabel("Number of Tourists")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# else:
#     print("\n⚠️ 'sarima_forecast.csv' not found — skipping Figure 3.2 overlay plot.")








# import os
# import random
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# # =========================
# # 0. Set seeds
# # =========================
# SEED = 42
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)


# # =========================
# # 1. Load Data
# # =========================
# data = pd.read_csv("tourism_dataset_final_clean.csv")
# data.columns = [c.strip() for c in data.columns]

# numeric_cols = [
#     'tourist arrival','USD','GDP (current US$)',
#     'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
#     'mean_temp','max_temp','min_temp','total_precip',
#     'Sri Lanka tourism','Sri Lanka travel'
# ]
# for col in numeric_cols:
#     if col in data.columns:
#         data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

# # Categorical → One Hot Encoding
# if 'festival' in data.columns:
#     data['festival'] = data['festival'].fillna('None')
#     data = pd.get_dummies(data, columns=['festival'], drop_first=True)

# data['date'] = pd.to_datetime(data[['year','month']].assign(day=1))
# data.sort_values('date', inplace=True)
# data.set_index('date', inplace=True)


# # =========================
# # 2. Features & Target
# # =========================
# target_col = "tourist arrival"
# y = data[target_col].values.reshape(-1,1)
# X = data.drop(columns=['year','month',target_col], errors='ignore')


# # =========================
# # 3. Train-Test Split (80% train, 20% test)
# # =========================
# train_size = int(len(X) * 0.8)
# X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
# y_train_raw, y_test_raw = y[:train_size], y[train_size:]


# # =========================
# # 4. Scaling - Fit only on train data
# # =========================
# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

# X_train_scaled = scaler_X.fit_transform(X_train_raw)
# X_test_scaled = scaler_X.transform(X_test_raw)

# y_train_scaled = scaler_y.fit_transform(y_train_raw)
# y_test_scaled = scaler_y.transform(y_test_raw)


# # =========================
# # 5. Convert to sequences
# # =========================
# time_steps = 12

# def create_sequences(X_data, y_data):
#     Xs, ys = [], []
#     for i in range(len(X_data) - time_steps):
#         Xs.append(X_data[i:i+time_steps])
#         ys.append(y_data[i+time_steps])
#     return np.array(Xs), np.array(ys)

# X_train, y_train = create_sequences(X_train_scaled, y_train_scaled)
# X_test, y_test = create_sequences(X_test_scaled, y_test_scaled)


# # =========================
# # 6. LSTM Model
# # =========================
# model = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.2),
#     LSTM(64),
#     Dropout(0.2),
#     Dense(1)
# ])
# model.compile(optimizer="adam", loss="mse")


# # =========================
# # 7. Train Model
# # =========================
# history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# # Plot loss curve
# plt.figure()
# plt.plot(history.history['loss'])
# plt.title("Training Loss Curve")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()


# # =========================
# # 8. Predict Test Set
# # =========================
# y_pred_scaled = model.predict(X_test)
# y_pred = scaler_y.inverse_transform(y_pred_scaled)
# y_true = scaler_y.inverse_transform(y_test)


# # =========================
# # 9. Evaluation on Test Set
# # =========================
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_true, y_pred)

# print("\n=== LSTM Performance (Test Set) ===")
# print(f"MAE  : {mae:.2f}")
# print(f"RMSE : {rmse:.2f}")
# print(f"R²   : {r2:.3f}")


# # =========================
# # 10. Plot Actual vs Predicted on Test
# # =========================
# test_index = data.index[train_size+time_steps:]
# plt.figure(figsize=(12,6))
# plt.plot(test_index, y_true, label="Actual")
# plt.plot(test_index, y_pred, label="LSTM Predicted")
# plt.title("LSTM - Actual vs Predicted (Test Set)")
# plt.xlabel("Date")
# plt.ylabel("Tourist Arrivals")
# plt.legend()
# plt.show()


# # =========================
# # 11. Forecast Next 6 Months
# # =========================
# future_steps = 6
# last_seq = X_test_scaled[-time_steps:]
# forecast_scaled = []

# for _ in range(future_steps):
#     pred = model.predict(last_seq.reshape(1, time_steps, X_train.shape[2]), verbose=0)
#     forecast_scaled.append(pred[0])
#     last_seq = np.vstack([last_seq[1:], pred])

# future = scaler_y.inverse_transform(np.array(forecast_scaled))

# future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1),
#                              periods=future_steps, freq='MS')

# forecast_df = pd.DataFrame({
#     "Date": future_dates,
#     "Forecast_Arrivals": future.flatten()
# })

# print("\n=== Future Forecast ===")
# print(forecast_df)





# ==========================================
# Tourism Demand Forecasting using LSTM
# ==========================================

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
# 3️⃣ Train / Test Split
# -------------------------
train_size = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
y_train_raw, y_test_raw = y[:train_size], y[train_size:]

# -------------------------
# 4️⃣ Scaling
# -------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# -------------------------
# 5️⃣ Sequence Conversion
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
# 6️⃣ LSTM Model
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
# 7️⃣ Train
# -------------------------
history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# -------------------------
# 8️⃣ Predictions
# -------------------------
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# -------------------------
# 9️⃣ Evaluation Metrics
# -------------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n=== LSTM Performance (Test Set) ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# -------------------------
# 🔟 Plot Actual vs Predicted
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
   

   