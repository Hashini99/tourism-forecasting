# # svr_tourist_prediction.py
# import pandas as pd
# import numpy as np
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # -------------------------------
# # 1. Load cleaned dataset
# # -------------------------------
# df = pd.read_csv("tourism_dataset_final_clean.csv")

# # Ensure numeric
# df['tourist arrival'] = pd.to_numeric(df['tourist arrival'], errors='coerce')
# df['USD'] = pd.to_numeric(df['USD'], errors='coerce')
# df['Sri Lanka tourism'] = pd.to_numeric(df['Sri Lanka tourism'], errors='coerce')
# df['Sri Lanka travel'] = pd.to_numeric(df['Sri Lanka travel'], errors='coerce')

# # Fill missing values
# df['tourist arrival'].interpolate(inplace=True)
# df['USD'].interpolate(inplace=True)
# df['Sri Lanka tourism'].interpolate(inplace=True)
# df['Sri Lanka travel'].interpolate(inplace=True)

# # -------------------------------
# # 2. Features & Target
# # -------------------------------
# # Use year, month, USD, Google trends as features
# features = ['year', 'month', 'USD', 'Sri Lanka tourism', 'Sri Lanka travel']
# X = df[features].values
# y = df['tourist arrival'].values

# # -------------------------------
# # 3. Scaling
# # -------------------------------
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

# # -------------------------------
# # 4. Train-test split
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# # -------------------------------
# # 5. Train SVR model
# # -------------------------------
# svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
# svr.fit(X_train, y_train)

# # -------------------------------
# # 6. Predict tourist arrivals for September 2025
# # -------------------------------
# # Find USD and Google Trends for Sept 2025
# # For simplicity, use last known USD and Google Trends as proxy
# last_row = df.iloc[-1]
# usd_sept2025 = last_row['USD']       # approximate
# tourism_trend_sept2025 = last_row['Sri Lanka tourism']
# travel_trend_sept2025 = last_row['Sri Lanka travel']

# X_pred = np.array([[2025, 9, usd_sept2025, tourism_trend_sept2025, travel_trend_sept2025]])
# X_pred_scaled = scaler_X.transform(X_pred)

# y_pred_scaled = svr.predict(X_pred_scaled)
# y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

# print(f"Predicted tourist arrivals for September 2025: {int(y_pred[0,0])}")




# svr_tourist_prediction_full.py

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("tourism_dataset_final_clean.csv")

# Strip column names
df.columns = [c.strip() for c in df.columns]

# =========================
# 2. Handle missing values
# =========================
numeric_cols = ['tourist arrival','USD','GDP (current US$)',
                'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
                'mean_temp','max_temp','min_temp','total_precip',
                'Sri Lanka tourism','Sri Lanka travel']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(method='ffill', inplace=True)
        df[col].fillna(method='bfill', inplace=True)

# =========================
# 3. Encode festival (categorical)
# =========================
if 'festival' in df.columns:
    df['festival'] = df['festival'].fillna('None')
    df = pd.get_dummies(df, columns=['festival'], drop_first=True)

# =========================
# 4. Features & Target
# =========================
target_col = 'tourist arrival'

# Drop date column if present
drop_cols = ['date', target_col]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target_col].values

# =========================
# 5. Scaling
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

# =========================
# 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# =========================
# 7. Train SVR model
# =========================
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

# =========================
# 8. Predict for September 2025
# =========================
# For simplicity, use last available values for numeric & festival features
last_row = df.iloc[-1]

# Build feature vector for Sept 2025
X_pred_dict = last_row.to_dict()
X_pred_dict['year'] = 2025
X_pred_dict['month'] = 9

# Keep only model features
X_pred = np.array([[X_pred_dict[col] for col in X.columns]])

# Scale and predict
X_pred_scaled = scaler_X.transform(X_pred)
y_pred_scaled = svr.predict(X_pred_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

print(f"Predicted tourist arrivals for September 2025: {int(y_pred[0,0])}")


# =========================
# 9. Model Evaluation
# =========================
y_pred_test = svr.predict(X_test)
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
r2 = r2_score(y_test_inv, y_pred_test_inv)

print("\n=== SVR Model Performance ===")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

# =========================
# 10. Visualizations
# =========================

# --- (a) Predicted vs Actual (Scatter Plot)
plt.figure(figsize=(7, 6))
plt.scatter(y_test_inv, y_pred_test_inv, alpha=0.7, color='teal')
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--', lw=2)
plt.title(" Predicted vs Actual Tourist Arrivals (SVR)")
plt.xlabel("Actual Tourist Arrivals")
plt.ylabel("Predicted Tourist Arrivals")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- (b) Residual Plot
residuals = y_test_inv.flatten() - y_pred_test_inv.flatten()
plt.figure(figsize=(7, 6))
plt.scatter(y_pred_test_inv, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title(" Residual Plot for SVR Predictions")
plt.xlabel("Predicted Tourist Arrivals")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- (c) Actual vs Predicted Line Plot (for trend comparison)
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual', color='blue', linewidth=2)
plt.plot(y_pred_test_inv, label='Predicted', color='orange', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Tourist Arrivals (SVR Model Trend)")
plt.xlabel("Test Sample Index")
plt.ylabel("Number of Tourists")
plt.legend()
plt.tight_layout()
plt.show()

# Predicted tourist arrivals for September 2025: 188475