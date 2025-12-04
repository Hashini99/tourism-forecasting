# # rfr_tourist_prediction.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # -------------------------------
# # 1. Load cleaned dataset
# # -------------------------------
# df = pd.read_csv("tourism_dataset_final_clean.csv")

# # Ensure numeric types
# numeric_cols = ['tourist arrival', 'USD', 'Sri Lanka tourism', 'Sri Lanka travel']
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#     df[col].interpolate(inplace=True)

# # -------------------------------
# # 2. Feature Selection
# # -------------------------------
# features = ['year', 'month', 'USD', 'Sri Lanka tourism', 'Sri Lanka travel']
# X = df[features].values
# y = df['tourist arrival'].values

# # -------------------------------
# # 3. Scaling
# # -------------------------------
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# # -------------------------------
# # 4. Train-Test Split
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y_scaled, test_size=0.2, random_state=42
# )

# # -------------------------------
# # 5. Train Random Forest Regressor
# # -------------------------------
# rfr = RandomForestRegressor(
#     n_estimators=200,
#     random_state=42,
#     max_depth=10,
#     min_samples_split=2,
#     min_samples_leaf=1
# )
# rfr.fit(X_train, y_train)

# # -------------------------------
# # 6. Predict Tourist Arrivals for September 2025
# # -------------------------------
# last_row = df.iloc[-1]
# usd_sept2025 = last_row['USD']
# tourism_trend_sept2025 = last_row['Sri Lanka tourism']
# travel_trend_sept2025 = last_row['Sri Lanka travel']

# X_pred = np.array([[2025, 9, usd_sept2025, tourism_trend_sept2025, travel_trend_sept2025]])
# X_pred_scaled = scaler_X.transform(X_pred)

# y_pred_scaled = rfr.predict(X_pred_scaled)
# y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# print(f"Predicted tourist arrivals for September 2025 (RFR): {int(y_pred[0,0])}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("tourism_dataset_final_clean.csv")

df.columns = [c.strip() for c in df.columns]

# =========================
# 2. Handle missing numeric values
# =========================
numeric_cols = [
    'tourist arrival', 'USD', 'GDP (current US$)',
    'Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))',
    'mean_temp', 'max_temp', 'min_temp', 'total_precip',
    'Sri Lanka tourism', 'Sri Lanka travel'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(method='ffill', inplace=True)
        df[col].fillna(method='bfill', inplace=True)

# =========================
# 3. Encode categorical (festival)
# =========================
if 'festival' in df.columns:
    df['festival'] = df['festival'].fillna('None')
    df = pd.get_dummies(df, columns=['festival'], drop_first=True)

# =========================
# 4. Define features & target
# =========================
target_col = 'tourist arrival'
drop_cols = ['date', target_col]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target_col].values

# =========================
# 5. Scaling
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# =========================
# 7. Train Random Forest Regressor
# =========================
rfr = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rfr.fit(X_train, y_train)

# =========================
# 8. Predictions & Metrics
# =========================
y_pred_scaled = rfr.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print("=== Random Forest Model Performance ===")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

# =========================
# 9. Predict September 2025
# =========================
last_row = df.iloc[-1].to_dict()
last_row['year'] = 2025
last_row['month'] = 9

X_pred = np.array([[last_row[col] for col in X.columns]])
X_pred_scaled = scaler_X.transform(X_pred)
y_future_scaled = rfr.predict(X_pred_scaled)
y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1))
future_prediction = int(y_future[0, 0])
print(f"\nPredicted tourist arrivals for September 2025: {future_prediction}")

# =========================
# 10. 🔹 Visualization Section
# =========================

# --- (1) Feature Importance Plot ---
plt.figure(figsize=(10, 6))
importances = rfr.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=np.array(X.columns)[indices], palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- (2) Actual vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred, alpha=0.6, color="teal")
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()],
         'r--', lw=2)
plt.title("Actual vs Predicted Tourist Arrivals")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

# --- (3) Residual Plot ---
residuals = y_test_orig.flatten() - y_pred.flatten()
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color="purple")
plt.title("Residual Distribution (Prediction Errors)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- (4) Forecast Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(df['tourist arrival'].values, label='Historical Data', color='blue')
plt.axhline(future_prediction, color='orange', linestyle='--', label='Predicted Sep 2025')
plt.title("Tourist Arrivals Forecast - Random Forest Model")
plt.xlabel("Time (Monthly Index)")
plt.ylabel("Tourist Arrivals")
plt.legend()
plt.tight_layout()
plt.show()
