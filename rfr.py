

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("tourism_dataset_final_clean.csv")
df.columns = [c.strip() for c in df.columns]

# ===============================
# 2. DATE → YEAR & MONTH
# ===============================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# ===============================
# 3. HANDLE NUMERIC MISSING VALUES
# ===============================
numeric_cols = [
    "tourist arrival", "USD", "GDP (current US$)",
    "Kerosine type jet fuel (U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon))",
    "mean_temp", "max_temp", "min_temp", "total_precip",
    "Sri Lanka tourism", "Sri Lanka travel"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(method='ffill', inplace=True)
        df[col].fillna(method='bfill', inplace=True)

# ===============================
# 4. ONE-HOT ENCODE FESTIVAL
# ===============================
if "festival" in df.columns:
    df["festival"] = df["festival"].fillna("None")
    df = pd.get_dummies(df, columns=["festival"], drop_first=True)

# ===============================
# 5. DEFINE FEATURES & TARGET
# ===============================
target = "tourist arrival"
X = df.drop(columns=["date", target])
y = df[target].values

# ===============================
# 6. SCALE FEATURES
# ===============================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ===============================
# 7. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ===============================
# 8. TRAIN RANDOM FOREST
# ===============================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# 9. EVALUATE MODEL
# ===============================
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print("\n=== MODEL PERFORMANCE (Random Forest) ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# ===============================
# 10. PREDICT SEPTEMBER 2025
# ===============================
print("\n=== Predict Tourist Arrivals for September 2025 ===")

last_row = df.iloc[-1].copy()
last_row["year"] = 2025
last_row["month"] = 9

X_pred = np.array([[last_row[col] for col in X.columns]])
X_pred_scaled = scaler_X.transform(X_pred)

y_future_scaled = model.predict(X_pred_scaled)
y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1))

print(f"Predicted Arrivals (Sep 2025): {int(y_future[0][0])}")

# ===============================
# 11. 6-MONTH FUTURE FORECAST
# ===============================
print("\n=== 6-Month Forecast ===")

future_row = df.iloc[-1].copy()
current_year = int(future_row["year"])
current_month = int(future_row["month"])

future_predictions = []
future_dates = []

for _ in range(6):
    # Next month
    next_month = current_month + 1
    next_year = current_year

    if next_month > 12:
        next_month = 1
        next_year += 1

    future_row["year"] = next_year
    future_row["month"] = next_month

    # Prepare features
    X_future = np.array([[future_row[col] for col in X.columns]])
    X_future_scaled = scaler_X.transform(X_future)

    # Predict
    y_future_scaled = model.predict(X_future_scaled)
    y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1))[0][0]

    # Store
    future_dates.append(f"{next_year}-{str(next_month).zfill(2)}")
    future_predictions.append(int(y_future))

    # Update row for recursive prediction
    future_row["tourist arrival"] = y_future

    current_year = next_year
    current_month = next_month

# Print results
for d, v in zip(future_dates, future_predictions):
    print(f"{d}: {v} ")

# ===============================
# 12. ACTUAL VS PREDICTED PLOT
# ===============================
plt.figure(figsize=(8,6))
plt.scatter(y_test_orig, y_pred, alpha=0.6)
plt.plot(
    [y_test_orig.min(), y_test_orig.max()],
    [y_test_orig.min(), y_test_orig.max()],
    'r--'
)
plt.title("Actual vs Predicted Tourist Arrivals")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()
