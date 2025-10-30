# LSTM Rainfall Forecasting for Kericho County (2000–2024)
# Author: Timothy Kiprop | Northwest Nazarene University
# Requirements: pandas, numpy, tensorflow, sklearn, matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df = pd.read_csv("ClimateEngine.csv")

# Ensure correct column names
df.columns = ['Date', 'Rainfall']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Fill missing values (if any)
df['Rainfall'].fillna(method='ffill', inplace=True)

# Extract rainfall values as numpy array
data = df['Rainfall'].values.reshape(-1, 1)

# Normalize between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# -----------------------------
# 2. Create sequences
# -----------------------------
def create_sequences(dataset, sequence_length=30):
    X, y = [], []
    for i in range(sequence_length, len(dataset)):
        X.append(dataset[i-sequence_length:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

SEQ_LEN = 30  # use past 30 days to predict next day
X, y = create_sequences(scaled_data, SEQ_LEN)

# Reshape for LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# -----------------------------
# 3. Split into train/test sets
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 4. Build LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6. Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# 7. Evaluate Model
# -----------------------------
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"\nModel Performance:")
print(f"MAE  = {mae:.3f} mm")
print(f"RMSE = {rmse:.3f} mm")
print(f"R²   = {r2:.3f}")

# -----------------------------
# 8. Plot Results
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test_inv[:300], label='Observed Rainfall')
plt.plot(y_pred_inv[:300], label='Predicted Rainfall')
plt.title('Observed vs Predicted Daily Rainfall (Kericho, 2023–2024)')
plt.xlabel('Days')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()
