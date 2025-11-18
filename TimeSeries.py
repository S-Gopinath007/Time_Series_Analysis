import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
import tensorflow as tf

# Suppress warnings and set seed for reproducibility
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# --- CONFIGURATION ---
FILE_NAME = "all_stocks_2006-01-01_to_2018-01-01.csv"
STOCK_TICKER = 'MMM'
FEATURES = ['Open', 'High', 'Low', 'Volume', 'Close']
TARGET_FEATURE_NAME = 'Close'
TIME_STEPS = 10 # Look-back window

# --- DATA PREPARATION FUNCTIONS ---

def create_sequences(data, time_steps, target_feature_index):
    """Converts the time series data into X (sequences) and y (targets)."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, target_feature_index])
    return np.array(X), np.array(y)

# --- 1. DATA ACQUISITION AND PREPROCESSING ---

# Load and Filter Data
df = pd.read_csv(FILE_NAME)
df_filtered = df[df['Name'] == STOCK_TICKER].copy()
data = df_filtered[FEATURES]

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data[[TARGET_FEATURE_NAME]])

# Create Sequences
TARGET_INDEX = FEATURES.index(TARGET_FEATURE_NAME)
X, y = create_sequences(scaled_data, TIME_STEPS, TARGET_INDEX)
y = y.reshape(-1, 1)

# Time-based Split (70% train, 15% val, 15% test)
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

X_train_val = np.concatenate((X_train, X_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)

print("--- Step 1: Data Preparation Complete ---")
print(f"Final LSTM Input Shape (X_train): {X_train.shape}")
print("-" * 50)

# --- 2. BASELINE LSTM MODEL IMPLEMENTATION ---

def build_baseline_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

baseline_model = build_baseline_model((X_train.shape[1], X_train.shape[2]))

print("Training Baseline Model (20 epochs)...")
baseline_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Evaluate Baseline Model
y_pred_scaled_base = baseline_model.predict(X_test, verbose=0)
y_test_inv = close_scaler.inverse_transform(y_test)
y_pred_inv_base = close_scaler.inverse_transform(y_pred_scaled_base)

rmse_baseline = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv_base))
mae_baseline = mean_absolute_error(y_test_inv, y_pred_inv_base)

print("Baseline LSTM Performance (on Test Set):")
print(f"RMSE: ${rmse_baseline:.4f}, MAE: ${mae_baseline:.4f}")
print("-" * 50)

# --- 2b. OPTIMIZED LSTM ARCHITECTURE (Hyperparameter Tuning) ---

def create_optimized_model(units=50, dropout=0.2, learning_rate=0.001):
    """Creates a 2-layer LSTM model with tunable hyperparameters."""
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout),
        LSTM(units=units),
        Dropout(dropout),
        Dense(units=1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Setup GridSearchCV with a MINIMAL search space for speed
model_wrapper = KerasRegressor(build_fn=create_optimized_model, verbose=0, epochs=10)

param_grid = {
    # Highly constrained grid to avoid execution timeout
    'units': [50, 60],
    'dropout': [0.1, 0.3],
    'learning_rate': [0.001, 0.0005]
}

print("Starting Hyperparameter Tuning with GridSearchCV (CV=2, 10 epochs/run)...")
grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
grid_result = grid.fit(X_train_val, y_train_val)

print(f"Best Parameters Found: {grid_result.best_params_}")

# Retrain the best model for final evaluation (using more epochs)
best_model = grid_result.best_estimator_.model
print("Retraining Best Model (20 epochs) for Final Evaluation...")
best_model.fit(X_train_val, y_train_val, epochs=20, batch_size=grid_result.best_params_['batch_size'], verbose=0)

# Evaluate Optimized Model
y_pred_scaled_opt = best_model.predict(X_test, verbose=0)
y_pred_inv_opt = close_scaler.inverse_transform(y_pred_scaled_opt)

rmse_optimized = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv_opt))
mae_optimized = mean_absolute_error(y_test_inv, y_pred_inv_opt)

print("Optimized LSTM Performance (on Test Set):")
print(f"RMSE: ${rmse_optimized:.4f}, MAE: ${mae_optimized:.4f}")
print("-" * 50)

# --- 3a. CLASSICAL STATISTICAL BENCHMARK (ARIMA) ---

# Prepare unscaled 'Close' data for ARIMA
close_data = df_filtered[TARGET_FEATURE_NAME].values
data_len_for_split = len(close_data) - TIME_STEPS
val_end_index = int(data_len_for_split * 0.70) + int(data_len_for_split * 0.15) + TIME_STEPS

# Separate the training history and the test set using original indices
train_for_arima = close_data[0:val_end_index]
test_for_arima = close_data[val_end_index:]

# Rolling Forecast
order = (5, 1, 0)
history = [x for x in train_for_arima]
predictions = []

for t in range(len(test_for_arima)):
    model = ARIMA(history, order=order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)
    obs = test_for_arima[t]
    history.append(obs)

rmse_arima = np.sqrt(mean_squared_error(test_for_arima, predictions))
mae_arima = mean_absolute_error(test_for_arima, predictions)

print("ARIMA(5,1,0) Benchmark Performance (on Test Set):")
print(f"RMSE: ${rmse_arima:.4f}, MAE: ${mae_arima:.4f}")
print("-" * 50)

# --- 3b. TIME SERIES MODEL EXPLAINABILITY (Time Step Influence) ---

# Choose a single sample from the test set
sample_index = 5
X_sample = X_test[sample_index:sample_index+1]

# Calculate the baseline prediction (scaled)
P_base = best_model.predict(X_sample, verbose=0)[0][0]

# Perturb and calculate influence
influence_scores = []
num_time_steps = X_sample.shape[1]

for t in range(num_time_steps):
    X_perturbed = X_sample.copy()
    # Set the time step features to the normalized min value (0.0)
    X_perturbed[0, t, :] = 0.0
    P_t = best_model.predict(X_perturbed, verbose=0)[0][0]
    influence = np.abs(P_base - P_t)
    influence_scores.append(influence)

# Normalize scores for interpretation
total_influence = sum(influence_scores)
normalized_scores = [score / total_influence for score in influence_scores]

print("Time Step Influence Analysis (Optimized LSTM):")
print("Scores indicate the relative influence of a historical day on the current prediction (t-1 is the most recent).")
for t, score in enumerate(normalized_scores):
    # t-10 is the oldest, t-1 is the most recent
    print(f"Time Step t-{num_time_steps - t}: {score:.4f}")

# --- 4. FINAL EVALUATION & DOCUMENTATION ---

final_comparison_df = pd.DataFrame({
    'Model': ['Baseline LSTM', 'Optimized LSTM', 'ARIMA(5,1,0)'],
    'RMSE (USD)': [rmse_baseline, rmse_optimized, rmse_arima],
    'MAE (USD)': [mae_baseline, mae_optimized, mae_arima]
})

print("-" * 50)
print("FINAL MODEL COMPARISON (Performance on Test Set)")
print(final_comparison_df.to_markdown(index=False))
