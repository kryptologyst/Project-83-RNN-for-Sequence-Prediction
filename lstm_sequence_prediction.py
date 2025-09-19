# Project 83. RNN for sequence prediction
# Description:
# A Recurrent Neural Network (RNN) is a type of neural network designed for sequential data such as time series, language, or sensor readings. In this project, we build a basic RNN using Keras to predict the next value in a numerical sequence based on past observations.

# Python Implementation:


# Install if not already: pip install tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def generate_data(data_points=200):
    """Generate synthetic sequence data (e.g., sine wave)."""
    np.random.seed(42)
    t = np.linspace(0, 20, data_points)
    data = np.sin(t) + np.random.normal(0, 0.1, data_points)
    return data

def preprocess_data(data, window_size=10):
    """Normalize data and create sequences."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], window_size, 1))
    X_test = X_test.reshape((X_test.shape[0], window_size, 1))

    return X_train, y_train, X_test, y_test, scaler

def build_and_train_model(X_train, y_train, window_size=10):
    """Build and train the LSTM model."""
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, verbose=0)
    return model

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the model and inverse transform the predictions."""
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return y_test_inv, y_pred_inv

def print_evaluation_results(y_test_inv, y_pred_inv):
    """Calculate and print the Mean Squared Error."""
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"Mean Squared Error: {mse}")

def main():
    """Main function to run the RNN sequence prediction."""
    data = generate_data()
    X_train, y_train, X_test, y_test, scaler = preprocess_data(data)
    model = build_and_train_model(X_train, y_train)
    y_test_inv, y_pred_inv = evaluate_model(model, X_test, y_test, scaler)
    print_evaluation_results(y_test_inv, y_pred_inv)

if __name__ == '__main__':
    main()


# What This Project Demonstrates:
# Implements a Simple RNN for predicting sequential values

# Prepares sliding window sequences from a time series

# Trains and visualizes predictions vs actual values