# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:36:07 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the data
data = yf.download('BTC-USD', start='2020-01-01', end='2024-07-01')

# Step 2: Preprocess the data
data = data[['Close']].copy()  # Use the 'Close' price
data['Returns'] = data['Close'].pct_change()  # Calculate daily returns
data.dropna(inplace=True)  # Remove missing values

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['Close', 'Returns']] = scaler.fit_transform(data[['Close', 'Returns']])

# Step 3: Feature engineering
features = data[['Close', 'Returns']].values  # Use both Close price and Returns as features

# Step 4: Build and train the autoencoder
input_dim = features.shape[1]

autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(features, features, epochs=50, batch_size=32, validation_split=0.1, shuffle=True)

# Step 5: Detect anomalies
reconstructions = autoencoder.predict(features)
reconstruction_error = np.mean(np.square(reconstructions - features), axis=1)

# Determine the threshold for anomalies
threshold = np.percentile(reconstruction_error, 95)
data['Anomaly'] = reconstruction_error > threshold

# Convert the anomaly column to integers (0 for normal, 1 for anomaly)
data['Anomaly'] = data['Anomaly'].astype(int)

# Step 6: Plot the results with vivid colors for anomalies
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['Close'], label='Bitcoin Price', color='blue')
plt.scatter(data.index[data['Anomaly'] == 1], data['Close'][data['Anomaly'] == 1], 
            color='red', marker='o', s=100, label='Anomalies')  # Red circles for anomalies
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Anomaly Detection using Autoencoder')
plt.legend()
plt.show()
