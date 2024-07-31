# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:38:52 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import yfinance as yf

# Step 1: Load the data
data = yf.download('BTC-USD', start='2020-01-01', end='2024-07-01')

# Step 2: Preprocess the data
data = data[['Close']].copy()  # Use the 'Close' price
data['Returns'] = data['Close'].pct_change()  # Calculate daily returns
data.dropna(inplace=True)  # Remove missing values

# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['Close', 'Returns']] = scaler.fit_transform(data[['Close', 'Returns']])

# Step 3: Feature engineering
features = data[['Close', 'Returns']].values  # Use both Close price and Returns as features

# Step 4: Apply Local Outlier Factor (LOF)
model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
data['Anomaly'] = model.fit_predict(features)

# Anomalies are labeled as -1, normal points are labeled as 1
data['Anomaly'] = data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Step 5: Plot the results with vivid colors for anomalies
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['Close'], label='Bitcoin Price', color='blue')
plt.scatter(data.index[data['Anomaly'] == 1], data['Close'][data['Anomaly'] == 1], 
            color='red', marker='o', s=100, label='Anomalies')  # Red circles for anomalies
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Anomaly Detection using Local Outlier Factor (LOF)')
plt.legend()
plt.show()
