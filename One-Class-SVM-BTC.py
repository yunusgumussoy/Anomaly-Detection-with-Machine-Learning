# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:12:57 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import yfinance as yf

# Step 1: Load the data
data = yf.download('BTC-USD', start='2020-01-01', end='2024-07-01')

# Step 2: Preprocess the data
data = data[['Close']].copy()  # Use the 'Close' price
data['Returns'] = data['Close'].pct_change()  # Calculate daily returns
data.dropna(inplace=True)  # Remove missing values

# Step 3: Feature engineering
features = data[['Close', 'Returns']].values  # Use both Close price and Returns as features

# Step 4: Apply One-Class SVM
model = OneClassSVM(gamma='auto', nu=0.01)  # nu is the contamination parameter
model.fit(features)
data['Anomaly'] = model.predict(features)

# Anomalies are labeled as -1, normal points are labeled as 1
data['Anomaly'] = data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Step 5: Plot the results with vivid colors for anomalies
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['Close'], label='Bitcoin Price', color='blue')
plt.scatter(data.index[data['Anomaly'] == 1], data['Close'][data['Anomaly'] == 1], 
            color='red', marker='o', s=100, label='Anomalies')  # Red circles for anomalies
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Anomaly Detection using One-Class SVM')
plt.legend()
plt.show()
