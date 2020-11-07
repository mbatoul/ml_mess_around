#!usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

if __name__ == '__main__':
  URL = 'https://raw.githubusercontent.com/hadrienj/essential_math_for_data_science/master/data/covid19.csv'
  data = pd.read_csv(URL)
  data['days'] = data['Date'].str.split('/').apply(lambda x: x[2]).astype(int)
  X = data['days'].to_numpy().reshape(-1, 1)
  y = data['Ile-de-France'].to_numpy()
  
  # Add noise to y using normal distribution
  y_noise = y + (np.random.normal(0, 100, X.shape[0]) * X.flatten()).flatten()
  scaler = StandardScaler()
  scaler.fit(X)
  
  # Transforms data such that mean is zero and standard deviation is 1
  X_transformed = scaler.transform(X)
  y_transformed = scaler.transform(y.reshape(-1, 1)).flatten()
  y_noise_transformed = scaler.transform(y_noise.reshape(-1, 1)).flatten()
  poly = PolynomialFeatures(10, include_bias=False)
  X_poly = poly.fit_transform(X_transformed)
  
  # Linear model using MSE with L2 regularization
  X_axis = np.random.normal(size=X_poly.shape)
  X_axis = poly.fit_transform(X_axis)
  X_axis = X_axis.reshape(-1, 1)
  f, axes = plt.subplots(1, 6, figsize=(16, 4), sharey=True)
  
  for alpha, ax in zip([0, 1e-5, 1e-3, 1e-2, 1, 1e5], axes.flatten()):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly, y_noise_transformed)
    y_axis_predicted = ridge.predict(X_axis)

    # Regression curve
    ax.plot(X_axis[:, 0], y_axis_predicted, c='r')
    # Data without noise
    ax.scatter(X_transformed, y_transformed, alpha=0.6, s=60, label='Data')
    # Data with noise
    ax.scatter(X_transformed, y_noise_transformed, alpha=0.6, s=60, label='Data with noise')
    ax.set_title('alpha: %.5f' % alpha)
