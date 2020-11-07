#!usr/bin/python3 python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv', sep=',', na_values=['-', 'NaN'])

data = data.replace("< ", "", regex=True).replace('traces', 0)
cols = data.loc[:, data.columns != 'alim_ssssgrp_nom_eng'].columns
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

food = ['vegetables. raw', 'vegetables. cooked', 'vegetables. dried or dehydrated', 'legumes. cooked', 'legumes. raw', 'legumes. dried', 'fresh fruits']
data = data[['alim_ssssgrp_nom_eng', "Phosphorus (mg/100g)", "Zinc (mg/100g)"]]
# data = data[(data['alim_ssssgrp_nom_eng'].isin(food))]
data = data.dropna()
plt.scatter(data["Phosphorus (mg/100g)"], data["Zinc (mg/100g)"], c='b')

x = data["Phosphorus (mg/100g)"].to_numpy()
b = data["Zinc (mg/100g)"]

A = np.array([x, np.ones(x.shape[0])]).T
x_hat = np.linalg.inv(A.T @ A) @ A.T @ b # normal equation
a, b = x_hat

x1 = np.linspace(-100, 1000, 100000)
y1 = a * x1 + b
plt.plot(x1, y1, c='r')
plt.show()