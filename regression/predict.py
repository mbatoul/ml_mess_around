#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressor:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
  
  def predict(self, X, w, b):
    return X * w + b

  def loss(self, X, Y, w, b):
    return np.average((self.predict(X, w, b) - Y) ** 2)

  def train(self, iterations, lr):
    w = b = 0
    for _ in range(iterations):
      curr_loss = self.loss(X, Y, w, b)
      if curr_loss > self.loss(X, Y, w + lr, b):
        w += lr
      elif curr_loss > self.loss(X, Y, w - lr, b):
        w -= lr
      elif curr_loss > self.loss(X, Y, w, b + lr):
        b += lr
      elif curr_loss > self.loss(X, Y, w, b - lr):
        b -= lr
      else:
        return w, b
    raise Exception(f'{iterations} iterations were not enough to converge.')


X, Y = np.loadtxt('data.txt', skiprows=1, unpack=True)

print(f'Input variables: number of reservations\n{X}\n')
print(f'Labels: number of pizzas ordered\n{Y}\n')

reg = LinearRegressor(X, Y)

w, b = reg.train(iterations=100_000, lr=0.001)
print('Slope: %.3f, Intercept: %.3f\n' % (w, b))

print('According to the predictions, for 20 reservations, %i pizzas will be ordered.' % reg.predict(20, w, b))
