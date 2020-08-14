import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Takes input variable (number of reservations) and the weight.
# Returns the number of pizzas.
def predict(X, w, b):
  return X * w + b

# Computes MSE (mean squared error).
# Calculates the distance from each data point to the regression line, square it, and sum all of the squared errors together.
def loss(X, Y, w, b):
  return np.average((predict(X, w, b) - Y) ** 2)

# We use a learning rate (lr) as a scale factor. The weight is updated in the direction towards minimizing the error (Gradient Descent method).
def train(X, Y, iterations, lr):
  w = b = 0
  for i in range(iterations):
    current_loss = loss(X, Y, w, b)
    print('Iteration n°%d: current loss is %.3f' % (i, current_loss))
    if loss(X, Y, w + lr, b) < current_loss:
      w += lr
    elif loss(X, Y, w - lr, b) < current_loss:
      w -= lr
    elif loss(X, Y, w, b + lr) < current_loss:
      b += lr
    elif loss(X, Y, w, b - lr) < current_loss:
      b -= lr
    else:
      return w, b
  raise Exception('Could not converge within %d iterations.', iterations)

X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)

w, b = train(X, Y, iterations=10_000, lr=0.01)
print('w = %.3f, b = %.3f' % (w, b))
print('Prediction: x = %d => y = %.2f' % (20, predict(20, w, b)))

sns.set()                                                # activate Seaborn
plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
plt.xticks(fontsize=15)                                  # set x axis ticks
plt.yticks(fontsize=15)                                  # set y axis ticks
plt.xlabel("Reservations", fontsize=15)                  # set x axis label
plt.ylabel("Pizzas", fontsize=15)                        # set y axis label
plt.plot(X, Y, "bo")                                     # plot data
x = np.linspace(0, 50, 100)
y = x * w + b
plt.plot(x, y, '-r', label='y = {w} * w + {b}')
plt.show()                                               # display chart
