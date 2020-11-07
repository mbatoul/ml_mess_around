import numpy as np

def predict(X, w, b):
  return w * X + b

def loss(X, Y, w, b):
  return np.average((predict(X, w, b) - Y) ** 2)

def train(X, Y, iterations, lr):
  w = b = 0
  for _ in range(iterations):
    curr_loss = loss(X, Y, w, b)
    if curr_loss > loss(X, Y, w + lr, b):
      w += lr
    elif curr_loss > loss(X, Y, w - lr, b):
      w -= lr
    elif curr_loss > loss(X, Y, w, b + lr):
      b += lr
    elif curr_loss > loss(X, Y, w, b - lr):
      b -= lr
    else:
      return w, b
  raise Exception(f'Did not converge in {iterations}')

X, Y = np.loadtxt('data.txt', skiprows=1, unpack=True)
X_train, Y_train = X[::2], Y[::2]
X_test, Y_test = X[1::2], Y[1::2]

iterations, lr = 100_000, 0.01
w, b = train(X_train, Y_train, iterations, lr)
print('w: %.3f, b: %.3f' % (w, b))

Y_pred = predict(X_test, w, b)
acc = 100.0 * len(set(Y_pred) & set(Y_test)) / len(Y_test)
print('Accuracy: %.3f' % acc)
