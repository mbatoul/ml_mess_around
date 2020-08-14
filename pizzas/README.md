# Given the number of reservations at a restaurant, how many pizzas should be prepared?

This example comes from `Programming Machine Learning, Paolo Perrotta (Programmatic Bookshelf - March 31, 2020)`.

It consists of a programmatic implementation of a simple supervised learning system. The `data.txt` file contains the input variables (number of reservations) and the labels (number of pizzas).

The idea is to find an approximating function for the number of pizzas. This model is identified by two parameters: the weight and the bias that are determined using a simple linear regression.

The system tweaks the parameters (weight and bias) during the training phase (see function train()) to appromixate the examples. It is guided by a loss() function that measures the distance between the approximated results and the ground truth: the lower the loss, the better the model. The loss is calculated using the MSE (Mean Squared Error) formula, a canonical risk function.
