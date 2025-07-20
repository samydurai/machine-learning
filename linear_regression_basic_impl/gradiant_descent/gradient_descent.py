import numpy as np

from linear_regression_basic_impl.gradiant_descent.gradient_function import GradientFunction
from linear_regression_basic_impl.costfn.mean_sq_cost_function import MSECostFunction


class GradientDescent:
    def fit_without_regularization(self, x_train, y_train, learning_rate, epochs, batch_size):
        m, n = x_train.shape[0], x_train.shape[1]
        weights, bias = np.full(n, 0.01), 0.01
        gradient_fn, cost_fn = GradientFunction(), MSECostFunction()
        iterations = []
        cost = []
        curr_itr = 0
        iterations.append(curr_itr)
        cost.append(cost_fn.compute_cost(x_train, y_train, weights, bias))
        for e in range(epochs):
            permutation = np.random.permutation(m)
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]
            for i in range(0, m, batch_size):
                batch_end_index = max(m, i + batch_size)
                x_batch = x_train_shuffled[i:batch_end_index]
                y_batch = y_train_shuffled[i:batch_end_index]
                d_dw, d_db = gradient_fn.compute_gradient(x_batch, y_batch, weights, bias)
                weights = weights - (learning_rate * d_dw)
                bias = bias - (learning_rate * d_db)
                curr_itr += 1
                iterations.append(curr_itr)
                cost.append(cost_fn.compute_cost(x_train, y_train, weights, bias))
        return weights, bias, iterations, cost

    def fit_with_regularization(self, x_train, y_train, learning_rate, epochs, batch_size, _lambda):
        m, n = x_train.shape[0], x_train.shape[1]
        weights, bias = np.full(n, 0.01), 0.01
        gradient_fn, cost_fn = GradientFunction(), MSECostFunction()
        iterations = []
        cost = []
        curr_itr = 0
        iterations.append(curr_itr)
        cost.append(cost_fn.compute_cost(x_train, y_train, weights, bias))
        for e in range(epochs):
            permutation = np.random.permutation(m)
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]
            for i in range(m):
                batch_end_index = max(m, i + batch_size)
                x_batch = x_train_shuffled[i:batch_end_index]
                y_batch = y_train_shuffled[i:batch_end_index]
                d_dw, d_db = gradient_fn.compute_gradient_with_regularization(x_batch, y_batch, weights, bias, _lambda)
                weights = weights - (learning_rate * d_dw)
                bias = bias - (learning_rate * d_db)
                curr_itr += 1
                iterations.append(curr_itr)
                cost.append(cost_fn.compute_cost(x_train, y_train, weights, bias))
        return weights, bias, iterations, cost
