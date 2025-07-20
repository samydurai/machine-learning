import numpy as np


class GradientFunction:
    def compute_gradient(self, x_train, y_train, weights, bias):
        m = x_train.shape[0]
        f_predicted = np.dot(x_train, weights) + bias
        error = f_predicted - y_train
        d_dw = np.dot(x_train.T, error) / m
        d_db = np.sum(error) / m
        return d_dw, d_db

    def compute_gradient_with_regularization(self, x_train, y_train, weights, bias, _lambda):
        m = x_train.shape[0]
        f_predicted = np.dot(x_train, weights) + bias
        error = f_predicted - y_train
        d_dw_non_reg = np.dot(x_train.T, error)
        d_dw_reg = weights * (_lambda / m)
        d_dw = d_dw_reg + d_dw_non_reg
        d_db = np.sum(error) / m
        return d_dw, d_db
