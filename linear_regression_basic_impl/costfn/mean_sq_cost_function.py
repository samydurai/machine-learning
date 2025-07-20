import numpy as np


class MSECostFunction:
    def compute_cost(self, x_train, y_train, weights, bias):
        training_set_count = x_train.shape[0]
        training_label_count = y_train.shape[0]
        if training_label_count == 0 or training_label_count == 0 or training_set_count != training_label_count:
            raise ValueError("Dimensions of training data and labels are not matching or they're empty")

        f_predicted = np.dot(x_train, weights) + bias
        squared_errors = np.power(f_predicted - y_train, 2)
        cost = np.sum(squared_errors) / (2 * training_set_count)
        return cost

    def compute_cost_with_regularization(self, x_train, y_train, weights, bias, _lambda):
        mse_cost = self.compute_cost(x_train, y_train, weights, bias)
        training_set_count = x_train.shape[0]
        sum_of_weights_sq = np.sum(np.power(weights, 2))
        regularization_cost_term = (_lambda / (2 * training_set_count)) * sum_of_weights_sq
        return regularization_cost_term + mse_cost
