import unittest

import numpy as np

from mean_sq_cost_function import MSECostFunction


class MSECostFunctionTest(unittest.TestCase):
    def setUp(self):
        self.cost_fn = MSECostFunction()

    def test_cost_calculation_with_no_features(self):
        empty_np_array = np.array([])
        with self.assertRaises(ValueError):
            self.cost_fn.compute_cost(empty_np_array, empty_np_array, empty_np_array, 0.0)

    def test_calculate_cost_with_features(self):
        x_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0.5, 11.5])
        weights = np.array([1.5, -2.66])
        bias = -2.5
        cost = self.cost_fn.compute_cost(x_train, y_train, weights, bias)
        self.assertAlmostEqual(113.03, cost, 2)

    def test_calculate_regularized_cost(self):
        x_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0.5, 11.5])
        weights = np.array([1.5, -2.66])
        bias = -2.5
        _lambda = 100
        cost = self.cost_fn.compute_cost_with_regularization(x_train, y_train, weights, bias, _lambda)
        self.assertAlmostEqual(346.17, cost, 2)
