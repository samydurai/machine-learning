from sklearn.preprocessing import StandardScaler

from linear_regression_basic_impl.gradiant_descent.gradient_descent import GradientDescent
from linear_regression_basic_impl.linear_regression_dataset_parser import DataSetParser
from linear_regression_basic_impl.plot.perfect_line import PerfectLine
from linear_regression_basic_impl.plot.plot_box_plot import BoxPlot
from linear_regression_basic_impl.plot.plot_cost_curve import CostCurve
from linear_regression_basic_impl.plot.plot_histogram import PlotHistogram
from linear_regression_basic_impl.plot.plot_input import VisualizeInput

from linear_regression_basic_impl.plot.plot_pairs import PairPlot
import numpy as np
import pandas as pd

dataset_slug = "nikhil7280/student-performance-multiple-linear-regression"
file_name_in_dataset = "Student_Performance.csv"
x_train_features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
y_train_label = 'Performance Index'
dataset_parser = DataSetParser()
x_train_df, y_train_df, df = dataset_parser.parse(dataset_slug, file_name_in_dataset, x_train_features, y_train_label)
x_train_df = pd.get_dummies(x_train_df, columns=['Extracurricular Activities'], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=['Extracurricular Activities'], drop_first=True, dtype=int)
x_train_features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Extracurricular Activities_Yes']
x_train, y_train = x_train_df.values, y_train_df.values
plot_input = VisualizeInput()
plot_input.visualize(x_train, y_train, x_train_features, y_train_label)
plot_histogram = PlotHistogram()
plot_histogram.plot(df, x_train_features, y_train_label)
box_plot = BoxPlot()
box_plot.plot_box_plots(df, x_train_features, y_train_label)
pair_plot = PairPlot()
pair_plot.plot_all_features_pair_plot(df)

#Based on plotting, hours studied and previous scores shows strong correlation.

x_train_df = x_train_df.drop(columns = ['Sleep Hours', 'Sample Question Papers Practiced'])
x_train, y_train = x_train_df.values, y_train_df.values
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
print(f'Shape of feature1 is {x_train.shape}')
gd = GradientDescent()
weights, bias, iterations, cost = gd.fit_without_regularization(x_train[1:7000], y_train[1:7000], 0.01, 50, 500)
cost_curve = CostCurve()
cost_curve.plot(iterations, cost)

y_test = y_train[7000:]
y_pred = []
for i in range(7000, x_train.shape[0]):
    y_pred.append(np.dot(weights, x_train[i]) + bias)

perfect_line_fn = PerfectLine()
perfect_line_fn.plot_perfect_line(y_test, y_pred)




