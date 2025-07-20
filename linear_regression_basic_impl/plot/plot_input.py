import matplotlib.pyplot as plt
import math

class VisualizeInput:
    def visualize(self, x_train, y_train, x_train_features, y_train_label):
        m, n = x_train.shape[0], x_train.shape[1]
        #assumning 3 * 3 grid, we can plot one features in one row
        n_rows, n_cols = math.ceil(n / 3), 3
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        for i in range(n):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.scatter(x_train[:, i], y_train, alpha=0.7)
            plt.title(f'{x_train_features[i]} vs {y_train_label}')
            plt.xlabel(x_train_features[i])
            plt.ylabel(y_train_label)
            plt.grid(True, linestyle = '--', alpha = 0.6)
        plt.tight_layout()
        plt.show()


