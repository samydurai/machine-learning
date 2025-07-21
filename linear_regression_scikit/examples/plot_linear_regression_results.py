import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegressionResultPlot:
    def plot_residuals(self, y_test, y_pred):
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.title('Residual Plot (Predicted vs. Residuals)')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_residual_distribution(self, y_test, y_pred):
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30)  # kde=True adds a Kernel Density Estimate curve
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(self, x_train, y_train, y_test, y_pred):
        y_train_min = y_train.min()
        y_train_max = y_train.max()

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, label='Test Set Predictions')

        min_plot_val = min(y_test.min(), y_pred.min())
        max_plot_val = max(y_test.max(), y_pred.max())
        plt.plot([min_plot_val, max_plot_val], [min_plot_val, max_plot_val],
                 color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')

        # Plot horizontal lines for y_train min and max
        plt.axhline(y=y_train_min, color='green', linestyle=':', linewidth=2, label=f'Y_train Min ({y_train_min:.2f})')
        plt.axhline(y=y_train_max, color='purple', linestyle=':', linewidth=2, label=f'Y_train Max ({y_train_max:.2f})')

        plt.title('Actual vs. Predicted Values with Training Range')
        plt.xlabel('Actual Values (Y_test)')
        plt.ylabel('Predicted Values (Y_pred)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
