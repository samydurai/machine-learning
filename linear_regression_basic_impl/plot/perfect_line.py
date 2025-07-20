import matplotlib.pyplot as plt

class PerfectLine:
    def plot_perfect_line(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2,
                 label='Perfect Prediction Line')
        plt.xlabel("Actual Performance Index")
        plt.ylabel("Predicted Performance Index")
        plt.title("Actual vs. Predicted Performance Index")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.0)
        plt.show()