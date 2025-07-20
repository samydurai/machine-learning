import matplotlib.pyplot as plt
class CostCurve:
    def plot(self, iterations, cost):
        n_rows, n_cols = 1, 1
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        plt.subplot(1, 1, 1)
        plt.scatter(iterations, cost, alpha=0.7)
        plt.title('iterations vs cost')
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()