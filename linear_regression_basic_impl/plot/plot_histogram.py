import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PlotHistogram:
    def plot(self, dataframe, features, output_feature=None, bins=30, figsize=(15, 10)):
        num_plots = len(features)
        if output_feature:
            num_plots += 1
            plot_features = features + [output_feature]
        else:
            plot_features = features

        # Determine grid size for subplots
        # We'll try to make it somewhat square-ish
        n_cols = 3  # You can adjust this for more or fewer columns
        n_rows = (num_plots + n_cols - 1) // n_cols  # Calculate rows needed

        plt.figure(figsize=figsize)

        for i, col in enumerate(plot_features):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(dataframe[col], kde=True, bins=bins, color='skyblue' if col != output_feature else 'salmon')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(axis='y', alpha=0.75)  # Add grid lines for better readability

        plt.tight_layout()  # Adjust subplot parameters for a tight layout
        plt.show()