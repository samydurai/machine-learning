import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class BoxPlot:
    def plot_box_plots(self, dataframe, numerical_features, output_feature=None, categorical_feature=None, figsize=(15, 10)):
        if categorical_feature:
            print(f"\nPlotting box plots for numerical data against '{categorical_feature}'...")
            plot_cols = numerical_features
            if output_feature:
                plot_cols.append(output_feature)

            # Determine grid size for subplots
            n_cols = 2  # Usually good for categorical comparisons
            n_rows = (len(plot_cols) + n_cols - 1) // n_cols

            plt.figure(figsize=figsize)
            for i, col in enumerate(plot_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.boxplot(x=categorical_feature, y=col, data=dataframe, palette='viridis')
                sns.stripplot(x=categorical_feature, y=col, data=dataframe, color='black', size=3, jitter=True,
                              alpha=0.6)  # Add individual data points
                plt.title(f'{col} by {categorical_feature}')
                plt.xlabel(categorical_feature)
                plt.ylabel(col)
            plt.tight_layout()
            plt.show()

        else:  # Plotting individual box plots for numerical features
            print("\nPlotting individual box plots for numerical features...")
            num_plots = len(numerical_features)
            if output_feature:
                num_plots += 1
                plot_features = numerical_features + [output_feature]
            else:
                plot_features = numerical_features

            # Determine grid size for subplots
            n_cols = 3  # You can adjust this
            n_rows = (num_plots + n_cols - 1) // n_cols

            plt.figure(figsize=figsize)

            for i, col in enumerate(plot_features):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.boxplot(y=dataframe[col], color='lightgreen' if col != output_feature else 'salmon')
                plt.title(f'Box Plot of {col}')
                plt.ylabel(col)
                plt.grid(axis='y', alpha=0.75)  # Add grid lines

            plt.tight_layout()
            plt.show()