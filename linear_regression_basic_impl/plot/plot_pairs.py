import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PairPlot:
    def plot_all_features_pair_plot(self, dataframe, hue_column=None, kind='scatter', diag_kind='kde', **kwargs):
        g = sns.pairplot(
            data=dataframe,
            hue=hue_column,
            kind=kind,
            diag_kind=diag_kind,
            **kwargs
        )
        plt.suptitle('Pair Plot of All Numerical Features', y=1.02)  # Adjust y to prevent overlap with title
        plt.show()