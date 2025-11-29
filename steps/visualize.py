from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def visualize(X, Y):
    # figure with custom grid layout - GridSpec allows to create flexible subplot arrangements
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Linear Regression Visualizations', fontsize=16, fontweight='bold')

    plt.show()