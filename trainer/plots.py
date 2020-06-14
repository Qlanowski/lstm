import matplotlib.pyplot as plt
import numpy as np


def create_accuracy_plot(results):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy %')
    ax.set_yticks(np.r_[0:100:10])
    if len(results) >= 16:
        ax.xaxis.set_major_locator(plt.MaxNLocator(16))
    else:
        ax.set_xticks(np.r_[0:len(results)])
    ax.grid(True, axis='y')
    ax.errorbar(
        x=results.index,
        y=results['accuracy'] * 100
    )
    return fig


def create_loss_plot(results):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value')
    if len(results) >= 16:
        ax.xaxis.set_major_locator(plt.MaxNLocator(16))
    else:
        ax.set_xticks(np.r_[0:len(results)])
    ax.grid(True, axis='y')
    ax.errorbar(
        x=results.index,
        y=results['loss']
    )
    return fig