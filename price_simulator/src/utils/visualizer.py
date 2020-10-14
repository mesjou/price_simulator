from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate(data: List, desired_length: int = 100) -> pd.Series:
    """Convert data to pandas series and group data to desired length."""
    if len(data) > desired_length:
        df = pd.Series(data, index=np.floor(np.linspace(1, desired_length + 1, len(data), endpoint=False)))
        average_df = df.groupby(df.index).mean()
        average_df.index = average_df.index * len(df) / desired_length
        return average_df
    else:
        return pd.Series(data)


def create_subplot(yy: List, label: str, ax=None, agg: bool = False):
    palette = plt.get_cmap("Set1")
    ax = ax or plt.gca()
    num = 0
    for y in yy:
        num += 1
        if agg:
            ax.plot(aggregate(y), marker="", color=palette(num), linewidth=1, alpha=0.9, label="Agent {}".format(num))
        else:
            ax.plot(y, marker="", color=palette(num), linewidth=1, alpha=0.9, label="Agent {}".format(num))

    ax.legend(loc=2, ncol=2)
    ax.set(xlabel="Period", ylabel=label)

    return ax
