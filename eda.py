import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import HOLDING_PERIODS

def plot_feature(df, feature, target, figsize = (20, 5)): 
    fig, ax = plt.subplots(ncols = 4, figsize = figsize)
    #First plot: representative feature 
    plot_kde(df, col = feature, ax = ax[0])
    #Second plot: scatterplot 
    hb = ax[1].hexbin(df[feature], df[target], gridsize=50, bins = 'log')
    cb = fig.colorbar(hb, ax=ax[1], label='counts')
    ax[1].set_title(f"hex scatterplot x = {feature}, y = {target}")
    #Quantile plot 
    qq_plot(df, feature, target, ax = ax[2])
    fig.tight_layout()
    #returns plot 
    plot_returns(df, feature, ax[3])

def plot_scatter_hex(df, feature, target, fig = None, ax=None): 
    if ax is None: 
        fig, ax = plt.subplots()
        #Second plot: scatterplot 
        hb = ax.hexbin(df[feature], df[target], gridsize=50, bins = 'log')
        cb = fig.colorbar(hb, ax=ax, label='counts')
    else: 
        hb = ax.hexbin(df[feature], df[target], gridsize=50, bins = 'log')
        cb = fig.colorbar(hb, ax=ax, label='counts')

def qq_plot(df, x, y, n_bins = 20, quantile = 0.5, ax = None): 
    data = df.copy()
    data['bin'] = pd.cut(df[x], bins = n_bins)
    binned = data.groupby('bin')[y].quantile(q = quantile)
    bin_centers = [interval.mid for interval in binned.index]

    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(bin_centers, binned.values, lw=2, marker='o', ms=4)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_xlabel(x)
        ax.set_ylabel(f'{y, quantile*100} percentile')
        ax.set_title(f'{quantile*100}th percentile {y} by {x} bin')
    else: 
        ax.plot(bin_centers, binned.values, lw=2, marker='o', ms=4)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_xlabel(x)
        ax.set_ylabel(f'{y, quantile*100} percentile')
        ax.set_title(f'{quantile*100}th percentile {y} by {x} bin')

def plot_kde(df, col, ax): 
    sns.kdeplot(data=df, x=col, fill=True, alpha=0.3, ax = ax)
    sns.rugplot(data=df, x=col, color='black', ax = ax)
    ax.set_title(f"Density and rugplot of {col}")

def plot_returns(df, col, ax): 
    df[[col]+ [f'fwd_ret_{h}_bp' for h in HOLDING_PERIODS]].corr('spearman')[col].iloc[1:].plot.bar(ax=ax)
    ax.set_title(f"Spearman correlations between {col} and returns")

