from matplotlib import pyplot as plt 
import numpy as np
from scipy.stats import spearmanr
BASIS_POINTS_CONVERSION = 10_000

def plot_train_val_distributions(df, features, target, train_idx, val_idx):
    train, val = df.iloc[train_idx], df.iloc[val_idx]
    
    cols = features + [target]
    fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols)*2, 8))
    if len(cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols):
        ax.hist(train[col].dropna(), bins=50, alpha=0.6, color='steelblue', label='Train', density=True)
        ax.hist(val[col].dropna(), bins=50, alpha=0.6, color='red', label='Val', density=True)
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('Train vs Val distributions', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_train_val_scatter(df, feature, target, train_idx, val_idx, bp_conversion=10_000):
    train, val = df.iloc[train_idx], df.iloc[val_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(train[feature], train[target]*bp_conversion, alpha=0.1, s=3, 
               color='steelblue', label='Train')
    ax.scatter(val[feature], val[target]*bp_conversion, alpha=0.1, s=3, 
               color='red', label='Val')
    
    # regression lines - much easier to see the relationship
    for data, color, label in [(train, 'blue', 'Train trend'), (val, 'darkred', 'Val trend')]:
        m, b = np.polyfit(data[feature].dropna(), (data[target]*bp_conversion).dropna(), 1)
        x = np.linspace(data[feature].min(), data[feature].max(), 100)
        ax.plot(x, m*x + b, color=color, lw=2, label=f'{label} (slope={m:.2f})')
    
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(0.5, color='black', lw=0.8, ls='--')  # midpoint of 0-1 range
    
    ax.set_xlabel(feature)
    ax.set_ylabel(f'{target} (bps)')
    ax.set_title(f'{feature} vs {target}')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_rolling_correlation(df, feature, target, train_idx=None, val_idx=None, window=100):
    if train_idx is not None and val_idx is not None:
        fold_df = df.iloc[list(train_idx) + list(val_idx)].copy()
        split_time = df.index[val_idx[0]]
    else:
        fold_df = df.copy()
        split_time = None
    
    rolling_corr = fold_df[feature].rolling(window).corr(fold_df[target])
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(fold_df.index, rolling_corr, color='steelblue', lw=1, label=f'Rolling {window}-bar correlation')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    
    if split_time is not None:
        ax.axvline(split_time, color='red', lw=1.5, ls='--', label='Train/Val split')
        ax.axvspan(fold_df.index[0], split_time, alpha=0.05, color='green', label='Train')
        ax.axvspan(split_time, fold_df.index[-1], alpha=0.05, color='red', label='Val')
    
    ax.set_title(f'Rolling correlation: {feature} vs {target}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Correlation')
    ax.legend()
    fig.tight_layout()


def plot_rolling_mae(df, target, train_idx=None, val_idx=None, window=100, multiplier = 1):
    if train_idx is not None and val_idx is not None:
        fold_df = df.iloc[list(train_idx) + list(val_idx)].copy()
        split_time = df.index[val_idx[0]]
    else:
        fold_df = df.copy()
        split_time = None
    
    rolling_mae_median = fold_df[target].rolling(window).median()*multiplier
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(fold_df.index, rolling_mae_median, color='steelblue', lw=1, label=f'Rolling {window}-bar median')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    
    if split_time is not None:
        ax.axvline(split_time, color='red', lw=1.5, ls='--', label='Train/Val split')
        ax.axvspan(fold_df.index[0], split_time, alpha=0.05, color='green', label='Train')
        ax.axvspan(split_time, fold_df.index[-1], alpha=0.05, color='red', label='Val')
    
    ax.set_title(f'Rolling median: {target} (multiplier applied = {multiplier})')
    ax.set_ylabel('Median')
    ax.legend()
    fig.tight_layout()


def plot_rolling_spearman(df, feature, target, train_idx=None, val_idx=None, window=500):
    if train_idx is not None and val_idx is not None:
        fold_df = df.iloc[list(train_idx) + list(val_idx)].copy()
        split_time = df.index[val_idx[0]]
    else:
        fold_df = df.copy()
        split_time = None

    ranked = fold_df[[feature, target]].rank()
    corrs = ranked[feature].rolling(window).corr(ranked[target])

    q10, q90 = corrs.quantile(0.1), corrs.quantile(0.9)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(fold_df.index, corrs, color='steelblue', lw=1)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axhline(q10, color='red', lw=1, ls=':', label=f'10th pct ({q10:.2f})')
    ax.axhline(q90, color='green', lw=1, ls=':', label=f'90th pct ({q90:.2f})')

    if split_time is not None:
        ax.axvline(split_time, color='red', lw=1.5, ls='--', label='Train/Val split')
        ax.axvspan(fold_df.index[0], split_time, alpha=0.05, color='green', label='Train')
        ax.axvspan(split_time, fold_df.index[-1], alpha=0.05, color='red', label='Val')

    ax.set_title(f'Rolling Spearman: {feature} vs {target} (window={window})')
    ax.set_ylabel('Spearman correlation')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_train_val_scores(results: dict, baseline_model: dict): 
    fig, ax = plt.subplots(figsize=(12, 5))
    folds = np.arange(1, len(results['val_scores']) + 1)
    width = 0.35

    ax.bar(folds - width/2, results['train_scores'], width=width, label='Train', alpha=0.7, color='steelblue')
    ax.bar(folds + width/2, results['val_scores'], width=width, label='Val', alpha=0.7, color='red')

    #ax.axhline(BASELINE_PERFORMANCE, color='black', lw=1.5, ls='--', label='Baseline')

    # step plot gives a per-fold horizontal line that changes each fold
    ax.step(folds, baseline_model['train_scores'], where='mid',color='steelblue', lw=1.5, ls='--', label='Baseline Train')
    ax.step(folds, baseline_model['val_scores'], where='mid',color='red', lw=1.5, ls='--', label='Baseline Val')

    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE (bps)')
    ax.set_title('QR Train vs Val MAE across folds')
    ax.set_xticks(folds)
    ax.legend()
    plt.tight_layout()
    plt.show()