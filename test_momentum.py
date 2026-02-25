#plot correlations - time series momentum test 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
from scipy import stats

CONFIDENCE_BOUND = 1.96

def test_correlation(prices: pd.Series, lookback, holding) -> pd.DataFrame:
    '''Calculate correlation between lookback and holding, making sure observations are independent'''
    results = {'lookback': [], 'holding': [], 'correlation': [], 'pvalue': [], 'n': []}
    for lb in lookback: 
        for h in holding:
            step = max(lb, h) #Avoid any overlap between pairs 
            
            lookback_returns = (prices - prices.shift(lb)) / prices.shift(lb)
            hold_returns = (prices.shift(-h) - prices) / prices
            
            lookback_returns = lookback_returns.iloc[lb::step]
            hold_returns = hold_returns.iloc[lb::step]
            
            # drop NaNs jointly
            mask = lookback_returns.notna() & hold_returns.notna()
            lr = lookback_returns[mask]
            hr = hold_returns[mask]
            
            r, p = stats.pearsonr(lr, hr)
            
            results['lookback'].append(lb)
            results['holding'].append(h)
            results['correlation'].append(r)
            results['pvalue'].append(p)
            results['n'].append(len(lr))
    
    return pd.DataFrame(results)

def block_permutation_test(x, y, block_size=None, n_permutations=10000):
    """
    Vectorized block permutation test for two pandas Series.
    
    Parameters:
    -----------
    x : pd.Series (predictor, e.g., imbalance)
    y : pd.Series (outcome, e.g., forward returns)
    block_size : int, if None will be determined by ACF
    n_permutations : int
    
    Returns:
    --------
    dict with results
    """
    # Align and clean
    data = pd.DataFrame({'x': x, 'y': y}).dropna()
    x_vals = data['x'].values
    y_vals = data['y'].values
    n = len(x_vals)
    
    # Determine block size from ACF if not provided
    if block_size is None:
        block_size = get_block_size(data['y'])
    
    # Observed correlation
    observed_corr = np.corrcoef(x_vals, y_vals)[0, 1]
    
    # Setup blocks - trim both x and y to exact multiple of block_size
    n_blocks = n // block_size
    trim = n_blocks * block_size
    x_vals = x_vals[:trim]
    y_vals = y_vals[:trim]
    
    y_blocks = y_vals.reshape(n_blocks, block_size)
    
    # Pre-allocate results
    permuted_corrs = np.zeros(n_permutations)
    
    print(f"n={n}, block_size={block_size}, n_blocks={n_blocks}")
    print(f"Running {n_permutations} permutations...")
    
    for i in tqdm(range(n_permutations)):
        # Shuffle block indices and reconstruct y
        shuffled_indices = np.random.permutation(n_blocks)
        y_permuted = y_blocks[shuffled_indices].flatten()
        
        # Calculate correlation
        permuted_corrs[i] = np.corrcoef(x_vals, y_permuted)[0, 1]
    
    # Calculate two-sided p-value
    p_value = np.mean(np.abs(permuted_corrs) >= np.abs(observed_corr))
    
    return {
        'observed_correlation': observed_corr,
        'p_value': p_value,
        'permutation_distribution': permuted_corrs,
        'permutation_mean': permuted_corrs.mean(),
        'permutation_std': permuted_corrs.std()
    }


def get_block_size(series, max_lag=500):
    """
    Find first lag where ACF becomes insignificant.
    That's your block size.
    """
    clean = series.dropna().values
    n = len(clean)
    
    # Calculate ACF
    acf_vals = acf(clean, nlags=max_lag, fft=True)
    
    # 95% confidence bound
    threshold = CONFIDENCE_BOUND / np.sqrt(n)
    
    # Find first insignificant lag
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) < threshold:
            print(f"ACF insignificant at lag {lag}")
            print(f"Block size: {lag}")
            return lag
    
    print(f"ACF still significant at lag {max_lag}, using max_lag as block size")
    return max_lag

    