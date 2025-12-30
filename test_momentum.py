#plot correlations - time series momentum test 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from statsmodels.tsa.stattools import acf

lookback = np.arange(1, 30)
holding = np.arange(1,5)

def test_correlation(prices: pd.Series, lookback, holding) -> pd.DataFrame:
    results = {'lookback': [], 'holding': [], 'correlation': []}
    for lb in lookback: 
        for h in holding: 
            results['lookback'].append(lb)
            results['holding'].append(h)
            #calculate correlation 
            lookback_prices = prices.shift(lb)
            hold_prices = prices.shift(-h)
            lookback_returns = (prices - lookback_prices)/lookback_prices
            hold_returns = (hold_prices - prices)/prices 
            results['correlation'].append(lookback_returns.corr(hold_returns))
    
    return pd.DataFrame(results)

import numpy as np
import pandas as pd

def block_permutation_test(x, y, block_size=30, n_permutations=10000):
    """
    Vectorized block permutation test for two pandas Series.
    
    Parameters:
    -----------
    x : pd.Series (predictor, e.g., imbalance)
    y : pd.Series (outcome, e.g., forward returns)
    block_size : int
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
    
    # Observed correlation
    observed_corr = np.corrcoef(x_vals, y_vals)[0, 1]
    
    # Setup blocks
    n_blocks = n // block_size
    remainder = n % block_size
    
    # Reshape y into blocks (ignore remainder for now)
    y_blocks = y_vals[:n_blocks * block_size].reshape(n_blocks, block_size)
    
    # Pre-allocate results
    permuted_corrs = np.zeros(n_permutations)
    
    print(f"Running {n_permutations} permutations...")
    
    for i in tqdm(range(n_permutations)):
        # Shuffle block indices
        shuffled_indices = np.random.permutation(n_blocks)
        
        # Reconstruct y by shuffling blocks
        y_permuted = y_blocks[shuffled_indices].flatten()
        
        # Handle remainder if exists
        if remainder > 0:
            # Add remainder from random block
            random_block_idx = np.random.randint(0, n_blocks)
            y_permuted = np.concatenate([y_permuted, 
                                         y_blocks[random_block_idx, :remainder]])
        
        # Calculate correlation (vectorized)
        permuted_corrs[i] = np.corrcoef(x_vals, y_permuted)[0, 1]
    
    # Calculate p-value
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
    threshold = 1.96 / np.sqrt(n)
    
    # Find first insignificant lag
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) < threshold:
            print(f"ACF insignificant at lag {lag}")
            print(f"Block size: {lag}")
            return lag
    
    # If never insignificant, use max
    print(f"ACF still significant at lag {max_lag}")
    print(f"Block size: {max_lag}")
    return max_lag

    