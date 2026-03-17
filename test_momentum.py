import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
from scipy import stats
from scipy.stats import pearsonr

CONFIDENCE_BOUND = 1.96

def simple_momentum_backtest(prices: pd.Series, lookback: int, holding: int): 
    lookback_ret = (prices - prices.shift(lookback)) / prices.shift(lookback)
    forward_ret  = (prices.shift(-holding) - prices) / prices

    #impose non-overlapping periods 
    step = max(lookback, holding)
    lookback_ret = lookback_ret.iloc[lookback::step]
    forward_ret  = forward_ret.iloc[lookback::step]

    mask = lookback_ret.notna() & forward_ret.notna()
    lr, fr = lookback_ret[mask], forward_ret[mask]

    # signal: long if positive lookback, short if negative
    direction = np.sign(lr)
    ret = direction * fr
    
    print(f"lb={lookback}, h={holding} | "
          f"mean={ret.mean()*10000:.2f} bps | "
          f"hit={( ret > 0).mean():.2%} | "
          f"n={len(ret)}")
    
    return ret

def test_correlation(prices: pd.Series, lookback, holding, 
                     method: str = 'pearson',  # 'pearson' or 'spearman'
                     n_permutations: int = 0,  # 0 = no permutation test
                     ) -> pd.DataFrame:
    '''Calculate correlation between lookback and holding periods.
    
    method: 'pearson' or 'spearman'
    n_permutations: if > 0, compute permutation p-value (recommended for spearman)
    '''
    results = {'lookback': [], 'holding': [], 'correlation': [], 'pvalue': [], 'n': []}
    if n_permutations > 0:
        results['perm_pvalue'] = []

    for lb in lookback: 
        for h in holding:
            step = max(lb, h)
            
            lookback_returns = (prices - prices.shift(lb)) / prices.shift(lb)
            hold_returns = (prices.shift(-h) - prices) / prices
            
            lookback_returns = lookback_returns.iloc[lb::step]
            hold_returns = hold_returns.iloc[lb::step]
            
            mask = lookback_returns.notna() & hold_returns.notna()
            lr = lookback_returns[mask].values
            hr = hold_returns[mask].values
            
            # compute correlation
            if method == 'pearson':
                r, p = stats.pearsonr(lr, hr)
            elif method == 'spearman':
                r, p = stats.spearmanr(lr, hr)
            else:
                raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")
            
            results['lookback'].append(lb)
            results['holding'].append(h)
            results['correlation'].append(r)
            results['pvalue'].append(p)
            results['n'].append(len(lr))

            # permutation test
            if n_permutations > 0:
                perm_stats = np.array([
                    stats.spearmanr(np.random.permutation(lr), hr)[0]
                    for _ in range(n_permutations)
                ])
                perm_p = np.mean(np.abs(perm_stats) >= np.abs(r))
                results['perm_pvalue'].append(perm_p)
    
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


def calc_obi_corr(holding: list[int], df: pd.DataFrame): 
    '''Calculate correlation between instantaneous top level imbalance and forward period returns'''
    results = {}
    for horizon in holding:
        #Calculate forward returns 
        fwd_ret = df['mid'].pct_change(horizon).shift(-horizon)
        # subsample every `horizon` steps to avoid overlapping forward return windows
        idx = range(0, len(df), horizon)
        obi = df['obi'].iloc[idx]
        fr = fwd_ret.iloc[idx]
        mask = obi.notna() & fr.notna()
        r, p = pearsonr(obi[mask], fr[mask])
        results[horizon] = {'correlation': r, 'pvalue': p, 'n': mask.sum()}
    
    final = pd.DataFrame(results).T
    final.index.name = 'holding_period'
    return final

