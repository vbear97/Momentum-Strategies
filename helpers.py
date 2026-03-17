import pandas as pd 
from matplotlib import pyplot as plt


HOLDING_PERIODS =  [3, 6, 18, 30, 60, 120, 300]
ROLLING = [1, 3, 6, 10, 30, 60, 120, 300]
BASIS_POINTS_CONVERSION = 10_000


def make_features(raw_df: pd.DataFrame): 
    df = raw_df.copy()
    #Make initial features 
    df['obi'] = (df['bid_0_size'] - df['ask_0_size']) / (df['bid_0_size'] + df['ask_0_size'])
    df['mid'] = (df['ask_0_price'] + df['bid_0_price'])/2
    feas_list = ['obi']

    #make targets 
    for horizon in HOLDING_PERIODS: 
        #Calculate forward returns 
        df[f'fwd_ret_{horizon}']= df['mid'].pct_change(horizon).shift(-horizon)
        df[f'fwd_ret_{horizon}_bp'] = df[f'fwd_ret_{horizon}']*BASIS_POINTS_CONVERSION

    #instantaneous features 
    ##order size 
    df['total_bid_size'] = df[[f'bid_{level}_size' for level in range(20)]].sum(axis=1)
    df['total_ask_size'] = df[[f'ask_{level}_size' for level in range(20)]].sum(axis=1)
    df['total_size'] = df['total_bid_size'] + df['total_ask_size']
    df['total_imbalance'] = (df['total_bid_size'] - df['total_ask_size'])/df['total_size']

    feas_list.extend(['total_imbalance'])

    ##bid/ask spread - normalised as % of mid price and in basis points
    df['bid_ask_spread_0'] = ((df['ask_0_price']-df['bid_0_price'])/df['mid'])

    feas_list.extend(['bid_ask_spread_0'])

    #smoothed 
    ##order size 
    for r in ROLLING: 
        df[f'rolling_obi_{r}']=df['obi'].rolling(r).mean()
        df[f'rolling_total_imbalance_{r}']= df['total_imbalance'].rolling(r).mean()
        feas_list.extend([f'rolling_obi_{r}', 
                        f'rolling_total_imbalance_{r}'
                        ])
    ##price 
    for r in ROLLING: 
        col_name = f'rolling_bid_ask_spread_{r}'
        df[col_name] = df['bid_ask_spread_0'].rolling(r).mean()
        feas_list.append(col_name)

    #momentum 
    ##obi
    ###velocity 
    df[f'obi_velocity']= df['obi'].diff()
    df[f'total_imbalance_velocity']= df['total_imbalance'].diff()
    ###acceleration
    df[f'obi_acc'] = df['obi'].diff().diff()
    df[f'total_imbalance_acc'] = df['total_imbalance'].diff().diff()

    feas_list.extend(['obi_velocity', 
                    'total_imbalance_velocity', 
                    'obi_acc', 
                    'total_imbalance_acc'
                    ])

    #smoothed velocity 
    #TODO - How can we make the rolling averages smoother? 
    for r in ROLLING: 
        df[f'rolling_obi_{r}_velocity'] = df[f'rolling_obi_{r}'].diff()
        df[f'rolling_total_imbalance_{r}_velocity'] = df[f'rolling_total_imbalance_{r}'].diff()
        feas_list.extend([
            f'rolling_obi_{r}_velocity', 
            f'rolling_total_imbalance_{r}_velocity'
        ])

    ##price momentum 
    df['bid_ask_spread_0_velocity'] = df['bid_ask_spread_0'].diff()
    feas_list.extend([
        'bid_ask_spread_0_velocity'
    ])
    ##boolean flags
    df['imbalance_flag'] = (df['obi'] > 0).astype('category')
    feas_list.extend([
        'imbalance_flag'
    ])

    return df, feas_list 


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
