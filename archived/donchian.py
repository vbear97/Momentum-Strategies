from datetime import datetime, time, timedelta 
import pandas as pd
import matplotlib.pyplot as plt

INTRADAY_TRADING_HOURS = timedelta(hours=7.5)
DAILY_CLOSE_TIME = time(hour = 16, minute = 00)
DAILY_OPEN_TIME = time(hour = 9, minute = 30)

def _donchian_signal(ohlc: pd.DataFrame, lookback: timedelta | int, intraday: bool = True):
    '''Assumes ohlc has a datetime index'''
    if intraday and lookback > INTRADAY_TRADING_HOURS: 
        raise ValueError("Lookback must be smaller than 1 day for intraday trading")
    
    #Use shift to account for lag between signal and execution 
    upper = ohlc['high'].rolling(lookback).max().shift(1)
    lower = ohlc['low'].rolling(lookback).min().shift(1)
    
    signal = pd.Series(pd.NA, index = ohlc.index)
    signal.loc[ohlc['close']>upper] = 1 #if you are above the highest high, enter a long position 
    signal.loc[ohlc['close']<lower] = -1 #if you are below lowest low, enter a short position 
    signal = signal.ffill()
    
    #intraday liquidation - liquidate all positions by close of day 
    #TODO - any logic for how we can close out the trading strategy? 
    if intraday:
        #fill in start/end of day 
        signal.loc[signal.index.time == DAILY_OPEN_TIME] = 0.0
        signal.loc[signal.index.time == DAILY_CLOSE_TIME] = 0.0
        
    return pd.Series(signal, name = 'Signal')

def _signal_returns(ohlc: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    '''Assumes ohlc and signal have same datetime index'''
    m2m_position = ohlc['close']*signal 
    #group on date, since intraday returns 
    pnl = m2m_position.groupby(m2m_position.index.date).diff()
    return pd.DataFrame({'m2m_position': m2m_position, 'pnl': pnl})

def _performance_metrics(returns_df: pd.DataFrame, n_bars_in_year: int): 
    #pre-compute 
    returns = returns_df['pnl']

    #metrics 
    sharpe = returns.mean()/returns.std()
    sortino = returns.mean()/returns.loc[lambda x: x<0].std()
    profit_factor = (returns[returns>0].sum())/(returns[returns<0].abs().sum())

    #annualize 
    sharpe *= n_bars_in_year**0.5
    sortino *= n_bars_in_year**0.5

    return {
        'profit_factor': profit_factor, 
        'sharpe_ratio': sharpe, 
        'sortino_ratio': sortino
    }

def donchian(ohlc: pd.DataFrame, lookback: timedelta | int, n_bars_in_year: int, intraday: bool=True): 
    signal = _donchian_signal(ohlc, lookback, intraday)
    returns_df = _signal_returns(ohlc, signal)
    metrics = _performance_metrics(returns_df, n_bars_in_year)
    return {
        'summary_df': pd.concat([signal, returns_df], axis=1), 
        'metrics': metrics
    }

def plot_donchian(ohlc: pd.DataFrame, signal: pd.Series, lookback: int, figsize=(14,8), sample_date: str =  None): 
    # Recalculate channels for visualization (lightweight)
    upper = ohlc['high'].rolling(lookback).max().shift(1)
    lower = ohlc['low'].rolling(lookback).min().shift(1)

    # Find signal changes for plotting markers
    signal_changes = signal.diff()
    
    # Buy signals: entering long (0->1 or -1->1)
    buy_signals = signal_changes[signal_changes > 0].index
    
    # Sell signals: entering short (0->-1 or 1->-1)  
    sell_signals = signal_changes[signal_changes < 0].index

    fig, ax = plt.subplots(figsize=figsize)

    if sample_date is not None: 
        ohlc = ohlc.loc[sample_date]
        buy_signals = buy_signals[buy_signals.date==sample_date]
        sell_signals = sell_signals[sell_signals.date==sample_date]
    
    # Plot price
    ax.plot(ohlc.index, ohlc['close'], label='Close', color='black', linewidth=1)
    
    # Plot Donchian channels 
    ax.plot(ohlc.index, upper, label='Upper', color='#2962FF', linewidth=1.5)
    ax.plot(ohlc.index, lower, label='Lower', color='#2962FF', linewidth=1.5)
    ax.fill_between(ohlc.index, upper, lower, alpha=0.1, color='#2962FF')
    
    if len(buy_signals) > 0:
        buy_prices = ohlc.loc[buy_signals, 'close']
        ax.scatter(buy_signals, buy_prices, 
                   marker='^', color='#00FF00', s=100, label='Long', zorder=5, edgecolors='black')
        
    if len(sell_signals) > 0:
        sell_prices = ohlc.loc[sell_signals, 'close']
        ax.scatter(sell_signals, sell_prices, 
                   marker='v', color='#FF0000', s=100, label='Short', zorder=5, edgecolors='black')
        
    ax.set_title(f'Donchian Channel (Length={lookback})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()