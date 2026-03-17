import pandas as pd 

BASIS_POINTS_CONVERSION = 10_000

def make_bid_ask_features(raw_df: pd.DataFrame):
    df =  raw_df[['bid_0_price', 'ask_0_price', 'bid_0_size', 'ask_0_size']].rename(columns={
    'bid_0_price': 'bid', 
    'ask_0_price': 'ask', 
    'bid_0_size': 'bid_size', 
    'ask_0_size': 'ask_size'
    })
    df['mid'] = (df['bid'] + df['ask'])/2
    return df

def calc_net_returns(df: pd.DataFrame, holding_periods: list[int], fee: float, price_col: str = 'mid'): 
    net_returns_df = pd.DataFrame()
    for horizon in holding_periods: 
        entry_cost = df[price_col]*(1+fee)
        exit_proceeds = df[price_col].shift(-horizon) * (1 - fee)
        if fee == 0: 
            net_returns_df[f'gross_returns_{horizon}']=((exit_proceeds - entry_cost)/entry_cost)*BASIS_POINTS_CONVERSION
        else: 
            net_returns_df[f'net_returns_{horizon}']=((exit_proceeds - entry_cost)/entry_cost)*BASIS_POINTS_CONVERSION
    return net_returns_df
