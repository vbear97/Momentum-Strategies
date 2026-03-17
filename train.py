import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm.notebook import tqdm
from xgboost import XGBRegressor 

BASIS_POINTS_CONVERSION = 10_000
ROUND_TRIP_FEE = 20
TIME_TICK = 10 
BASELINE_PREDICTION = 0

def train_test_split(data_length: int, num_splits: int = 10, split_ratio = 0.8) -> list[(pd.Index, pd.Index)]: 
    '''Split time series data into data into non-overlapping train-test sets. '''
    numbered_index = np.arange(data_length)
    chunks = np.array_split(numbered_index, num_splits) #use positional index instead 
    tss = [
        (c[0:int(len(c)*split_ratio)], c[int(len(c)*split_ratio):]) #split up into train, test splits 
        for c in chunks 
    ]
    return tss 

def train_test_split_by_day(df: pd.DataFrame, num_splits: int = 10, split_ratio: float = 0.8) -> list:
    '''Split time series into non-overlapping sets, taking care not to saw through day'''
    days = df.groupby(df.index.date).apply(lambda x: x.index)
    day_chunks = np.array_split(np.arange(len(days)), num_splits)
    
    splits = []
    for chunk in day_chunks:
        chunk_days = days.iloc[chunk]
        
        split_point = int(len(chunk) * split_ratio)
        train_days = chunk_days.iloc[:split_point]
        val_days = chunk_days.iloc[split_point:]
        
        train_idx = np.concatenate([df.index.get_indexer(d) for d in train_days])
        val_idx = np.concatenate([df.index.get_indexer(d) for d in val_days])
        splits.append((train_idx, val_idx))
    
    return splits

def rolling_cv_split(data_length: int, train_size, test_size, step_size = None): 
    '''Split time series data into overlapping train-test sets.'''
    if step_size is None: 
        step_size = test_size
    numbered_index = np.arange(data_length)
    splits = []
    start_idx = range(0, data_length - train_size - test_size + 1, step_size)
    for start in start_idx: 
        train_idx = numbered_index[start: start + train_size]
        val_idx = numbered_index[start+train_size : start + train_size + test_size]
        splits.append((train_idx, val_idx))
    return splits


def get_cv_splitter(df: pd.DataFrame, cv: str, **kwargs): 
    '''Factory function that returns appropriate CV splitter'''
    data_length = len(df)
    if cv == 'split_day': 
        return train_test_split_by_day(df,
                                num_splits = kwargs.get('num_splits', 10), 
                                split_ratio = kwargs.get('split_ratio', 0.8)
                                )

    if cv == 'split': 
        return train_test_split(data_length,
                                num_splits = kwargs.get('num_splits', 10), 
                                split_ratio = kwargs.get('split_ratio', 0.8)
                                )
    elif cv == 'rolling': 
        return rolling_cv_split(data_length, 
                                train_size = kwargs['train_size'], 
                                test_size = kwargs['test_size']
                                )
    
    elif cv == 'expanding': 
        return TimeSeriesSplit(n_splits = kwargs.get('num_splits', 10)).split(df)
    
    else:
        raise ValueError(f"Unknown cv type '{cv}', use 'split', 'rolling', or 'expanding'")


def get_model(model_type: str, **kwargs):
    """Factory function that returns the appropriate model."""
    if model_type == 'xgb':
        return XGBRegressor(**kwargs)
    elif model_type == 'quantile':
        return QuantileRegressor(quantile=0.5, **kwargs)
    elif model_type == 'baseline': 
        return DummyClassifier(strategy = 'constant', constant = BASELINE_PREDICTION)
    else:
        raise ValueError(f"Unknown model_type '{model_type}', use 'xgb' or 'quantile'")
    
def check_consecutive(df: pd.DataFrame, fold_type: str):
    expected = pd.Timedelta(seconds=TIME_TICK)
    gaps = df.index.to_series().diff().dropna()
    bad_gaps = gaps[gaps != expected]
    if len(bad_gaps) > 0:
        raise ValueError(f"{fold_type} fold has irregular timestamps:\n{bad_gaps.value_counts()}")
    
    
def walk_forward_validate(
    df: pd.DataFrame, 
    features: list[str], 
    target: str, 
    model_type: str = 'xgb',
    cv: str = 'split', 
    model_kwargs: dict = None, 
    cv_kwargs: dict = None,                  
):
    model_kwargs = model_kwargs or {}
    cv_kwargs = cv_kwargs or {}

    val_scores = []
    train_scores = []
    models = []
    
    tss = get_cv_splitter(df=df, cv=cv, **cv_kwargs)

    for train_idx, val_idx in tqdm(tss, total=len(tss), leave=False): 
        train, val = df.iloc[train_idx], df.iloc[val_idx]    
        X_train, y_train = train[features], train[target]

        # 3-way split: early stopping on first half of val, score on second half
        if model_type == 'xgb':
            mid = len(val) // 2 #50-50% data split
            es_val, score_val = val.iloc[:mid], val.iloc[mid:]
            X_es, y_es = es_val[features], es_val[target]
            X_val, y_val = score_val[features], score_val[target]
        else:
            X_val, y_val = val[features], val[target]

        model = get_model(model_type, **model_kwargs)

        if model_type == 'xgb':
            model.fit(X_train, y_train, 
                     eval_set=[(X_train, y_train), 
                               (X_val, y_val), 
                               (X_es, y_es)], #used to evaluate early stopping 
                     verbose=False)
        else:
            model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_scores.append(mean_absolute_error(y_train, y_pred_train))
        val_scores.append(mean_absolute_error(y_val, y_pred_val))
        models.append(model)

    val_scores = np.array(val_scores) * BASIS_POINTS_CONVERSION
    train_scores = np.array(train_scores) * BASIS_POINTS_CONVERSION

    return {
        'val_mean': np.mean(val_scores),
        'train_mean': np.mean(train_scores),
        'val_scores': val_scores, 
        'val_std': np.std(val_scores), 
        'train_scores': train_scores,
        'train_std': np.std(train_scores),
        'models': models, 
    }





