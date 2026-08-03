import pandas as pd
import yaml

def engineer_features(df_path: str, config_path: str = 'config.yaml') -> pd.DataFrame:
    """Ingests raw telemetry and engineers time-series features (lags, rolling windows)."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    df = pd.read_csv(df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    
    sensors = config['features']['sensor_columns']
    window = config['features']['rolling_window_mins']
    
    # Time-series feature engineering: Rolling averages and Rate of Change
    for col in sensors:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_roc'] = df[col].diff().fillna(0)
        
    return df.dropna()