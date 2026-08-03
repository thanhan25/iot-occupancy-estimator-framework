import pandas as pd
import logging

logger = logging.getLogger(__name__)

def ingest_and_synchronize(file_path: str, freq: str = '1min') -> pd.DataFrame:
    """
    Ingests raw IoT telemetry and synchronizes asynchronous sensor streams.
    Real sensors ping at different intervals; this resamples them to a unified time grid.
    """
    logger.info(f"Ingesting raw IoT telemetry from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Synchronizing asynchronous data streams to {freq} intervals...")
        
        # Resample to fixed frequency. 
        # Forward-fill missing values for continuous metrics (CO2, Temp, Luminance)
        # Fill missing PIR motion events with 0 (no motion detected)
        resampled_df = df.resample(freq).ffill()
        if 'pir_motion' in resampled_df.columns:
            resampled_df['pir_motion'] = resampled_df['pir_motion'].fillna(0)
            
        resampled_df.dropna(inplace=True)
        logger.info(f"Ingestion complete. Synchronized shape: {resampled_df.shape}")
        
        return resampled_df
    except Exception as e:
        logger.error(f"Critical failure during data ingestion: {e}")
        raise