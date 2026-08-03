import os
import yaml
import logging
from src.data_ingestion import ingest_and_synchronize
from src.feature_engineering import engineer_features
from src.model_training import train_occupancy_model
from src.visualization import plot_sensor_fusion

# ==============================================================================
# ENTERPRISE LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OccupancyFramework")

def execute_pipeline():
    logger.info("Starting LLEC IoT Occupancy Framework Pipeline...")
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Phase 1: Real-World Ingestion & Synchronization
    df_raw = ingest_and_synchronize(config['data']['raw_path'])
    
    # Phase 2: Visual Analytics Generation
    # We plot a 24-hour slice of the data so the graph is readable
    plot_sensor_fusion(df_raw.head(24 * 60)) 
    
    # Phase 3: Feature Engineering (Lags, Rolling Windows)
    logger.info("Engineering time-series structural features...")
    df_processed = engineer_features(config['data']['raw_path'])
    os.makedirs('data/processed', exist_ok=True)
    df_processed.to_csv(config['data']['processed_path'])
    
    # Phase 4: Model Training & Serialization
    logger.info("Initiating Random Forest model training protocols...")
    os.makedirs('models', exist_ok=True)
    train_occupancy_model(df_processed)
    
    logger.info("Pipeline Execution Complete. System Ready.")

if __name__ == "__main__":
    execute_pipeline()