import os
import yaml
from src.feature_engineering import engineer_features
from src.model_training import train_occupancy_model

def execute_pipeline():
    print("--- Starting IoT Occupancy Pipeline ---")
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    print("[1/2] Engineering Features from Time-Series Telemetry...")
    df_processed = engineer_features(config['data']['raw_path'])
    
    os.makedirs('data/processed', exist_ok=True)
    df_processed.to_csv(config['data']['processed_path'])
    
    print("[2/2] Training & Evaluating Classification Model...")
    os.makedirs('models', exist_ok=True)
    train_occupancy_model(df_processed)
    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    execute_pipeline()