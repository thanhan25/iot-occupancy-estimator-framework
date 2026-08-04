import os
import pytest
import pandas as pd
from generate_mock_data import generate_telemetry
from src.data_ingestion import ingest_and_synchronize
from src.feature_engineering import engineer_features
from src.model_training import train_occupancy_model

# Pytest fixture to ensure fresh data is generated before tests run
@pytest.fixture(scope="module")
def setup_environment():
    # Generate a smaller 2-day sample specifically for testing
    generate_telemetry(days=2, freq='1min')
    yield

def test_telemetry_generation(setup_environment):
    """Verifies that the mock data generator successfully outputs realistic telemetry."""
    file_path = 'data/raw/sensor_telemetry.csv'
    assert os.path.exists(file_path), "Raw telemetry file was not created."
    
    df = pd.read_csv(file_path)
    assert not df.empty, "Generated telemetry dataframe is empty."
    assert 'co2_ppm' in df.columns, "Missing critical CO2 sensor column."
    assert 'occupancy' in df.columns, "Missing target variable 'occupancy'."

def test_data_ingestion_synchronization(setup_environment):
    """Verifies that the ingestion engine correctly resamples and aligns timestamps."""
    df = ingest_and_synchronize('data/raw/sensor_telemetry.csv')
    
    assert df.index.name == 'timestamp', "Index was not properly set to timestamp."
    assert df.isna().sum().sum() == 0, "Forward-fill failed; Null values detected in synchronized data."

def test_feature_engineering_logic(setup_environment):
    """Verifies the creation of rolling windows and rate-of-change (RoC) metrics."""
    df = engineer_features('data/raw/sensor_telemetry.csv')
    
    assert 'co2_ppm_rolling_mean' in df.columns, "Rolling mean feature was not engineered."
    assert 'luminance_lux_roc' in df.columns, "Rate of change (RoC) feature was not engineered."
    assert len(df) > 0, "Feature engineering dropped all rows."

def test_model_compilation(setup_environment):
    """Verifies that the Random Forest model trains and serializes to disk."""
    df = engineer_features('data/raw/sensor_telemetry.csv')
    train_occupancy_model(df)
    
    assert os.path.exists('models/occupancy_rf_model.pkl'), "Model binary was not saved to the models directory."