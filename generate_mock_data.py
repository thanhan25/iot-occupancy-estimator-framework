import pandas as pd
import numpy as np
import os

def generate_telemetry(days=7, freq='1min'):
    print("Initializing IoT Telemetry Simulation...")
    timestamps = pd.date_range(start='2026-08-03', periods=(days * 24 * 60), freq=freq)
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Simulate human behavior (Office hours 08:00 - 18:00)
    df['hour'] = df['timestamp'].dt.hour
    df['is_workday'] = df['timestamp'].dt.weekday < 5
    df['occupancy'] = np.where((df['hour'] >= 8) & (df['hour'] <= 18) & df['is_workday'], 
                               np.random.choice([0, 1], p=[0.2, 0.8], size=len(df)), 0)
    
    # Simulate Sensor Physics based on Occupancy
    df['pir_motion'] = np.where(df['occupancy'] == 1, np.random.choice([0, 1], p=[0.1, 0.9], size=len(df)), 0)
    df['luminance_lux'] = np.where(df['occupancy'] == 1, np.random.normal(400, 50, len(df)), np.random.normal(50, 10, len(df)))
    
    # CO2 and Temp have lag (Cumulative effects)
    co2_base, temp_base = 400.0, 20.0
    co2, temp = [co2_base], [temp_base]
    
    for i in range(1, len(df)):
        if df['occupancy'].iloc[i] == 1:
            co2.append(min(co2[-1] + np.random.normal(5, 1), 1200))
            temp.append(min(temp[-1] + np.random.normal(0.1, 0.02), 24.0))
        else:
            co2.append(max(co2[-1] - np.random.normal(10, 2), 400))
            temp.append(max(temp[-1] - np.random.normal(0.1, 0.02), 20.0))
            
    df['co2_ppm'] = co2
    df['temperature_c'] = temp
    
    os.makedirs('data/raw', exist_ok=True)
    df.drop(columns=['hour', 'is_workday']).to_csv('data/raw/sensor_telemetry.csv', index=False)
    print("-> Successfully generated data/raw/sensor_telemetry.csv")

if __name__ == "__main__":
    generate_telemetry()