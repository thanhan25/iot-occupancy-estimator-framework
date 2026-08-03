import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def plot_sensor_fusion(df: pd.DataFrame, output_dir: str = 'assets'):
    """Generates a publication-grade visualization of sensor metrics vs. occupancy."""
    logger.info("Generating sensor fusion visual analytics...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a clean, scientific aesthetic
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot Continuous Metric (CO2)
    co2_color = '#d62728' # Professional muted red
    ax1.set_xlabel('Time (Asynchronous Sensor Stream)', fontweight='bold')
    ax1.set_ylabel('CO2 Concentration (ppm)', color=co2_color, fontweight='bold')
    ax1.plot(df.index, df['co2_ppm'], color=co2_color, alpha=0.85, linewidth=1.5, label='CO2 (ppm)')
    ax1.tick_params(axis='y', labelcolor=co2_color)
    
    # Overlay Ground Truth Occupancy as shaded regions
    if 'occupancy' in df.columns:
        ax1.fill_between(df.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                         where=(df['occupancy'] == 1), 
                         color='#7f7f7f', alpha=0.2, label='Room Occupied')
    
    plt.title('LLEC Simulated Environment: Sensor Telemetry vs. Actual Occupancy', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    save_path = os.path.join(output_dir, 'sensor_fusion_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visual analytics successfully saved to {save_path}")
    plt.close()