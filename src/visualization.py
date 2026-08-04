import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def plot_sensor_fusion(df: pd.DataFrame, output_dir: str = 'assets'):
    """Generates a publication-grade visualization of dual-axis sensor metrics vs. occupancy."""
    logger.info("Generating sensor fusion visual analytics...")
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Left Axis: CO2 Concentration
    co2_color = '#d62728'
    ax1.set_xlabel('Time (Asynchronous Sensor Stream)', fontweight='bold')
    ax1.set_ylabel('CO2 Concentration (ppm)', color=co2_color, fontweight='bold')
    ax1.plot(df.index, df['co2_ppm'], color=co2_color, alpha=0.9, linewidth=2, label='CO2 (ppm)')
    ax1.tick_params(axis='y', labelcolor=co2_color)
    ax1.set_ylim(350, 1300)
    
    # Right Axis: PIR Motion Triggers
    ax2 = ax1.twinx()
    pir_color = '#1f77b4'
    ax2.set_ylabel('PIR Motion Detected', color=pir_color, fontweight='bold')
    ax2.scatter(df.index[df['pir_motion'] == 1], df['pir_motion'][df['pir_motion'] == 1], 
                color=pir_color, alpha=0.6, marker='|', s=100, label='PIR Trigger')
    ax2.set_yticks([]) # Hide right y-ticks for cleaner look
    
    # Ground Truth Occupancy Shading
    if 'occupancy' in df.columns:
        ax1.fill_between(df.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                         where=(df['occupancy'] == 1), 
                         color='#7f7f7f', alpha=0.2, label='True Occupancy')
    
    plt.title('LLEC Simulated Environment: Sensor Telemetry vs. Actual Occupancy', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    save_path = os.path.join(output_dir, 'sensor_fusion_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visual analytics successfully saved to {save_path}")
    plt.close()