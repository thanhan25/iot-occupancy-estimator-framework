
# 🏢 IoT Occupancy Estimator & Sensor Fusion Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

## 📌 Executive Summary

An enterprise-grade, time-series machine learning framework designed to estimate real-time room occupancy in office buildings using multi-modal IoT sensor telemetry. By fusing asynchronous data streams (PIR motion, $CO_2$ concentration, luminance, and temperature), this framework provides a highly accurate, non-intrusive alternative to direct camera surveillance, directly supporting the intelligent operation of smart HVAC and shading systems.

## ⚙️ System Architecture

The framework is built with strict adherence to reproducibility and scalable data engineering principles:

1. **Telemetry Simulation & Ingestion:** Generates realistic, physically-constrained time-series data mirroring standard office IoT arrays (5-10 devices per room).
2. **Sensor Fusion & Feature Engineering:** Processes raw streams to calculate rolling window averages, delayed response lags, and rates-of-change (RoC) to account for cumulative physical effects (e.g., $CO_2$ dissipation latency).
3. **Estimation Modeling:** Utilizes a Random Forest classification architecture to learn complex, non-linear relationships between environmental noise and actual human occupancy.

## 🚀 Quickstart & Reproducibility

**1. Clone the repository and install dependencies:**

```bash
git clone [https://github.com/thanhan25/iot-occupancy-estimator-framework.git](https://github.com/thanhan25/iot-occupancy-estimator-framework.git)
cd iot-occupancy-estimator-framework
pip install -r requirements.txt
```

**2. Generate Mock Telemetry Data:**
Because physical sensor data is proprietary, this module generates a realistic, 7-day minute-by-minute testing array.

```bash
python generate_mock_data.py
```

**3. Execute the Full Pipeline:**
This will trigger the ETL process, engineer time-series features, evaluate the model, and save the binary weights.

```bash
python -m src.pipeline
```

## 📊 Analytical Methodology
- Asynchronous Integration: Handles the reality that PIR sensors trigger asynchronously compared to continuous-polling PQI meters.
- Algorithm Evaluation: Model hyper-parameters and target variables are completely decoupled via config.yaml to allow for rapid experimental design and A/B testing of different estimation approaches.

## 👤 Author
An Vo
M.Sc. Quantitative Economics Candidate | University of Bonn

