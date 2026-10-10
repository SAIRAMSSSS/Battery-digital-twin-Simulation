# Lithium-Ion Battery Digital Twin Simulation Framework

A modular, production-grade Python simulation framework implementing a physics-informed **Digital Twin** for lithium-ion battery cells (NMC and LFP chemistries). Designed from the perspective of a Senior Battery Systems Engineer and TinyML Edge Researcher.

---

## 🌟 Key Features

1. **State of Charge (SOC) Estimator**:
   - Discrete-time **Extended Kalman Filter (EKF)** built on a 1st-order **Thevenin Equivalent Circuit Model (ECM)**.
   - Strictly monotonic Open Circuit Voltage curve $OCV(SOC)$ using shape-preserving cubic Hermite splines (PCHIP).
   - Exact analytical Jacobian observation vector $C_k = \left[\frac{\partial OCV}{\partial z}, -1\right]$.
   - Numerically robust **Joseph-form covariance update** guaranteeing positive semi-definiteness ($P \succeq 0$).
   - Deadbeat convergence rejecting initial prior offsets ($\pm 20\%$) in under 20 seconds.

2. **State of Health (SOH) Estimator**:
   - Power-law / SEI layer capacity fade tracking: $SOH_{cap} = \frac{Q_{current}}{Q_{nominal}} \times 100\%$.
   - Online **Recursive Least Squares (RLS)** tracker with exponential forgetting factor ($\lambda = 0.998$) identifying ohmic resistance ($R_0$) in real time from noisy dynamic load steps.
   - Temperature compensation via Arrhenius kinetics: $R_0(T) = R_{0, ref} \exp\left(\frac{E_a}{R} \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)$.

3. **Remaining Useful Life (RUL) Predictor**:
   - Dual-engine prediction combining:
     - **Analytical Degradation Extrapolator**: Closed-form solver projecting cycle count until $SOH \le 80.0\%$ (End of Life threshold).
     - **TinyML Edge Model**: Lightweight Scikit-Learn Random Forest regressor trained on electro-thermal degradation signatures with sub-millisecond inference latency.

4. **Safety Supervision & Telemetry**:
   - Evaluates operational envelope according to ISO 26262 functional safety and UL 1973 standards.
   - Generates structured JSON-compatible telemetry per step:
     ```json
     {
       "timestamp": 100.0,
       "voltage": 4.0694,
       "current": 1.4369,
       "temp": 25.27,
       "soc": 89.738,
       "soh": 92.086,
       "rul_cycles": 845,
       "status": "NORMAL"
     }
     ```

5. **Dynamic 1,000-Step Simulation & Publication Visualization**:
   - Ground-truth physics generator simulating dynamic stress cycles (DST / WLTP profile): acceleration pulses, regenerative braking, cruising, and open-circuit diffusion relaxation.
   - Gaussian sensor measurement noise: $\sigma_V = 5\text{ mV}$, $\sigma_I = 25\text{ mA}$, $\sigma_T = 0.20^\circ\text{C}$.
   - High-resolution 4-subplot engineering figure: Terminal Voltage, Estimated SOC vs True SOC, SOH Trajectory, and RUL Decay Curve.

---

## 📐 Mathematical Formulation

### 1. Equivalent Circuit Model (ECM)
The terminal voltage $V_t$ is modeled as:
$$V_t(t) = OCV(z(t)) - V_1(t) - I(t) R_0(T)$$

where:
- $z(t)$ is State of Charge ($SOC \in [0, 1]$),
- $V_1(t)$ is diffusion polarization voltage across $R_1 \parallel C_1$,
- $I(t)$ is current ($I > 0$ discharge, $I < 0$ charge),
- $R_0(T)$ is temperature-dependent ohmic internal resistance.

Continuous-time state dynamics:
$$\dot{z}(t) = -\frac{\eta(I) \cdot I(t)}{Q_{cap} \times 3600}$$
$$\dot{V}_1(t) = -\frac{V_1(t)}{R_1 C_1} + \frac{I(t)}{C_1}$$

Zero-Order-Hold (ZOH) discrete state space:
$$x_k = \begin{bmatrix} z_k \\ V_{1,k} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & \exp(-\Delta t / \tau_1) \end{bmatrix} x_{k-1} + \begin{bmatrix} -\frac{\eta \Delta t}{Q_{coulombs}} \\ R_1(1 - \exp(-\Delta t / \tau_1)) \end{bmatrix} I_{k-1}$$

### 2. Extended Kalman Filter (EKF) Updates
- **Prediction**:
  $$\hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B I_{k-1}$$
  $$P_{k|k-1} = A P_{k-1|k-1} A^T + Q$$

- **Correction**:
  $$\tilde{y}_k = V_{meas, k} - \left( OCV(\hat{z}_{k|k-1}) - \hat{V}_{1, k|k-1} - I_k R_0 \right)$$
  $$C_k = \left[ \left.\frac{dOCV}{dz}\right|_{\hat{z}_{k|k-1}}, -1.0 \right]$$
  $$S_k = C_k P_{k|k-1} C_k^T + R$$
  $$K_k = P_{k|k-1} C_k^T S_k^{-1}$$
  $$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k$$
  $$P_{k|k} = (I - K_k C_k) P_{k|k-1} (I - K_k C_k)^T + K_k R K_k^T \quad \text{(Joseph Form)}$$

### 3. Online Recursive Least Squares (RLS) for $R_0$
$$y_k = OCV(z_k) - V_{1,k} - V_{meas, k} \approx \phi_k \cdot R_0$$
where $\phi_k = I_k$. For $|I_k| \ge 0.2\text{ A}$:
$$K_{rls} = \frac{P_{rls} \phi_k}{\lambda + \phi_k^2 P_{rls}}, \quad \hat{R}_{0, k} = \hat{R}_{0, k-1} + K_{rls}(y_k - \phi_k \hat{R}_{0, k-1})$$
$$P_{rls} \leftarrow \frac{1}{\lambda} (1 - K_{rls} \phi_k) P_{rls}$$

---

## 🚀 Quickstart & Usage

### 1. Run the Full 1,000-Step Synthetic Benchmark
```bash
python battery_digital_twin.py --mode synthetic --steps 1000 --chemistry NMC
```

Output:
- Generates `synthetic_battery_timeseries.csv` (1,000 rows of noisy sensor data).
- Generates `digital_twin_telemetry.json` (Structured JSON outputs for all steps).
- Generates `battery_digital_twin_simulation.png` (4-subplot engineering visualization).

### 2. Ingest Custom CSV Telemetry
```bash
python battery_digital_twin.py --mode csv --csv path/to/your_battery_data.csv
```
Input CSV must contain: `timestamp`, `voltage`, `current`, `temperature`, `cycle_count`.

### 3. Python API Integration
```python
from battery_digital_twin import BatteryDigitalTwin
from config import BatteryChemistry, get_default_config

# Initialize Twin
config = get_default_config(BatteryChemistry.NMC)
twin = BatteryDigitalTwin(params=config, initial_soc=0.90, initial_cycle_count=350)

# Real-time streaming update
telemetry = twin.update(
    voltage=3.952,
    current=2.15,
    temperature=26.4,
    dt=1.0
)

print(telemetry)
# Output:
# {'timestamp': 1.0, 'voltage': 3.952, 'current': 2.15, 'temp': 26.4,
#  'soc': 89.81, 'soh': 92.09, 'rul_cycles': 845, 'status': 'NORMAL'}
```

### 4. Run Unit Tests
```bash
python -m unittest test_battery_digital_twin.py
```

---

## 📊 File Structure
- `config.py`: Electrochemical, ECM, thermal, and degradation parameters for NMC/LFP.
- `ocv_model.py`: Shape-preserving cubic Hermite spline OCV curve and analytical Jacobian derivative.
- `ekf.py`: Dual-state Extended Kalman Filter with Joseph covariance formulation.
- `soh_estimator.py`: Capacity fade model and online RLS ohmic resistance ($R_0$) tracker.
- `rul_predictor.py`: Analytical degradation extrapolator and TinyML Random Forest regressor.
- `battery_digital_twin.py`: Master orchestrator, simulator, data pipelines, CLI, and visualization harness.
- `test_battery_digital_twin.py`: Automated test suite verifying convergence, stability, and safety.
