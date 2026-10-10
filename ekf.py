"""
Extended Kalman Filter (EKF) State of Charge (SOC) Estimator.
Combines a 1st-order Thevenin Equivalent Circuit Model (ECM) with Coulomb Counting
and nonlinear OCV linearization using the exact Jacobian matrix.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from config import CellParameters
from ocv_model import OCVModel


class ExtendedKalmanFilterSOC:
    """
    Dual-state Extended Kalman Filter for real-time lithium-ion battery SOC estimation.
    State vector: x = [SOC, V1]^T
    - x[0]: State of Charge (dimensionless, 0.0 - 1.0)
    - x[1]: Polarization voltage across R1 || C1 pair (Volts)
    """

    def __init__(
        self,
        params: CellParameters,
        ocv_model: OCVModel,
        initial_soc: float = 1.0,
        initial_v1: float = 0.0,
        soc_process_noise: float = 1e-7,
        v1_process_noise: float = 1e-6,
        voltage_meas_noise: float = 2.5e-5,  # (5 mV)^2 = 2.5e-5 V^2
    ):
        self.params = params
        self.ocv_model = ocv_model

        # State vector: [SOC, V1]
        self.x = np.array([float(initial_soc), float(initial_v1)], dtype=np.float64)

        # State Error Covariance Matrix P
        self.P = np.diag([0.04, 1e-4])  # Initial uncertainty: +/-20% SOC, +/-10mV V1

        # Process Noise Covariance Matrix Q
        self.Q = np.diag([soc_process_noise, v1_process_noise])

        # Measurement Noise Covariance R (Scalar variance)
        self.R = float(voltage_meas_noise)

        # Pure Coulomb Counting accumulator for benchmark comparison
        self.soc_coulomb_counting = float(initial_soc)

        # Diagnostic telemetry
        self.last_innovation: float = 0.0
        self.last_kalman_gain: np.ndarray = np.zeros(2)
        self.last_v_model: float = 0.0

    def predict(self, current: float, dt: float, current_capacity_ah: Optional[float] = None) -> np.ndarray:
        """
        Time Update (Prediction Step).
        Discretizes 1st-order Thevenin differential equations using zero-order hold.
        current > 0 for discharge, current < 0 for charge.
        """
        capacity_ah = current_capacity_ah if current_capacity_ah is not None else self.params.nominal_capacity_ah
        capacity_coulombs = capacity_ah * 3600.0

        # Coulombic efficiency (eta): < 1.0 during charging, 1.0 during discharge
        eta = self.params.coulombic_efficiency if current < 0 else 1.0

        tau1 = self.params.tau1
        exp_decay = np.exp(-dt / tau1)
        r1 = self.params.r1_nominal

        # State transition matrix A and input matrix B
        # x_k = A * x_{k-1} + B * I_{k-1}
        A = np.array([
            [1.0, 0.0],
            [0.0, exp_decay]
        ], dtype=np.float64)

        B = np.array([
            -(eta * dt) / capacity_coulombs,
            r1 * (1.0 - exp_decay)
        ], dtype=np.float64)

        # State propagation
        self.x = A @ self.x + B * current

        # Keep predicted SOC bounded
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))

        # Covariance propagation: P = A * P * A^T + Q
        self.P = A @ self.P @ A.T + self.Q

        # Pure Coulomb Counting update
        self.soc_coulomb_counting -= (eta * current * dt) / capacity_coulombs
        self.soc_coulomb_counting = float(np.clip(self.soc_coulomb_counting, 0.0, 1.0))

        return self.x

    def update(
        self,
        measured_voltage: float,
        current: float,
        temperature_c: float = 25.0,
        r0_override: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Measurement Update (Correction Step).
        Linearizes the terminal voltage measurement equation around the predicted state:
        V_t = OCV(SOC) - V1 - I * R0
        Returns:
            estimated_soc: float in [0.0, 1.0]
            estimated_v1: float in Volts
            model_terminal_voltage: float in Volts
        """
        soc_pred = self.x[0]
        v1_pred = self.x[1]

        # Determine effective R0 (Arrhenius temperature compensated or RLS estimated)
        if r0_override is not None:
            r0 = float(r0_override)
        else:
            r0 = self.params.get_temperature_adjusted_r0(temperature_c)

        # Model predicted terminal voltage: V_pred = OCV(z) - V1 - I * R0
        ocv_pred = self.ocv_model.ocv(soc_pred)
        v_model_pred = ocv_pred - v1_pred - (current * r0)
        self.last_v_model = float(v_model_pred)

        # Innovation residual
        innovation = measured_voltage - v_model_pred
        self.last_innovation = float(innovation)

        # Jacobian Observation Vector C = [dOCV/dSOC, -1.0]
        docv_dz = self.ocv_model.docv_dsoc(soc_pred)
        C = np.array([[docv_dz, -1.0]], dtype=np.float64)  # Shape (1, 2)

        # Innovation Covariance S = C * P * C^T + R
        S = float((C @ self.P @ C.T)[0, 0] + self.R)

        # Optimal Kalman Gain: K = P * C^T * S^(-1)  (Shape: (2, 1))
        K = (self.P @ C.T) / S
        self.last_kalman_gain = K.flatten()

        # State Correction: x = x + K * innovation
        self.x = self.x + (K.flatten() * innovation)

        # Physical constraint clamping: SOC in [0.0, 1.0]
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))

        # Joseph Form Covariance Update for guaranteed positive semi-definiteness:
        # P = (I - K*C) * P * (I - K*C)^T + K * R * K^T
        I2 = np.eye(2, dtype=np.float64)
        IKC = I2 - (K @ C)
        self.P = IKC @ self.P @ IKC.T + (self.R * (K @ K.T))

        # Symmetrize to prevent numerical asymmetry drift
        self.P = 0.5 * (self.P + self.P.T)

        return float(self.x[0]), float(self.x[1]), float(v_model_pred)

    def step(
        self,
        measured_voltage: float,
        current: float,
        dt: float,
        temperature_c: float = 25.0,
        r0_override: Optional[float] = None,
        current_capacity_ah: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes a complete EKF cycle (Predict + Update) in one step.
        """
        self.predict(current, dt, current_capacity_ah=current_capacity_ah)
        soc_est, v1_est, v_pred = self.update(
            measured_voltage, current, temperature_c=temperature_c, r0_override=r0_override
        )
        return {
            "soc": soc_est,
            "soc_pct": soc_est * 100.0,
            "v1": v1_est,
            "v_model": v_pred,
            "innovation": self.last_innovation,
            "coulomb_counting_soc": self.soc_coulomb_counting,
            "soc_error_variance": float(self.P[0, 0]),
        }
