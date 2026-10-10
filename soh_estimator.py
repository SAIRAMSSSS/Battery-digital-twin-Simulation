"""
State of Health (SOH) Estimator.
Implements dual tracking:
1. Capacity fade estimation: SOH_cap = (Q_current / Q_nominal) * 100
2. Online internal resistance (R0) increase tracking via Recursive Least Squares (RLS).
"""

from typing import Dict, Any, Optional
import numpy as np

from config import CellParameters
from ocv_model import OCVModel


class RecursiveLeastSquaresR0:
    """
    Online parameter tracker for ohmic internal resistance R0 using Recursive Least Squares (RLS)
    with exponential forgetting factor.
    System identification model: y = phi * R0 + e
    where:
        phi = I (current in Amperes)
        y = OCV(SOC) - V1 - V_measured (Volts)
    """

    def __init__(
        self,
        nominal_r0: float = 0.025,
        forgetting_factor: float = 0.998,
        initial_covariance: float = 0.5,
        min_current_excitation: float = 0.20,
    ):
        self.r0_est = float(nominal_r0)
        self.nominal_r0 = float(nominal_r0)
        self.lam = float(forgetting_factor)
        self.P = float(initial_covariance)
        self.min_excitation = float(min_current_excitation)

    def update(self, current: float, measured_voltage: float, ocv: float, v1: float) -> float:
        """
        Executes an RLS update step if sufficient current excitation exists.
        Returns the updated R0 estimate.
        """
        # Only update when current magnitude provides adequate Signal-to-Noise Ratio (SNR)
        if abs(current) < self.min_excitation:
            return self.r0_est

        phi = current
        y = ocv - v1 - measured_voltage

        # RLS Gain: K = (P * phi) / (lambda + phi^2 * P)
        denom = self.lam + (phi * phi * self.P)
        K = (self.P * phi) / denom

        # Prediction error
        prediction_error = y - (phi * self.r0_est)

        # Parameter update
        self.r0_est += K * prediction_error

        # Ensure physically realistic bounds for R0: [0.3 * R0_nom, 4.0 * R0_nom]
        self.r0_est = max(0.008, min(0.120, self.r0_est))

        # Covariance update: P = (1 - K * phi) * P / lambda
        self.P = ((1.0 - K * phi) * self.P) / self.lam

        return self.r0_est


class SOHEstimator:
    """
    Comprehensive State of Health (SOH) Estimator.
    Tracks capacity fade, Ampere-hour throughput, and ohmic resistance degradation.
    """

    def __init__(self, params: CellParameters, ocv_model: OCVModel, initial_cycle: int = 0):
        self.params = params
        self.ocv_model = ocv_model
        self.cycle_count = int(initial_cycle)

        # Current capacity initialized based on power-law aging prior
        self.q_nominal = self.params.nominal_capacity_ah
        self.q_current = self._calculate_capacity_from_cycle(self.cycle_count)

        # Online Recursive Least Squares tracker for R0
        initial_r0 = self._calculate_r0_from_cycle(self.cycle_count)
        self.rls_r0 = RecursiveLeastSquaresR0(nominal_r0=initial_r0)

        # Online Ah throughput and dynamic capacity tracking buffers
        self.cumulative_ah_throughput: float = 0.0
        self._anchor_soc: Optional[float] = None
        self._anchor_ah: float = 0.0

    def _calculate_capacity_from_cycle(self, cycle: int) -> float:
        """
        Analytical power-law degradation curve representing SEI layer growth:
        SOH_cap = 100 - alpha * cycle^beta
        Q_current = Q_nominal * (SOH_cap / 100)
        """
        if cycle <= 0:
            return self.q_nominal
        soh_pct = 100.0 - self.params.aging_alpha * (float(cycle) ** self.params.aging_beta)
        soh_pct = max(0.0, min(100.0, soh_pct))
        return self.q_nominal * (soh_pct / 100.0)

    def _calculate_r0_from_cycle(self, cycle: int) -> float:
        """Calculates expected ohmic resistance increase from cycle history."""
        if cycle <= 0:
            return self.params.r0_nominal
        growth = self.params.r0_aging_growth_rate * (float(cycle) ** 0.90)
        return self.params.r0_nominal * (1.0 + growth)

    def update(
        self,
        current: float,
        measured_voltage: float,
        soc_est: float,
        v1_est: float,
        dt: float,
        cycle_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Updates SOH metrics:
        1. Tracks cumulative Ah throughput and updates cycle count if unprovided.
        2. Updates RLS R0 tracker.
        3. Computes SOH_cap, resistance increase (Delta R0 %), and overall SOH.
        """
        # Ah throughput integration: dQ = |I| * dt / 3600
        step_ah = (abs(current) * dt) / 3600.0
        self.cumulative_ah_throughput += step_ah

        # Update cycle count
        if cycle_count is not None:
            self.cycle_count = int(cycle_count)
            self.q_current = self._calculate_capacity_from_cycle(self.cycle_count)
        else:
            # Equivalent Full Cycles (EFC): 1 full cycle = 2 * Q_nominal Ah throughput
            equivalent_cycles = int(self.cumulative_ah_throughput / (2.0 * self.q_nominal))
            if equivalent_cycles > self.cycle_count:
                self.cycle_count = equivalent_cycles
                self.q_current = self._calculate_capacity_from_cycle(self.cycle_count)

        # Update online R0 via Recursive Least Squares
        ocv_val = self.ocv_model.ocv(soc_est)
        r0_tracked = self.rls_r0.update(
            current=current,
            measured_voltage=measured_voltage,
            ocv=ocv_val,
            v1=v1_est
        )

        # Calculate SOH_capacity: (Q_current / Q_nominal) * 100
        soh_cap_pct = (self.q_current / self.q_nominal) * 100.0
        soh_cap_pct = float(np.clip(soh_cap_pct, 0.0, 100.0))

        # Calculate Internal Resistance Increase tracking
        r0_nom = self.params.r0_nominal
        delta_r0_pct = ((r0_tracked - r0_nom) / r0_nom) * 100.0
        soh_resistance_pct = max(0.0, 100.0 - delta_r0_pct)

        return {
            "soh_pct": soh_cap_pct,
            "soh_cap": soh_cap_pct,
            "soh_resistance": soh_resistance_pct,
            "q_current_ah": self.q_current,
            "q_nominal_ah": self.q_nominal,
            "r0_estimated_ohm": r0_tracked,
            "r0_nominal_ohm": r0_nom,
            "delta_r0_pct": delta_r0_pct,
            "cycle_count": self.cycle_count,
            "cumulative_ah": self.cumulative_ah_throughput,
        }
