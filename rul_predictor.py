"""
Remaining Useful Life (RUL) Predictor.
Implements:
1. Analytical physics-based exponential/power degradation model predicting cycles to 80% EOL.
2. Lightweight Scikit-Learn Random Forest TinyML estimator trained on operational cycling telemetry.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import CellParameters


class RULPredictor:
    """
    Remaining Useful Life (RUL) Predictor for lithium-ion battery cells.
    Predicts the number of operational discharge/charge cycles remaining until
    the cell State of Health drops below the 80.0% End-of-Life (EOL) threshold.
    """

    def __init__(self, params: CellParameters, use_ml_model: bool = True):
        self.params = params
        self.use_ml_model = use_ml_model
        self.eol_threshold = float(params.eol_soh_threshold)  # Default 80.0%
        self.ml_model: Optional[RandomForestRegressor] = None

        if self.use_ml_model:
            self._train_tinyml_surrogate_model()

    def _train_tinyml_surrogate_model(self) -> None:
        """
        Trains a lightweight, low-latency Random Forest regressor (< 25 trees, depth 6)
        representing a TinyML inference edge engine.
        Trained over a diverse distribution of degradation trajectories with variable C-rates,
        SEI growth rates, and ohmic resistance aging.
        """
        rng = np.random.RandomState(42)
        n_cells = 60
        x_train = []
        y_train = []

        for _ in range(n_cells):
            # Sample synthetic degradation variations around the base chemistry
            alpha = rng.uniform(self.params.aging_alpha * 0.85, self.params.aging_alpha * 1.15)
            beta = rng.uniform(self.params.aging_beta * 0.95, self.params.aging_beta * 1.05)
            
            # Theoretical cycle where SOH reaches exactly 80.0%
            # 100 - alpha * eol_cycle^beta = 80  =>  eol_cycle = (20 / alpha)^(1 / beta)
            eol_cycle = (20.0 / alpha) ** (1.0 / beta)

            # Sample 25 points along the cell life trajectory
            sample_cycles = np.linspace(1, eol_cycle * 0.99, 25)
            for cyc in sample_cycles:
                # Capacity SOH with minor measurement noise
                soh = 100.0 - alpha * (cyc ** beta) + rng.normal(0, 0.1)
                soh = max(75.0, min(100.0, soh))

                # Internal resistance R0 growth
                r0_growth = self.params.r0_aging_growth_rate * (cyc ** 0.90)
                r0_est = self.params.r0_nominal * (1.0 + r0_growth) + rng.normal(0, 0.0005)
                delta_r0_pct = ((r0_est - self.params.r0_nominal) / self.params.r0_nominal) * 100.0

                temp_mean = rng.uniform(23.0, 32.0)
                rul_target = max(0.0, eol_cycle - cyc)

                # Feature vector: [cycle_count, soh_pct, r0_est, delta_r0_pct, temp_mean]
                x_train.append([cyc, soh, r0_est, delta_r0_pct, temp_mean])
                y_train.append(rul_target)

        self.ml_model = RandomForestRegressor(
            n_estimators=25,
            max_depth=6,
            max_features="sqrt",
            random_state=42,
            n_jobs=1
        )
        self.ml_model.fit(np.array(x_train), np.array(y_train))

    def predict_exponential(self, cycle_count: int, soh_pct: float) -> float:
        """
        Analytical physics-based degradation extrapolation.
        Solves: SOH(k) = 100 - alpha_hat * k^beta = 80.0%
        """
        if soh_pct <= self.eol_threshold:
            return 0.0

        if cycle_count <= 2:
            # Baseline expected nominal cycles to EOL
            return float(self.params.nominal_cycle_life)

        loss = 100.0 - soh_pct
        if loss <= 0.001:
            return float(max(0, self.params.nominal_cycle_life - cycle_count))

        # Dynamically fit alpha_hat to the current observation
        beta = self.params.aging_beta
        alpha_hat = loss / (float(cycle_count) ** beta)

        # Solve for EOL cycle
        eol_cycle = (20.0 / alpha_hat) ** (1.0 / beta)
        rul = max(0.0, eol_cycle - cycle_count)
        return float(rul)

    def predict_tinyml(
        self,
        cycle_count: int,
        soh_pct: float,
        r0_est: float,
        delta_r0_pct: float,
        temp_mean: float = 25.0
    ) -> float:
        """Lightweight TinyML Random Forest regression prediction."""
        if soh_pct <= self.eol_threshold:
            return 0.0
        if self.ml_model is None:
            return self.predict_exponential(cycle_count, soh_pct)

        features = np.array([[cycle_count, soh_pct, r0_est, delta_r0_pct, temp_mean]])
        rul_pred = self.ml_model.predict(features)[0]
        return float(max(0.0, rul_pred))

    def predict(
        self,
        cycle_count: int,
        soh_pct: float,
        r0_est: Optional[float] = None,
        delta_r0_pct: Optional[float] = None,
        temp_mean: float = 25.0
    ) -> Dict[str, Any]:
        """
        Computes RUL using both analytical exponential degradation and TinyML regression.
        Returns a consensus prediction along with individual model estimates.
        """
        r0 = r0_est if r0_est is not None else self.params.r0_nominal
        dr0 = delta_r0_pct if delta_r0_pct is not None else 0.0

        rul_exp = self.predict_exponential(cycle_count, soh_pct)
        rul_ml = self.predict_tinyml(cycle_count, soh_pct, r0, dr0, temp_mean)

        # Weighted blend for optimal stability and TinyML accuracy
        if cycle_count <= 5:
            consensus_rul = rul_exp
        else:
            consensus_rul = 0.5 * rul_exp + 0.5 * rul_ml

        consensus_rul = max(0.0, consensus_rul)

        return {
            "rul_cycles": int(round(consensus_rul)),
            "rul_exponential": float(round(rul_exp, 1)),
            "rul_tinyml": float(round(rul_ml, 1)),
            "eol_threshold_soh": self.eol_threshold,
        }
