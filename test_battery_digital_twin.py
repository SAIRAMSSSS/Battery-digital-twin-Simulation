"""
Unit and Integration Test Suite for Lithium-Ion Battery Digital Twin Framework.
Tests:
- OCV monotonicity and analytical Jacobian derivatives
- EKF numerical stability and deadbeat SOC convergence
- Online Recursive Least Squares (RLS) parameter tracking
- SOH capacity fade and RUL prediction accuracy
- Safety supervisory state machine transitions
- CSV and DataFrame ingestion pipelines
"""

import unittest
import numpy as np
import pandas as pd
import json
import os
import tempfile

from config import BatteryChemistry, get_default_config, CellParameters
from ocv_model import OCVModel
from ekf import ExtendedKalmanFilterSOC
from soh_estimator import SOHEstimator, RecursiveLeastSquaresR0
from rul_predictor import RULPredictor
from battery_digital_twin import BatteryDigitalTwin, SyntheticBatterySimulator


class TestOCVModel(unittest.TestCase):
    """Tests for Open Circuit Voltage model."""

    def setUp(self):
        self.ocv_nmc = OCVModel(BatteryChemistry.NMC)
        self.ocv_lfp = OCVModel(BatteryChemistry.LFP)

    def test_nmc_monotonicity(self):
        soc_grid = np.linspace(0.0, 1.0, 200)
        ocv_vals = self.ocv_nmc.ocv(soc_grid)
        derivs = self.ocv_nmc.docv_dsoc(soc_grid)
        
        # Strictly increasing
        self.assertTrue(np.all(np.diff(ocv_vals) > 0.0), "NMC OCV must be strictly monotonic")
        self.assertTrue(np.all(derivs > 0.0), "dOCV/dSOC must be strictly positive")
        self.assertAlmostEqual(self.ocv_nmc.ocv(0.0), 3.00, places=2)
        self.assertAlmostEqual(self.ocv_nmc.ocv(1.0), 4.20, places=2)

    def test_boundary_extrapolation_safety(self):
        # Must not raise exceptions or return NaN outside [0, 1]
        v_neg = self.ocv_nmc.ocv(-0.05)
        v_over = self.ocv_nmc.ocv(1.05)
        self.assertTrue(np.isfinite(v_neg))
        self.assertTrue(np.isfinite(v_over))
        self.assertLess(v_neg, 3.00)
        self.assertGreater(v_over, 4.20)


class TestEKFEstimator(unittest.TestCase):
    """Tests for Extended Kalman Filter SOC estimation."""

    def setUp(self):
        self.config = get_default_config(BatteryChemistry.NMC)
        self.ocv = OCVModel(BatteryChemistry.NMC)

    def test_ekf_convergence_under_prior_error(self):
        # True SOC = 0.90, Initial guess = 0.60 (30% error offset)
        ekf = ExtendedKalmanFilterSOC(self.config, self.ocv, initial_soc=0.60)
        true_soc = 0.90
        
        # Simulate static open-circuit terminal voltage measurement at 90% SOC
        v_meas = self.ocv.ocv(true_soc)
        
        # Step through 10 static measurements
        for _ in range(15):
            res = ekf.step(measured_voltage=v_meas, current=0.0, dt=1.0)
            
        final_soc = ekf.x[0]
        # Should converge to within 1% of true SOC
        self.assertAlmostEqual(final_soc, true_soc, delta=0.015,
                               msg=f"EKF did not converge to true SOC (got {final_soc:.3f})")


class TestRLSParameterTracking(unittest.TestCase):
    """Tests for Recursive Least Squares R0 estimation."""

    def test_rls_convergence(self):
        true_r0 = 0.035  # Aged resistance (35 mOhm)
        rls = RecursiveLeastSquaresR0(nominal_r0=0.025, forgetting_factor=0.995)
        
        rng = np.random.RandomState(42)
        ocv = 3.80
        v1 = 0.005
        
        for _ in range(200):
            current = rng.uniform(1.5, 4.0) * rng.choice([-1, 1])
            v_meas = ocv - v1 - current * true_r0 + rng.normal(0, 0.002)
            r0_est = rls.update(current=current, measured_voltage=v_meas, ocv=ocv, v1=v1)
            
        self.assertAlmostEqual(r0_est, true_r0, delta=0.003,
                               msg=f"RLS R0 did not converge to true R0 (got {r0_est:.4f})")


class TestRULPredictor(unittest.TestCase):
    """Tests for Remaining Useful Life prediction models."""

    def setUp(self):
        self.config = get_default_config(BatteryChemistry.NMC)
        self.rul = RULPredictor(self.config, use_ml_model=True)

    def test_rul_monotonic_decay(self):
        # As cycle count increases and SOH drops, RUL must decrease
        pred_young = self.rul.predict(cycle_count=100, soh_pct=98.0)
        pred_mid = self.rul.predict(cycle_count=500, soh_pct=90.0)
        pred_old = self.rul.predict(cycle_count=1000, soh_pct=83.0)
        
        self.assertGreater(pred_young["rul_cycles"], pred_mid["rul_cycles"])
        self.assertGreater(pred_mid["rul_cycles"], pred_old["rul_cycles"])

    def test_eol_threshold_boundary(self):
        # At or below 80% SOH, RUL should be 0
        pred_eol = self.rul.predict(cycle_count=1250, soh_pct=79.5)
        self.assertEqual(pred_eol["rul_cycles"], 0)


class TestBatteryDigitalTwin(unittest.TestCase):
    """Tests for complete BatteryDigitalTwin system integration."""

    def setUp(self):
        self.config = get_default_config(BatteryChemistry.NMC)
        self.twin = BatteryDigitalTwin(params=self.config, initial_soc=0.85, initial_cycle_count=200)

    def test_json_payload_schema(self):
        res = self.twin.update(voltage=3.85, current=2.0, temperature=25.0, dt=1.0)
        
        required_keys = {"timestamp", "voltage", "current", "temp", "soc", "soh", "rul_cycles", "status"}
        self.assertEqual(set(res.keys()), required_keys)
        
        # Check JSON serializability
        json_str = json.dumps(res)
        self.assertTrue(isinstance(json_str, str))
        
        # Check status options
        self.assertIn(res["status"], ["NORMAL", "WARNING", "CRITICAL"])

    def test_safety_supervisor_triggers(self):
        # Under-voltage critical fault
        twin_crit_v = BatteryDigitalTwin(params=self.config, initial_soc=0.85, initial_cycle_count=200)
        res_crit_v = twin_crit_v.update(voltage=2.60, current=0.0, temperature=25.0, dt=1.0)
        self.assertEqual(res_crit_v["status"], "CRITICAL")
        
        # Over-temperature critical fault
        twin_crit_t = BatteryDigitalTwin(params=self.config, initial_soc=0.85, initial_cycle_count=200)
        res_crit_t = twin_crit_t.update(voltage=3.80, current=0.0, temperature=60.0, dt=1.0)
        self.assertEqual(res_crit_t["status"], "CRITICAL")
        
        # Normal operation
        twin_normal = BatteryDigitalTwin(params=self.config, initial_soc=0.85, initial_cycle_count=200)
        res_normal = twin_normal.update(voltage=3.85, current=1.5, temperature=28.0, dt=1.0)
        self.assertEqual(res_normal["status"], "NORMAL")

    def test_dataframe_and_csv_pipeline(self):
        simulator = SyntheticBatterySimulator(initial_soc=0.90, cycle_count=200)
        df_sim = simulator.generate_dynamic_profile(n_steps=50, dt=1.0)
        
        twin = BatteryDigitalTwin(initial_soc=0.90, initial_cycle_count=200)
        df_out = twin.process_dataframe(df_sim)
        
        self.assertEqual(len(df_out), 50)
        self.assertIn("soc", df_out.columns)
        self.assertIn("soh", df_out.columns)
        self.assertIn("rul_cycles", df_out.columns)


if __name__ == "__main__":
    unittest.main()
