"""
Open Circuit Voltage (OCV) Modeling and Electrochemical Characterization.
Provides high-fidelity, strictly monotonic OCV-SOC mapping and analytical derivatives
(dOCV/dSOC) essential for Extended Kalman Filter (EKF) linearization.
"""

from typing import Union
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from config import BatteryChemistry


class OCVModel:
    """
    High-fidelity Open Circuit Voltage (OCV) model.
    Guarantees strict monotonicity (dOCV/dSOC > 0) to prevent unphysical filter divergent states.
    """
    
    # Calibrated C/20 Galvanostatic Discharge Empirical Curves at 25°C
    # Reference: NMC 18650 (LiNiMnCoO2) High-Power / Energy Cell
    _NMC_SOC_POINTS = np.array([
        0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00
    ], dtype=np.float64)
    
    _NMC_OCV_POINTS = np.array([
        3.000, 3.190, 3.360, 3.515, 3.618, 3.675, 3.722,
        3.774, 3.835, 3.918, 4.020, 4.120, 4.168, 4.188, 4.200
    ], dtype=np.float64)
    
    # Reference: LFP 26650 (LiFePO4) Flat Plateau Cell
    _LFP_SOC_POINTS = np.array([
        0.00, 0.03, 0.08, 0.15, 0.30, 0.50, 0.70,
        0.85, 0.92, 0.96, 0.99, 1.00
    ], dtype=np.float64)
    
    _LFP_OCV_POINTS = np.array([
        2.500, 3.050, 3.200, 3.265, 3.290, 3.310, 3.328,
        3.342, 3.365, 3.420, 3.550, 3.650
    ], dtype=np.float64)

    def __init__(self, chemistry: BatteryChemistry = BatteryChemistry.NMC):
        self.chemistry = chemistry
        if chemistry == BatteryChemistry.LFP:
            self.soc_knots = self._LFP_SOC_POINTS
            self.ocv_knots = self._LFP_OCV_POINTS
        else:
            self.soc_knots = self._NMC_SOC_POINTS
            self.ocv_knots = self._NMC_OCV_POINTS
            
        if HAS_SCIPY:
            self._interpolator = PchipInterpolator(self.soc_knots, self.ocv_knots)
            self._deriv_interpolator = self._interpolator.derivative()
        else:
            self._interpolator = None
            self._deriv_interpolator = None
            
        # Slope boundaries for linear extrapolation outside [0, 1]
        self.slope_low = (self.ocv_knots[1] - self.ocv_knots[0]) / (self.soc_knots[1] - self.soc_knots[0])
        self.slope_high = (self.ocv_knots[-1] - self.ocv_knots[-2]) / (self.soc_knots[-1] - self.soc_knots[-2])

    def ocv(self, soc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes Open Circuit Voltage (V) as a function of State of Charge (SOC in [0, 1]).
        Supports both scalar and NumPy vector evaluation.
        """
        is_scalar = np.isscalar(soc)
        soc_arr = np.asarray(soc, dtype=np.float64)
        
        if HAS_SCIPY:
            # PCHIP evaluation inside [0, 1] with smooth continuous extrapolation
            val = self._interpolator(soc_arr)
        else:
            # Pure NumPy linear fallback if SciPy not present
            val = np.interp(soc_arr, self.soc_knots, self.ocv_knots)
            
        # Linear slope handling outside [0, 1] for filter stability
        if np.any(soc_arr < 0.0):
            mask = soc_arr < 0.0
            val[mask] = self.ocv_knots[0] + self.slope_low * soc_arr[mask]
        if np.any(soc_arr > 1.0):
            mask = soc_arr > 1.0
            val[mask] = self.ocv_knots[-1] + self.slope_high * (soc_arr[mask] - 1.0)
            
        return float(val) if is_scalar else val

    def docv_dsoc(self, soc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes analytical derivative dOCV/dSOC (V/unit_SOC).
        Used directly in the Extended Kalman Filter observation Jacobian matrix C_k.
        """
        is_scalar = np.isscalar(soc)
        soc_arr = np.asarray(soc, dtype=np.float64)
        
        if HAS_SCIPY:
            deriv = self._deriv_interpolator(soc_arr)
        else:
            # Numerical gradient / slope approximation fallback
            deriv = np.gradient(np.interp(soc_arr, self.soc_knots, self.ocv_knots), soc_arr)
            
        # Handle boundaries
        if np.any(soc_arr < 0.0):
            mask = soc_arr < 0.0
            deriv[mask] = self.slope_low
        if np.any(soc_arr > 1.0):
            mask = soc_arr > 1.0
            deriv[mask] = self.slope_high
            
        # Ensure non-zero positive derivative strictly > 0.01 for EKF observability
        deriv = np.maximum(deriv, 0.05)
        
        return float(deriv) if is_scalar else deriv
