"""
Configuration and Parameters for Lithium-Ion Battery Digital Twin.
Defines electrochemical, equivalent circuit model (ECM), thermal, and degradation parameters
for standard lithium-ion cell chemistries (NMC, LFP).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class BatteryChemistry(str, Enum):
    NMC = "NMC"  # Lithium Nickel Manganese Cobalt Oxide (3.0V - 4.2V)
    LFP = "LFP"  # Lithium Iron Phosphate (2.5V - 3.65V)


@dataclass
class CellParameters:
    """Electrochemical, Equivalent Circuit, and Thermal Parameters."""
    chemistry: BatteryChemistry = BatteryChemistry.NMC
    
    # Capacity & Voltage Specs
    nominal_capacity_ah: float = 2.50      # Nominal capacity (Ah), e.g., 18650 cell
    nominal_voltage: float = 3.70          # Nominal cell voltage (V)
    v_max: float = 4.20                    # Maximum upper cut-off voltage (V)
    v_min: float = 3.00                    # Minimum lower cut-off voltage (V)
    v_warning_high: float = 4.22           # Upper warning threshold (V)
    v_warning_low: float = 3.10            # Lower warning threshold (V)
    
    # 1st-Order Thevenin Equivalent Circuit Model (ECM) at 25°C
    r0_nominal: float = 0.025              # Ohmic internal resistance (Ohms)
    r1_nominal: float = 0.015              # Charge transfer / polarization resistance (Ohms)
    c1_nominal: float = 2500.0             # Polarization capacitance (Farads)
    coulombic_efficiency: float = 0.998    # Coulombic efficiency for charging (<= 1.0)
    
    # Thermal Model Parameters (Lumped single-node thermal mass)
    thermal_mass_cp: float = 45.0          # Cell heat capacity C_th (J/K)
    heat_transfer_ha: float = 0.18         # Convective cooling coefficient h*A (W/K)
    ambient_temp_c: float = 25.0           # Ambient temperature (°C)
    t_warning_c: float = 45.0              # Temperature warning threshold (°C)
    t_critical_c: float = 55.0             # Critical temperature threshold (°C)
    arrhenius_ea_r0: float = 20000.0       # Activation energy for R0 Arrhenius equation (J/mol)
    r_gas: float = 8.314462618             # Universal gas constant (J/(mol*K))
    t_ref_kelvin: float = 298.15           # Reference temperature 25°C in Kelvin
    
    # Degradation & Aging Parameters (Lifetime)
    eol_soh_threshold: float = 80.0        # End-of-Life SOH threshold (%)
    nominal_cycle_life: int = 1200         # Cycles to 80% EOL under standard 1C cycling
    aging_alpha: float = 0.0978            # Capacity fade coefficient SOH = 100 - alpha * cycle^beta
    aging_beta: float = 0.75               # Sub-linear power-law aging exponent (diffusion SEI growth)
    r0_aging_growth_rate: float = 0.0006   # Rate of ohmic resistance increase per cycle
    
    @property
    def nominal_capacity_coulombs(self) -> float:
        """Returns nominal capacity in Coulombs (Ampere-seconds)."""
        return self.nominal_capacity_ah * 3600.0
    
    @property
    def tau1(self) -> float:
        """Returns ECM polarization time constant tau = R1 * C1 (seconds)."""
        return self.r1_nominal * self.c1_nominal
    
    def get_temperature_adjusted_r0(self, temp_c: float) -> float:
        """
        Calculates temperature-dependent ohmic resistance R0 using Arrhenius relation.
        R0(T) = R0_ref * exp((Ea / R) * (1/T_K - 1/T_ref))
        """
        temp_k = temp_c + 273.15
        if temp_k <= 0:
            temp_k = 273.15
        ratio = (1.0 / temp_k) - (1.0 / self.t_ref_kelvin)
        factor = float(self.arrhenius_ea_r0 / self.r_gas * ratio)
        # Cap exponential to prevent numerical overflow in extreme conditions
        factor = max(-3.0, min(3.0, factor))
        import math
        return self.r0_nominal * math.exp(factor)


def get_default_config(chemistry: BatteryChemistry = BatteryChemistry.NMC) -> CellParameters:
    """Factory function returning standard configuration for chosen cell chemistry."""
    if chemistry == BatteryChemistry.LFP:
        return CellParameters(
            chemistry=BatteryChemistry.LFP,
            nominal_capacity_ah=2.50,
            nominal_voltage=3.20,
            v_max=3.65,
            v_min=2.50,
            v_warning_high=3.62,
            v_warning_low=2.65,
            r0_nominal=0.030,
            r1_nominal=0.020,
            c1_nominal=3000.0,
            nominal_cycle_life=2500,
            aging_alpha=0.045,
            aging_beta=0.77,
        )
    return CellParameters(chemistry=BatteryChemistry.NMC)
