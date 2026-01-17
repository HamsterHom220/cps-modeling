"""
Electrochemistry module for cathodic protection simulation.

Provides standardized Butler-Volmer kinetics and related calculations
for both pipe (cathode) and anode electrodes.

Reference potentials are vs Cu/CuSO4 (standard for CP systems).
"""

import numpy as np


# Physical constants
FARADAY = 96485.0  # C/mol
GAS_CONSTANT = 8.314  # J/(mol·K)


class ElectrodeKinetics:
    """
    Butler-Volmer kinetics for electrode reactions.

    The Butler-Volmer equation:
        i = i0 * [exp(alpha_a * n * F * eta / RT) - exp(-alpha_c * n * F * eta / RT)]

    where:
        i = current density (A/m²), positive = anodic (metal dissolution)
        i0 = exchange current density (A/m²)
        alpha_a, alpha_c = anodic/cathodic transfer coefficients (dimensionless, sum to 1)
        n = number of electrons transferred
        F = Faraday constant (96485 C/mol)
        R = gas constant (8.314 J/(mol·K))
        T = temperature (K)
        eta = overpotential = E - E_eq (V)
    """

    def __init__(self, E_eq, i0, alpha_a, alpha_c, n, name="electrode"):
        """
        Initialize electrode kinetics.

        Args:
            E_eq: Equilibrium potential (V vs Cu/CuSO4)
            i0: Exchange current density (A/m²)
            alpha_a: Anodic transfer coefficient (dimensionless)
            alpha_c: Cathodic transfer coefficient (dimensionless)
            n: Number of electrons transferred
            name: Electrode name for debugging
        """
        self.E_eq = E_eq
        self.i0_base = i0
        self.alpha_a = alpha_a
        self.alpha_c = alpha_c
        self.n = n
        self.name = name

        # Validate transfer coefficients
        if not (0 < alpha_a < 1 and 0 < alpha_c < 1):
            raise ValueError(f"Transfer coefficients must be between 0 and 1")
        if abs(alpha_a + alpha_c - 1.0) > 0.01:
            raise ValueError(f"Transfer coefficients should sum to ~1, got {alpha_a + alpha_c}")

    def get_exchange_current(self, T=288.15, humidity=0.8):
        """
        Get exchange current density with environmental corrections.

        Args:
            T: Temperature (K)
            humidity: Soil moisture content (0-1)

        Returns:
            Exchange current density (A/m²)
        """
        # Temperature correction (Arrhenius-like)
        T_ref = 288.15  # 15°C reference
        T_factor = np.exp(0.05 * (T - T_ref))  # ~5% per degree

        # Humidity correction (more moisture = better ionic conductivity)
        humidity_factor = humidity ** 0.3

        return self.i0_base * T_factor * humidity_factor

    def current_density(self, phi, T=288.15, humidity=0.8):
        """
        Calculate current density using Butler-Volmer equation.

        Args:
            phi: Electrode potential (V vs Cu/CuSO4)
            T: Temperature (K)
            humidity: Soil moisture (0-1)

        Returns:
            Current density (A/m²), positive = anodic
        """
        eta = phi - self.E_eq  # Overpotential
        i0 = self.get_exchange_current(T, humidity)

        # Thermal voltage
        RT_nF = GAS_CONSTANT * T / (self.n * FARADAY)

        # Butler-Volmer equation
        i = i0 * (np.exp(self.alpha_a * eta / RT_nF) -
                  np.exp(-self.alpha_c * eta / RT_nF))

        return i

    def linearized_resistance(self, phi, T=288.15, humidity=0.8):
        """
        Calculate linearized charge transfer resistance: R_ct = d(eta)/di

        This is the slope of the polarization curve at the operating point,
        used for Robin boundary condition linearization.

        Args:
            phi: Electrode potential (V vs Cu/CuSO4)
            T: Temperature (K)
            humidity: Soil moisture (0-1)

        Returns:
            Charge transfer resistance (Ohm·m²)
        """
        eta = phi - self.E_eq
        i0 = self.get_exchange_current(T, humidity)

        RT_nF = GAS_CONSTANT * T / (self.n * FARADAY)

        # Derivative of Butler-Volmer: di/d(eta)
        di_deta = (i0 / RT_nF) * (
            self.alpha_a * np.exp(self.alpha_a * eta / RT_nF) +
            self.alpha_c * np.exp(-self.alpha_c * eta / RT_nF)
        )

        # Resistance is inverse of derivative
        # Add small value to avoid division by zero
        R_ct = 1.0 / max(abs(di_deta), 1e-10)

        return R_ct

    def tafel_slope(self, branch='cathodic'):
        """
        Get Tafel slope for linear region of polarization curve.

        Args:
            branch: 'anodic' or 'cathodic'

        Returns:
            Tafel slope (V/decade)
        """
        T = 288.15  # Standard temperature
        if branch == 'anodic':
            return 2.303 * GAS_CONSTANT * T / (self.alpha_a * self.n * FARADAY)
        else:
            return 2.303 * GAS_CONSTANT * T / (self.alpha_c * self.n * FARADAY)


# =============================================================================
# Standard electrode configurations for ICCP systems
# =============================================================================

def create_steel_kinetics():
    """
    Create electrode kinetics for carbon steel pipe (cathode in CP system).

    Primary reaction: O2 + 2H2O + 4e- -> 4OH- (oxygen reduction)
    Secondary: 2H2O + 2e- -> H2 + 2OH- (hydrogen evolution at very negative potentials)

    Parameters from corrosion literature for steel in soil.
    """
    return ElectrodeKinetics(
        E_eq=-0.65,      # Corrosion potential of steel vs Cu/CuSO4
        i0=1e-6,         # Exchange current density (A/m²) - typical for O2 reduction on steel
        alpha_a=0.5,     # Anodic transfer coefficient
        alpha_c=0.5,     # Cathodic transfer coefficient
        n=4,             # Electrons for O2 reduction
        name="steel_cathode"
    )


def create_anode_kinetics():
    """
    Create electrode kinetics for ICCP (Impressed Current Cathodic Protection) anode.

    In ICCP systems, the anode is connected to a rectifier providing DC current.
    Common anode materials: MMO (Mixed Metal Oxide), graphite, silicon iron.
    These have relatively fast kinetics (low polarization) compared to the cathode.

    The effective equilibrium potential depends on the applied voltage and
    system configuration.
    """
    return ElectrodeKinetics(
        E_eq=1.2,        # Effective equilibrium for ICCP anode [V vs Cu/CuSO4]
        i0=1e-4,         # Higher exchange current (faster kinetics) [A/m²]
        alpha_a=0.5,     # Symmetric transfer coefficient
        alpha_c=0.5,
        n=2,             # Simplified electron transfer number
        name="iccp_anode"
    )


# =============================================================================
# Coating model
# =============================================================================

class CoatingModel:
    """
    Model for pipeline coating resistance.

    Coating provides electrical isolation but degrades over time,
    developing holidays (defects) that expose bare steel.
    """

    def __init__(self, initial_quality=0.95):
        """
        Args:
            initial_quality: Initial coating quality (0-1), 1 = perfect
        """
        self.initial_quality = initial_quality
        self.quality = initial_quality

    def apply_degradation(self, t_years):
        """
        Apply time-dependent degradation to coating.

        Args:
            t_years: Time in years

        Returns:
            Current coating quality
        """
        # Two-phase degradation model:
        # Phase 1 (0-10 years): slow degradation
        # Phase 2 (10+ years): accelerated degradation
        if t_years <= 10:
            degradation_rate = 0.02  # 2% per year
        else:
            degradation_rate = 0.02 + 0.02 * (t_years - 10) / 20  # Accelerating

        degradation = degradation_rate * t_years
        self.quality = max(0.15, self.initial_quality * np.exp(-degradation))

        return self.quality

    def get_resistance(self, humidity=0.8):
        """
        Calculate coating resistance.

        Model: Parallel combination of intact coating and defect paths.

        Args:
            humidity: Soil moisture (0-1)

        Returns:
            Coating resistance (Ohm·m²)
        """
        # Perfect coating resistance
        R_perfect = 1e6  # Ohm·m²

        # Intact coating contribution (quality-dependent)
        R_intact = R_perfect * (self.quality ** 2)

        # Defect path resistance (inverse of defect area fraction)
        defect_fraction = 1.0 - self.quality
        R_defects = 100.0 / max(defect_fraction, 0.01)

        # Parallel combination
        if R_intact > 0 and R_defects > 0:
            R_total = 1.0 / (1.0/R_intact + 1.0/R_defects)
        else:
            R_total = min(R_intact, R_defects)

        # Humidity effect: dry soil increases resistance
        humidity_factor = 1.0 + 2.0 * (1.0 - humidity)
        R_total *= humidity_factor

        return max(R_total, 0.01)


# =============================================================================
# Combined electrode model for CP simulation
# =============================================================================

class PipeElectrode:
    """
    Combined model for pipe electrode including coating and steel kinetics.
    """

    def __init__(self, initial_coating_quality=0.95):
        self.kinetics = create_steel_kinetics()
        self.coating = CoatingModel(initial_coating_quality)

    def apply_degradation(self, t_years):
        """Apply time-dependent degradation."""
        return self.coating.apply_degradation(t_years)

    def get_effective_resistance(self, phi, T=288.15, humidity=0.8):
        """
        Get total effective resistance for linearization.

        Total = coating resistance + charge transfer resistance

        Args:
            phi: Surface potential (V vs Cu/CuSO4)
            T: Temperature (K)
            humidity: Soil moisture (0-1)

        Returns:
            Effective resistance (Ohm·m²)
        """
        R_coating = self.coating.get_resistance(humidity)
        R_ct = self.kinetics.linearized_resistance(phi, T, humidity)

        return R_coating + R_ct

    def get_equilibrium_potential(self):
        """Get equilibrium potential for Robin BC."""
        return self.kinetics.E_eq


class AnodeElectrode:
    """
    Model for ICCP anode electrode.
    """

    def __init__(self, initial_efficiency=0.95):
        self.kinetics = create_anode_kinetics()
        self.efficiency = initial_efficiency
        self.age_years = 0.0

    def apply_degradation(self, t_years):
        """Apply time-dependent efficiency loss."""
        # Gradual efficiency loss
        if t_years <= 5:
            loss_rate = 0.03
        else:
            loss_rate = 0.03 + 0.02 * (t_years - 5) / 25

        self.efficiency = max(0.25, self.efficiency * np.exp(-loss_rate * t_years))
        self.age_years = t_years

        return self.efficiency

    def get_effective_resistance(self, phi=None, T=288.15, humidity=0.8):
        """
        Get effective anode resistance.

        For ICCP anodes, this is primarily the anode-to-electrolyte resistance.
        """
        R_base = 0.05  # Ohm·m² base resistance

        # Degradation increases resistance
        degradation_factor = 1.0 + 0.1 * self.age_years

        # Temperature effect
        T_ref = 288.15
        T_factor = 1.0 + 0.02 * (T - T_ref) / 10

        # Humidity effect
        humidity_factor = 1.0 / max(humidity, 0.1)

        return R_base * degradation_factor * T_factor * humidity_factor

    def get_equilibrium_potential(self):
        """Get equilibrium potential for Robin BC."""
        return self.kinetics.E_eq


# =============================================================================
# Utility functions
# =============================================================================

def protection_criterion(phi, threshold=-0.85):
    """
    Check if potential meets protection criterion.

    Standard criterion: phi <= -0.85 V vs Cu/CuSO4 for steel.

    Args:
        phi: Potential (V vs Cu/CuSO4), can be array
        threshold: Protection threshold (V)

    Returns:
        Boolean or boolean array indicating protection status
    """
    return phi <= threshold


def estimate_corrosion_rate(phi, i, T=288.15):
    """
    Estimate corrosion rate from potential and current density.

    Uses simplified Faraday's law for iron dissolution.

    Args:
        phi: Potential (V vs Cu/CuSO4)
        i: Anodic current density (A/m²)
        T: Temperature (K)

    Returns:
        Corrosion rate (mm/year)
    """
    if phi <= -0.85:
        # Protected: negligible corrosion
        return 0.001

    # Faraday's law: mass loss rate = i * M / (n * F)
    # For Fe: M = 55.85 g/mol, n = 2
    M_Fe = 55.85e-3  # kg/mol
    n_Fe = 2
    rho_Fe = 7874  # kg/m³

    # Convert to mm/year
    # i [A/m²] * M [kg/mol] / (n * F [C/mol]) / rho [kg/m³] * (3600*24*365) [s/year] * 1000 [mm/m]
    rate = abs(i) * M_Fe / (n_Fe * FARADAY) / rho_Fe * 3.15e10  # mm/year

    # Temperature acceleration (Q10 ~ 2)
    T_factor = 2.0 ** ((T - 288.15) / 10)

    return min(rate * T_factor, 10.0)  # Cap at 10 mm/year
