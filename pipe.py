import numpy as np
from electrochemistry import PipeElectrode, protection_criterion, estimate_corrosion_rate


class Pipe:
    """
    Pipeline model for impressed current cathodic protection (ICCP) simulation.

    Models the pipe as a cathode with:
    - Steel electrochemistry (Butler-Volmer kinetics for O2 reduction)
    - Coating resistance (degrades over time)
    - Geometric parameters
    """

    def __init__(self, length=10.0, radius=0.2, y_position=4.0, initial_coating_quality=0.95):
        """
        Initialize pipe parameters.

        Args:
            length: Pipe length in domain (m)
            radius: Pipe radius (m)
            y_position: Y-coordinate of pipe center
            initial_coating_quality: Initial coating quality (0-1)
        """
        self.length = length
        self.radius = radius
        self.y_position = y_position

        # Electrochemistry model (standardized Butler-Volmer + coating)
        self.electrode = PipeElectrode(initial_coating_quality)

        # Dynamic parameters (updated by apply_degradation)
        self.coating_quality = initial_coating_quality
        self.roughness = 0.1  # Surface roughness
        self.age_years = 0.0
        self.potential = -0.85  # Current potential (V vs Cu/CuSO4)
        self.current_density = 0.0  # Current density (A/m²)

        # Physical parameters
        self.metal_conductivity = 5.0e6  # Steel conductivity (S/m)
        self.polarization_resistance = 10.0  # Legacy parameter (Ohm·m²)
        
    def apply_degradation(self, base_coating_quality, base_roughness, t_years):
        """
        Apply time-dependent degradation to pipe.

        Args:
            base_coating_quality: Initial coating quality (0-1)
            base_roughness: Initial surface roughness
            t_years: Time in years

        Returns:
            Tuple of (coating_quality, roughness)
        """
        # Update electrode coating model
        self.electrode.coating.initial_quality = base_coating_quality
        self.coating_quality = self.electrode.apply_degradation(t_years)

        # Surface roughness increases due to corrosion
        corrosion_depth = 0.01 * t_years
        corrosion_factor = min(1.0, corrosion_depth / 10.0 * 3.0)
        self.roughness = min(1.5, base_roughness * (1.0 + corrosion_factor))

        # Legacy: polarization resistance (kept for compatibility)
        self.polarization_resistance = 10.0 * (1.0 + 0.05 * t_years)

        self.age_years = t_years

        return self.coating_quality, self.roughness
    
    def calculate_coating_resistance(self):
        """
        Расчет сопротивления покрытия трубы
        
        Returns:
            Сопротивление покрытия (Ом·м²)
        """
        # Базовое сопротивление хорошего покрытия
        base_resistance = 10000.0  # Ом·м²
        
        # Сопротивление уменьшается с ухудшением качества покрытия
        resistance = base_resistance * self.coating_quality
        
        return max(100.0, resistance)  # Минимальное сопротивление
    
    def calculate_boundary_condition(self, phi, V_app, soil_conductivity, T=288.15, humidity=0.8):
        """
        Calculate current density at pipe surface using Butler-Volmer kinetics.

        Args:
            phi: Potential at pipe surface (V vs Cu/CuSO4)
            V_app: Applied voltage (not used directly, kept for API compatibility)
            soil_conductivity: Soil conductivity (not used directly)
            T: Temperature (K)
            humidity: Soil moisture (0-1)

        Returns:
            Current density at pipe surface (A/m²), positive = anodic (corrosion)
        """
        # Get electrochemical current from standardized Butler-Volmer
        i_electrochem = self.electrode.kinetics.current_density(phi, T, humidity)

        # Coating leakage current
        R_coating = self.electrode.coating.get_resistance(humidity)
        E_eq = self.electrode.get_equilibrium_potential()
        i_coating = (phi - E_eq) / R_coating if R_coating > 0 else 0

        # Total current
        total_current = i_electrochem + i_coating

        self.current_density = total_current
        return total_current
    
    def calculate_nonlinear_boundary_condition(self, phi_external, V_app, soil_conductivity, t_years,
                                                T=288.15, humidity=0.8):
        """
        Nonlinear Butler-Volmer boundary condition for pipe.

        Args:
            phi_external: Potential at pipe surface (V vs Cu/CuSO4)
            V_app: Applied voltage (kept for API compatibility)
            soil_conductivity: Soil conductivity (kept for API compatibility)
            t_years: Time in years (used to update degradation if needed)
            T: Temperature (K)
            humidity: Soil moisture (0-1)

        Returns:
            Current density (A/m²), positive = anodic (corrosion)
        """
        # Use standardized Butler-Volmer from electrochemistry module
        i_electrochem = self.electrode.kinetics.current_density(phi_external, T, humidity)

        # Coating leakage current
        R_coating = self.electrode.coating.get_resistance(humidity)
        E_target = -0.85  # Protection target potential
        i_coating = (phi_external - E_target) / R_coating if R_coating > 0 else 0

        # Total current
        total_current = i_electrochem + i_coating

        # Limit to physically reasonable range
        max_current = 0.1  # A/m²
        return np.clip(total_current, -max_current, max_current)

    def get_pipe_segment(self, domain_width):
        """
        Получение координат сегмента трубы в расчетной области
        
        Args:
            domain_width: Ширина расчетной области
            
        Returns:
            (start_x, end_x, y, radius)
        """
        pipe_start = (domain_width - self.length) / 2
        pipe_end = (domain_width + self.length) / 2
        return pipe_start, pipe_end, self.y_position, self.radius
    
    def calculate_coating_resistance_detailed(self, t_years=None, humidity=0.8):
        """
        Calculate coating resistance with degradation effects.

        Delegates to electrochemistry.CoatingModel for standardized calculation.

        Args:
            t_years: Time in years (uses self.age_years if None)
            humidity: Soil moisture (0-1)

        Returns:
            Coating resistance (Ohm·m²)
        """
        if t_years is not None and t_years != self.age_years:
            # Temporarily update coating for this calculation
            original_quality = self.electrode.coating.quality
            self.electrode.coating.apply_degradation(t_years)
            R = self.electrode.coating.get_resistance(humidity)
            self.electrode.coating.quality = original_quality
            return R

        return self.electrode.coating.get_resistance(humidity)
    
    def calculate_effective_resistance(self, phi_surface, T=15.0, humidity=0.8):
        """
        Calculate effective resistance for Robin BC linearization.

        R_eff = R_coating + R_charge_transfer

        Uses standardized Butler-Volmer derivative from electrochemistry module.

        Args:
            phi_surface: Surface potential (V vs Cu/CuSO4)
            T: Temperature (Celsius)
            humidity: Soil moisture (0-1)

        Returns:
            Effective resistance (Ohm·m²)
        """
        # Convert temperature to Kelvin
        T_K = T + 273.15

        # Use standardized electrochemistry module
        return self.electrode.get_effective_resistance(phi_surface, T_K, humidity)
    
    def get_area(self, domain_width):
        """Площадь трубы в расчетной области"""
        pipe_start, pipe_end, pipe_y, pipe_radius = self.get_pipe_segment(domain_width)
        length_in_domain = pipe_end - pipe_start
        circumference = 2 * np.pi * pipe_radius
        return length_in_domain * circumference
    
    def get_boundary_condition_parameters(self, T=15.0, humidity=0.8):
        """
        Get parameters for boundary conditions (Robin BC).

        Returns dict with standardized electrochemical parameters.
        """
        T_K = T + 273.15
        return {
            'E_target': -0.85,  # Protection target potential (V vs Cu/CuSO4)
            'E_eq': self.electrode.get_equilibrium_potential(),
            'R_coating': self.electrode.coating.get_resistance(humidity),
            'R_effective': self.electrode.get_effective_resistance(self.potential, T_K, humidity),
            'i0': self.electrode.kinetics.get_exchange_current(T_K, humidity),
            'T': T,
            'humidity': humidity
        }