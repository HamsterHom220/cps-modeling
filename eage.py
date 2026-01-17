import numpy as np
from electrochemistry import AnodeElectrode


class ExtendedAnode:
    """
    Extended anode model for impressed current cathodic protection (ICCP) system.

    In ICCP systems, the anode is connected to a rectifier that provides
    the driving voltage. The anode material (MMO, graphite, etc.) has
    relatively fast kinetics compared to the pipe cathode.
    """

    def __init__(self, x_position=None, y_position=1.5, length=1.0, width=2.0,
                 initial_efficiency=0.95):
        """
        Initialize extended anode parameters.

        Args:
            x_position: X-coordinate of anode center (None = domain center)
            y_position: Y-coordinate of anode center
            length: Anode length in X direction (m)
            width: Anode width in Y direction (m)
            initial_efficiency: Initial anode efficiency (0-1)
        """
        self.x_position = x_position
        self.y_position = y_position
        self.length = length
        self.width = width

        # Electrochemistry model (standardized)
        self.electrode = AnodeElectrode(initial_efficiency)

        # Dynamic parameters
        self.efficiency = initial_efficiency
        self.age_years = 0.0
        self.potential = 0.0  # Current potential (V)
        self.current_output = 0.0  # Output current (A/m)

        # Material parameters
        self.material_conductivity = 1.0e4  # Anode material conductivity (S/m)
        self.driving_voltage = 1.5  # Driving voltage (V)
        
    def apply_degradation(self, base_efficiency, t_years):
        """
        Apply time-dependent degradation to anode.

        Args:
            base_efficiency: Initial anode efficiency (0-1)
            t_years: Time in years

        Returns:
            Updated anode efficiency
        """
        # Update electrode model
        self.electrode.efficiency = base_efficiency
        self.efficiency = self.electrode.apply_degradation(t_years)

        # Driving voltage decreases over time
        self.driving_voltage = 1.5 * (1.0 - 0.02 * min(t_years, 20))

        self.age_years = t_years
        self.electrode.age_years = t_years

        return self.efficiency
    
    def calculate_boundary_condition(self, phi, V_app):
        """
        Расчет граничного условия на поверхности анода
        
        Args:
            phi: Потенциал в грунте у поверхности анода
            V_app: Приложенное напряжение
            
        Returns:
            Плотность тока на поверхности анода
        """
        # Потенциал анодного материала
        E_anode = V_app * self.efficiency
        
        # Движущее напряжение
        driving_potential = E_anode - phi
        
        # Сопротивление выхода анода
        R_output = 0.1  # Ом·м² (сопротивление перехода анод-грунт)
        
        # Ток анода (линейная аппроксимация)
        current_density = driving_potential / R_output if R_output > 0 else 0
        
        # Ограничение максимальной плотности тока
        max_current_density = 1.0  # А/м²
        current_density = np.clip(current_density, -max_current_density, 0)
        
        self.current_output = abs(current_density) * self.length
        return current_density
    
    def calculate_nonlinear_boundary_condition(self, phi_external, V_app, soil_conductivity,
                                                T=288.15, humidity=0.8):
        """
        Nonlinear boundary condition for anode using Butler-Volmer kinetics.

        Args:
            phi_external: Potential at anode surface (V vs Cu/CuSO4)
            V_app: Applied voltage from rectifier (V)
            soil_conductivity: Soil conductivity (kept for API compatibility)
            T: Temperature (K)
            humidity: Soil moisture (0-1)

        Returns:
            Current density (A/m²), negative = current flowing into soil (anode operating)
        """
        # Effective anode potential
        E_anode = V_app * self.efficiency

        # Use standardized Butler-Volmer from electrochemistry module
        # Note: For ICCP anodes, kinetics are typically fast
        i_electrochem = self.electrode.kinetics.current_density(phi_external, T, humidity)

        # Add ohmic contribution from driving voltage
        R_output = self.electrode.get_effective_resistance(phi_external, T, humidity)
        driving_potential = E_anode - phi_external
        i_ohmic = driving_potential / R_output if R_output > 0 else 0

        # Total current (dominated by ohmic for ICCP anodes)
        current_density = i_ohmic + 0.1 * i_electrochem  # Kinetics is minor contribution

        # Limit to physically reasonable range
        max_current_density = 1.0  # A/m²
        return np.clip(current_density, -max_current_density, max_current_density)

    def get_anode_region(self, domain_width):
        """
        Получение координат области анода
        
        Args:
            domain_width: Ширина расчетной области
            
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        if self.x_position is None:
            self.x_position = domain_width / 2
            
        x_min = self.x_position - self.length / 2
        x_max = self.x_position + self.length / 2
        y_min = self.y_position - self.width / 2
        y_max = self.y_position + self.width / 2
        
        return x_min, x_max, y_min, y_max

    def calculate_anode_resistance(self, phi_surface=None, T=15.0, humidity=0.8):
        """
        Calculate anode resistance including degradation and environmental effects.

        Uses standardized electrochemistry module.

        Args:
            phi_surface: Surface potential (V vs Cu/CuSO4), optional
            T: Temperature (Celsius)
            humidity: Soil moisture (0-1)

        Returns:
            Anode resistance (Ohm·m²)
        """
        # Convert temperature to Kelvin
        T_K = T + 273.15

        # Use standardized electrochemistry module
        return self.electrode.get_effective_resistance(phi_surface, T_K, humidity)
    
    def get_boundary_condition_parameters(self, V_app, T=15.0, humidity=0.8):
        """
        Get parameters for anode boundary conditions (Robin BC).

        Returns dict with standardized electrochemical parameters.
        """
        T_K = T + 273.15
        return {
            'E_anode': V_app * self.efficiency,
            'E_eq': self.electrode.get_equilibrium_potential(),
            'R_output': self.electrode.get_effective_resistance(None, T_K, humidity),
            'i0': self.electrode.kinetics.get_exchange_current(T_K, humidity),
            'T': T,
            'humidity': humidity
        }
    
    def get_area(self):
        """Площадь анода"""
        return self.length * self.width