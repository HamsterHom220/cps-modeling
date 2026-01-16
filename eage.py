import numpy as np

class ExtendedAnode:
    """Класс для моделирования протяженного анода в системе катодной защиты"""
    
    def __init__(self, x_position=None, y_position=1.5, length=1.0, width=2.0):
        """
        Инициализация параметров протяженного анода
        
        Args:
            x_position: X-координата центра анода (None = середина области)
            y_position: Y-координата центра анода
            length: Длина анода в направлении X (м)
            width: Ширина анода в направлении Y (м)
        """
        self.x_position = x_position
        self.y_position = y_position
        self.length = length
        self.width = width
        
        # Динамические параметры
        self.efficiency = 0.9  # КПД анода (0-1)
        self.age_years = 0.0  # Возраст анода (лет)
        self.potential = 0.0  # Потенциал анода (В)
        self.current_output = 0.0  # Выходной ток (А/м)
        
        # Материальные параметры анода
        self.material_conductivity = 1.0e4  # Проводимость материала анода (См/м)
        self.driving_voltage = 1.5  # Движущее напряжение анода (В)
        
    def apply_degradation(self, base_efficiency, t_years):
        """
        Применение эффектов износа анода
        
        Args:
            base_efficiency: Исходный КПД анода
            t_years: Время эксплуатации (лет)
            
        Returns:
            Обновленный КПД анода
        """
        # Потребление анода (деградация эффективности)
        if t_years <= 5:
            consumption = 0.03 * t_years
        else:
            consumption = 0.03 * 5 + 0.05 * (t_years - 5)
        
        self.efficiency = max(0.25, base_efficiency * np.exp(-consumption))
        
        # Уменьшение движущего напряжения со временем
        self.driving_voltage = 1.5 * (1.0 - 0.02 * min(t_years, 20))
        
        self.age_years = t_years
        
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
    
    def calculate_nonlinear_boundary_condition(self, phi_external, V_app, soil_conductivity):
        """
        Нелинейное граничное условие для анода
        phi_external: потенциал в грунте у поверхности анода (В)
        Возвращает: плотность тока (А/м²), отрицательная = катодный ток (защита)
        """
        # Потенциал анодного материала
        E_anode = V_app * self.efficiency
        
        # Движущее напряжение
        driving_potential = E_anode - phi_external
        
        # Нелинейная характеристика (квадратичная зависимость)
        R_output = 0.1  # Базовое сопротивление выхода (Ом·м²)
        
        # Нелинейность: при больших напряжениях сопротивление уменьшается
        nonlinear_factor = 1.0 / (1.0 + 0.5 * abs(driving_potential))
        effective_resistance = R_output * nonlinear_factor
        
        # Плотность тока
        current_density = driving_potential / effective_resistance if effective_resistance > 0 else 0
        
        # Ограничение
        max_current_density = 1.0  # А/м²
        return np.clip(current_density, -max_current_density, 0)  # Только отрицательный (анодный) ток

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
        Расчет сопротивления анода с учетом деградации и условий
        """
        # Базовое сопротивление
        R_base = 0.05  # Ом·м²
        
        # Влияние времени (деградация анода)
        degradation_factor = 1.0 + 0.1 * self.age_years
        
        # Влияние температуры
        T_ref = 20.0  # °C
        T_factor = 1.0 + 0.02 * (T - T_ref)
        
        # Влияние влажности
        humidity_factor = 1.0 / max(humidity, 0.1)
        
        # Влияние тока (если задан потенциал)
        if phi_surface is not None:
            V_app = self.potential / self.efficiency if self.efficiency > 0 else 1.0
            current_density = (V_app * self.efficiency - phi_surface) / (R_base * degradation_factor)
            current_factor = 1.0 + 0.5 * min(abs(current_density), 1.0)
        else:
            current_factor = 1.0
        
        return R_base * degradation_factor * T_factor * humidity_factor * current_factor
    
    def get_boundary_condition_parameters(self, V_app, T=15.0, humidity=0.8):
        """
        Параметры для граничных условий анода
        """
        return {
            'E_anode': V_app * self.efficiency,
            'E_eq': 1.2,  # Равновесный потенциал Mg
            'R_output': self.calculate_anode_resistance(T=T, humidity=humidity),
            'i0': 1e-4 * (humidity ** 0.5),  # Ток обмена для Mg
            'T': T,
            'humidity': humidity
        }
    
    def get_area(self):
        """Площадь анода"""
        return self.length * self.width