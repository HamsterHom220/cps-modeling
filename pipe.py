import numpy as np

class Pipe:
    """Класс для моделирования трубы в системе катодной защиты"""
    
    def __init__(self, length=10.0, radius=0.2, y_position=4.0):
        """
        Инициализация параметров трубы
        
        Args:
            length: Длина трубы (м)
            radius: Радиус трубы (м)
            y_position: Y-координата центра трубы
        """
        self.length = length
        self.radius = radius
        self.y_position = y_position
        
        # Динамические параметры (могут меняться со временем)
        self.coating_quality = 1.0  # Качество покрытия (0-1)
        self.roughness = 0.1  # Шероховатость поверхности
        self.age_years = 0.0  # Возраст трубы (лет)
        self.potential = -0.85  # Потенциал трубы (В)
        self.current_density = 0.0  # Плотность тока (А/м²)
        
        # Физические параметры трубы
        self.metal_conductivity = 5.0e6  # Электропроводность стали (См/м)
        self.polarization_resistance = 10.0  # Поляризационное сопротивление (Ом·м²)
        
    def apply_degradation(self, base_coating_quality, base_roughness, t_years):
        """
        Применение эффектов деградации к трубе
        
        Args:
            base_coating_quality: Исходное качество покрытия
            base_roughness: Исходная шероховатость
            t_years: Время эксплуатации (лет)
            
        Returns:
            Обновленные параметры трубы
        """
        # Деградация покрытия
        if t_years <= 10:
            degradation = 0.02 * t_years
        else:
            degradation = 0.02 * 10 + 0.04 * (t_years - 10)
        
        self.coating_quality = max(0.15, base_coating_quality * np.exp(-degradation))
        
        # Увеличение шероховатости из-за коррозии
        corrosion_depth = 0.01 * t_years
        corrosion_factor = min(1.0, corrosion_depth / 10.0 * 3.0)
        self.roughness = min(1.5, base_roughness * (1.0 + corrosion_factor))
        
        # Увеличение поляризационного сопротивления из-за коррозионных продуктов
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
    
    def calculate_boundary_condition(self, phi, V_app, soil_conductivity):
        """
        Расчет граничного условия на поверхности трубы
        
        Args:
            phi: Потенциал в грунте у поверхности трубы
            V_app: Приложенное напряжение
            soil_conductivity: Проводимость грунта
            
        Returns:
            Плотность тока на поверхности трубы
        """
        # Потенциал стали трубы
        E_steel = -0.85  # Стандартный потенциал защиты
        
        # Сопротивление покрытия
        R_coating = self.calculate_coating_resistance()
        
        # Поляризационная характеристика (упрощенная Батлера-Фольмера)
        overpotential = phi - E_steel
        exchange_current = 1e-3  # Ток обмена (А/м²)
        
        if overpotential > 0:
            # Анодная реакция (коррозия)
            beta_a = 0.1  # Коэффициент переноса анодной реакции
            current = exchange_current * (np.exp(beta_a * overpotential / 0.025) - 1)
        else:
            # Катодная реакция (защита)
            beta_c = 0.5  # Коэффициент переноса катодной реакции
            current = exchange_current * (np.exp(-beta_c * overpotential / 0.025) - 1)
        
        # Учет сопротивления покрытия
        coating_current = (phi - E_steel) / R_coating if R_coating > 0 else 0
        
        # Суммарный ток
        total_current = current + coating_current
        
        self.current_density = total_current
        return total_current
    
    def calculate_nonlinear_boundary_condition(self, phi_external, V_app, soil_conductivity, t_years):
        """
        Нелинейное граничное условие Батлера-Фольмера для трубы
        phi_external: потенциал в грунте у поверхности трубы (В)
        Возвращает: плотность тока (А/м²), положительная = анодный ток (коррозия)
        """
        # Параметры электрохимической кинетики стали
        E_corr = -0.65  # Потенциал коррозии стали (В относительно Cu/CuSO4)
        i_corr = 1e-6   # Плотность тока коррозии (А/м²)
        beta_a = 0.1    # Анодный коэффициент переноса (В⁻¹)
        beta_c = 0.5    # Катодный коэффициент переноса (В⁻¹)
        
        # Перенапряжение
        eta = phi_external - E_corr
        
        # Уравнение Батлера-Фольмера
        if eta > 0:
            # Анодная реакция (коррозия)
            i = i_corr * (np.exp(beta_a * eta / 0.025) - np.exp(-beta_c * eta / 0.025))
        else:
            # Катодная реакция (защита)
            i = i_corr * (np.exp(beta_a * eta / 0.025) - np.exp(-beta_c * eta / 0.025))
        
        # Влияние покрытия (сопротивление)
        R_coating = self.calculate_coating_resistance()
        i_coating = (phi_external - (-0.85)) / R_coating if R_coating > 0 else 0
        
        # Суммарный ток
        total_current = i + i_coating
        
        # Ограничение
        max_current = 0.1  # А/м²
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
        Детальный расчет сопротивления покрытия с учетом деградации
        """
        if t_years is None:
            t_years = self.age_years
        
        coating_quality = self.coating_quality
        
        # Базовое сопротивление идеального покрытия [Ом·м²]
        R_perfect = 1e6
        
        # Влияние качества (экспоненциальная зависимость)
        R_quality = R_perfect * (coating_quality ** 2)
        
        # Деградация со временем
        degradation_rate = 0.1 * (1.0 - coating_quality)
        R_time = R_quality * np.exp(-degradation_rate * t_years)
        
        # Минимальное сопротивление через дефекты
        defect_factor = 1.0 - coating_quality
        R_defects = 100.0 / max(defect_factor, 0.01)
        
        # Общее сопротивление (параллельное соединение)
        if R_time > 0 and R_defects > 0:
            R_total = 1.0 / (1.0/R_time + 1.0/R_defects)
        else:
            R_total = min(R_time, R_defects)
        
        # Влияние влажности
        humidity_factor = 1.0 + 2.0 * (1.0 - humidity)  # Сухой грунт увеличивает сопротивление
        R_total *= humidity_factor
        
        return max(R_total, 0.01)
    
    def calculate_effective_resistance(self, phi_surface, T=15.0, humidity=0.8):
        """
        Расчет эффективного сопротивления для линеаризации
        R_eff = dφ/di
        """
        # Сопротивление покрытия
        R_coating = self.calculate_coating_resistance_detailed(self.age_years, humidity)
        
        # Электрохимическое сопротивление (производная Батлер-Вольмера)
        E_eq = -0.65  # Равновесный потенциал стали
        eta = phi_surface - E_eq
        
        i0 = 1e-6 * (humidity ** 0.3)  # Плотность тока обмена
        F = 96485
        R_gas = 8.314
        T_K = T + 273.15
        
        if abs(eta) < 1e-6:
            di_dphi = i0 * 4 * F / (R_gas * T_K)
        else:
            alpha = 0.5
            n = 4
            di_dphi = i0 * (alpha*n*F/(R_gas*T_K) * np.exp(-alpha*n*F*eta/(R_gas*T_K)) +
                          (1-alpha)*n*F/(R_gas*T_K) * np.exp((1-alpha)*n*F*eta/(R_gas*T_K)))
        
        R_electrochem = 1.0 / max(abs(di_dphi), 1e-6)
        
        return R_coating + R_electrochem
    
    def get_area(self, domain_width):
        """Площадь трубы в расчетной области"""
        pipe_start, pipe_end, pipe_y, pipe_radius = self.get_pipe_segment(domain_width)
        length_in_domain = pipe_end - pipe_start
        circumference = 2 * np.pi * pipe_radius
        return length_in_domain * circumference
    
    def get_boundary_condition_parameters(self, T=15.0, humidity=0.8):
        """
        Параметры для граничных условий
        """
        return {
            'E_target': -0.85,  # Целевой потенциал защиты
            'E_eq': -0.65,      # Равновесный потенциал
            'R_coating': self.calculate_coating_resistance_detailed(self.age_years, humidity),
            'i0': 1e-6 * (humidity ** 0.3),  # Ток обмена
            'T': T,
            'humidity': humidity
        }