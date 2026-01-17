import numpy as np
from dolfinx import fem, mesh
import matplotlib.pyplot as plt
from mpi4py import MPI
import ufl

class SoilModel:
    def __init__(self, domain, params, domain_height, pipe_y, enable_plotting=False, seed=None):
        self.R_sigma = params[0]
        self.wetness = params[5]
        self.params = params
        self.domain = domain
        self.domain_height = domain_height
        self.pipe_y = pipe_y
        self.plotting = enable_plotting
        
        # Пространство функций для проводимости
        self.V_sigma = fem.functionspace(domain, ("CG", 1))
        self.sigma = fem.Function(self.V_sigma)
        
        # Seed: если не задан, генерируем из параметров
        if seed is None:
            self.seed = int(abs(sum(params)) * 1000) % 1000000
        else:
            self.seed = seed
            
        np.random.seed(self.seed)
        
        # СОХРАНЯЕМ БАЗОВУЮ СТРУКТУРУ ГРУНТА (не зависит от времени!)
        self._initialize_soil_structure()
        
        print(f"  SoilModel initialized: R_sigma={self.R_sigma} Ω·m, wetness={self.wetness}, seed={self.seed}")
    
    def get_state_dict(self):
        """Возвращает состояние модели для сохранения"""
        return {
            'seed': self.seed,
            'base_factors': self.base_factors.tolist() if hasattr(self, 'base_factors') else [],
            'dof_coords': self.dof_coords.tolist() if hasattr(self, 'dof_coords') else [],
            'params': self.params,
            'layer_depths': self.layer_depths.tolist() if hasattr(self, 'layer_depths') else [],
            'layer_base_conductivities': self.layer_base_conductivities if hasattr(self, 'layer_base_conductivities') else []
        }

    def _initialize_soil_structure(self):
        """Инициализация БАЗОВОЙ структуры грунта (вызывается только один раз!)"""
        print(f"    Initializing BASE soil structure...")
        
        # Получаем DOF координаты
        self.dof_coords = self.V_sigma.tabulate_dof_coordinates()
        self.n_dofs = self.dof_coords.shape[0]
        
        # 1. СОЗДАЕМ БАЗОВЫЕ СЛОИ ГРУНТА (фиксированные для данного случая)
        np.random.seed(self.seed)
        
        # Создаем случайные, но ФИКСИРОВАННЫЕ слои
        n_layers = np.random.randint(3, 6)
        layer_depths = np.sort(np.random.uniform(1.0, self.domain_height - 1.0, n_layers - 1))
        self.layer_depths = np.concatenate([[0], layer_depths, [self.domain_height]])
        
        # Базовые проводимости слоев (без временных эффектов)
        self.layer_base_conductivities = []
        base_cond = 0.15
        
        for i in range(n_layers):
            depth_factor = 0.7 + 0.6 * (self.layer_depths[i] / self.domain_height)
            random_variation = np.random.uniform(0.8, 1.2)
            self.layer_base_conductivities.append(base_cond * depth_factor * random_variation)
        
        # 2. СОЗДАЕМ ФИКСИРОВАННЫЕ РЕГИОНАЛЬНЫЕ ЗОНЫ
        n_zones = np.random.randint(4, 8)
        self.zones = []
        
        for _ in range(n_zones):
            center_x = np.random.uniform(2, 18)
            center_y = np.random.uniform(2, 6)
            radius = np.random.uniform(2, 5)
            
            zone_type = np.random.choice(['sandy', 'clay', 'gravel', 'wet', 'dry'])
            
            if zone_type == 'sandy':
                modifier = np.random.uniform(0.6, 0.9)
            elif zone_type == 'clay':
                modifier = np.random.uniform(1.1, 1.4)
            elif zone_type == 'gravel':
                modifier = np.random.uniform(0.8, 1.1)
            elif zone_type == 'wet':
                modifier = np.random.uniform(1.2, 1.6)
            else:  # 'dry'
                modifier = np.random.uniform(0.5, 0.8)
            
            self.zones.append({
                'center': (center_x, center_y),
                'radius': radius,
                'modifier': modifier,
                'type': zone_type
            })
        
        # 3. СОЗДАЕМ ФИКСИРОВАННЫЕ ЛОКАЛЬНЫЕ ВКЛЮЧЕНИЯ
        n_inclusions = np.random.randint(20, 50)
        self.inclusions = []
        
        for _ in range(n_inclusions):
            inc_x = np.random.uniform(1, 19)
            inc_y = np.random.uniform(1, 7)
            inc_radius = np.random.uniform(0.3, 1.5)
            inc_modifier = np.random.uniform(0.7, 1.3)
            self.inclusions.append((inc_x, inc_y, inc_radius, inc_modifier))
        
        # 4. ВЫЧИСЛЯЕМ БАЗОВЫЕ ФАКТОРЫ ДЛЯ КАЖДОЙ ТОЧКИ (без временных эффектов)
        print(f"    Computing base conductivity factors...")
        self.base_factors = np.ones(self.n_dofs)
        
        for i in range(self.n_dofs):
            xi, yi = self.dof_coords[i, 0], self.dof_coords[i, 1]
            
            # Слои
            layer_factor = 1.0
            for j in range(1, len(self.layer_depths)):
                if yi >= self.layer_depths[j-1] and yi < self.layer_depths[j]:
                    layer_factor = self.layer_base_conductivities[j-1] / 0.15
                    break
            
            # Региональные зоны
            zone_factor = 1.0
            for zone in self.zones:
                center_x, center_y = zone['center']
                dist = np.sqrt((xi - center_x)**2 + (yi - center_y)**2)
                
                if dist < zone['radius']:
                    influence = 1.0 - (dist / zone['radius'])**2
                    zone_factor *= (1.0 + influence * (zone['modifier'] - 1.0))
            
            # Локальные включения
            inclusion_factor = 1.0
            for inc_x, inc_y, inc_radius, inc_modifier in self.inclusions:
                dist = np.sqrt((xi - inc_x)**2 + (yi - inc_y)**2)
                
                if dist < inc_radius:
                    influence = 1.0 - (dist / inc_radius)
                    inclusion_factor *= (1.0 + influence * (inc_modifier - 1.0))
            
            # Текстурированный шум (детерминированный на основе координат)
            noise = self._texture_noise(xi, yi)
            noise_factor = 1.0 + 0.3 * noise
            
            # Базовый влажностный профиль (без влияния времени)
            moisture_base = 0.3 + 0.4 * (1.0 - yi / self.domain_height)
            moisture_factor = 1.0 + 0.5 * self.wetness * moisture_base
            
            # Сохраняем базовый фактор (все кроме временных эффектов)
            self.base_factors[i] = (layer_factor * zone_factor * inclusion_factor * 
                                  noise_factor * moisture_factor)
        
        print(f"    Base structure initialized: {self.n_dofs} points, {n_layers} layers, {n_zones} zones")
    
    def _texture_noise(self, x, y):
        """Детерминированный шум на основе координат"""
        noise = 0.0
        frequencies = [0.3, 0.7, 1.5, 3.0]
        amplitudes = [0.4, 0.2, 0.1, 0.05]
        
        for freq, amp in zip(frequencies, amplitudes):
            noise += amp * (np.sin(x * freq) * np.cos(y * freq * 0.7) +
                           np.sin(x * freq * 1.3 + y * freq * 0.9))
        
        return noise / np.sum(amplitudes)
    
    def get_conductivity(self, t_years=0):
        """
        Получение проводимости грунта с учетом времени
        
        Args:
            t_years: Время эксплуатации (лет)
        Returns:
            Функция проводимости с временными эффектами, но ОДНОЙ И ТОЙ ЖЕ структурой
        """
        #print(f"    Getting conductivity for t={t_years} years (same structure)")
        
        # Базовое значение проводимости
        base_conductivity = 0.2
        
        # Вычисляем проводимость с временными эффектами
        sigma_values = np.zeros(self.n_dofs)
        
        for i in range(self.n_dofs):
            xi, yi = self.dof_coords[i, 0], self.dof_coords[i, 1]
            
            # ВРЕМЕННЫЕ ЭФФЕКТЫ (добавляются к базовой структуре):
            
            # 1. Общее увеличение проводимости со временем
            time_factor = 1.0 + 0.003 * t_years  # +0.3% в год
            
            # 2. Эффект коррозии возле трубы (увеличивается со временем)
            pipe_factor = 1.0
            pipe_dist = np.sqrt((xi - 10.0)**2 + (yi - self.pipe_y)**2)
            if pipe_dist < 1.5:
                pipe_influence = 1.0 - (pipe_dist / 1.5)
                pipe_factor = 1.0 + 0.15 * pipe_influence * t_years / 30.0
            
            # 3. Влияние анода (может высушивать грунт вокруг)
            anode_factor = 1.0
            anode_dist = np.sqrt((xi - 10.0)**2 + (yi - 1.5)**2)
            if anode_dist < 2.0:
                anode_influence = 1.0 - (anode_dist / 2.0)
                # Анод может немного снижать проводимость вокруг себя
                anode_factor = 1.0 - 0.1 * anode_influence * t_years / 30.0
            
            # ИТОГОВАЯ ПРОВОДИМОСТЬ:
            # БАЗОВАЯ СТРУКТУРА × ВРЕМЕННЫЕ ЭФФЕКТЫ
            conductivity = (base_conductivity * 
                          self.base_factors[i] * 
                          time_factor * 
                          pipe_factor * 
                          anode_factor)
            
            # Ограничиваем диапазон
            sigma_values[i] = np.clip(conductivity, 0.05, 0.35)
        
        # Записываем в функцию
        self.sigma.x.array[:] = sigma_values
        
        # Для отладки: сравниваем с предыдущими значениями
        if hasattr(self, 'last_sigma_values'):
            diff = np.mean(np.abs(sigma_values - self.last_sigma_values))
            #print(f"      Mean change from last time: {diff:.6f} S/m")
        
        self.last_sigma_values = sigma_values.copy()

        if self.plotting:
            self.visualize(self.dof_coords, sigma_values, t_years, self.zones, self.layer_depths)
        
        return self.sigma
    
    def get_conductivity_snapshot(self, t_years=0):
        """
        Быстрое получение проводимости (без лишних вычислений)
        """
        # Просто вызываем основной метод
        return self.get_conductivity(t_years)

    def visualize(self, coords, values, t_years, zones, layer_depths):
        """Комплексная визуализация проводимости"""
        print(f"    Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Основная карта проводимости
        scatter1 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], 
                                     c=values, cmap='viridis', s=15, alpha=0.8)
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].set_title(f'Soil Conductivity Map (t={t_years} years)')
        axes[0, 0].set_aspect('equal')
        
        # Показываем зоны
        for zone in zones:
            center_x, center_y = zone['center']
            circle = plt.Circle((center_x, center_y), zone['radius'], 
                              fill=False, color='red', alpha=0.5, linestyle='--')
            axes[0, 0].add_patch(circle)
            axes[0, 0].text(center_x, center_y, zone['type'][0], 
                          fontsize=8, ha='center', va='center', color='red')
        
        # Показываем слои
        for depth in layer_depths[1:-1]:
            axes[0, 0].axhline(y=depth, color='blue', linestyle=':', alpha=0.3)
        
        axes[0, 0].axhline(y=self.pipe_y, color='red', linewidth=2, alpha=0.7, label='Pipe')
        axes[0, 0].axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Anode')
        axes[0, 0].legend()
        plt.colorbar(scatter1, ax=axes[0, 0], label='Conductivity (S/m)')
        
        # 2. 3D поверхность
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = fig.add_subplot(2, 3, 2, projection='3d')
        
        # Для 3D визуализации берем подвыборку точек
        sample_idx = np.random.choice(len(coords), min(1000, len(coords)), replace=False)
        ax3d.scatter(coords[sample_idx, 0], coords[sample_idx, 1], 
                    values[sample_idx], c=values[sample_idx], cmap='viridis', 
                    s=10, alpha=0.6)
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Conductivity (S/m)')
        ax3d.set_title('3D Conductivity Distribution')
        
        # 3. Гистограмма
        axes[0, 2].hist(values, bins=40, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Conductivity (S/m)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Conductivity Distribution')
        axes[0, 2].axvline(x=np.mean(values), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(values):.4f}')
        axes[0, 2].legend()
        
        # 4. Горизонтальные профили на разных глубинах
        depths_to_plot = [1.0, 3.0, 5.0, 7.0]
        colors = ['red', 'green', 'blue', 'purple']
        
        for depth, color in zip(depths_to_plot, colors):
            # Находим точки близко к этой глубине
            depth_mask = np.abs(coords[:, 1] - depth) < 0.2
            if np.any(depth_mask):
                depth_coords = coords[depth_mask, 0]
                depth_values = values[depth_mask]
                
                # Сортируем по X
                sort_idx = np.argsort(depth_coords)
                axes[1, 0].plot(depth_coords[sort_idx], depth_values[sort_idx], 
                              color=color, alpha=0.7, label=f'y={depth}m')
        
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Conductivity (S/m)')
        axes[1, 0].set_title('Horizontal Profiles at Different Depths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Вертикальные профили в разных местах
        x_positions = [5.0, 10.0, 15.0]
        colors = ['red', 'green', 'blue']
        
        for x, color in zip(x_positions, colors):
            # Находим точки близко к этому X
            x_mask = np.abs(coords[:, 0] - x) < 0.2
            if np.any(x_mask):
                x_coords = coords[x_mask, 1]
                x_values = values[x_mask]
                
                # Сортируем по Y
                sort_idx = np.argsort(x_coords)
                axes[1, 1].plot(x_values[sort_idx], x_coords[sort_idx], 
                              color=color, alpha=0.7, label=f'x={x}m')
        
        axes[1, 1].set_xlabel('Conductivity (S/m)')
        axes[1, 1].set_ylabel('Depth Y (m)')
        axes[1, 1].set_title('Vertical Profiles at Different X Positions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis()  # Глубина увеличивается вниз
        
        # 6. Статистика и информация
        axes[1, 2].axis('off')
        
        stats_text = (f"SOIL CONDUCTIVITY STATISTICS (t={t_years} years)\n\n"
                     f"Parameters:\n"
                     f"  R_sigma = {self.R_sigma:.2f} Ω·m\n"
                     f"  Wetness = {self.wetness:.2f}\n"
                     f"  Random seed = {self.seed}\n\n"
                     f"Statistics:\n"
                     f"  Min: {np.min(values):.4f} S/m\n"
                     f"  Max: {np.max(values):.4f} S/m\n"
                     f"  Mean: {np.mean(values):.4f} S/m\n"
                     f"  Std: {np.std(values):.4f} S/m\n"
                     f"  Range: {np.max(values)-np.min(values):.4f} S/m\n\n"
                     f"Distribution:\n")
        
        # Добавляем распределение по квантилям
        quantiles = np.percentile(values, [10, 25, 50, 75, 90])
        for q, val in zip([10, 25, 50, 75, 90], quantiles):
            stats_text += f"  {q}%: {val:.4f} S/m\n"
        
        # Информация о зонах
        stats_text += f"\nSoil Zones ({len(zones)} total):\n"
        zone_types = {}
        for zone in zones:
            zone_types[zone['type']] = zone_types.get(zone['type'], 0) + 1
        
        for zone_type, count in zone_types.items():
            stats_text += f"  {zone_type}: {count} zones\n"
        
        axes[1, 2].text(0.05, 0.95, stats_text, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        
        plt.tight_layout()
        filename = f"./images/soil/soil_conductivity_random_t{t_years}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization saved to {filename}")
        
        # Выводим краткую статистику
        print(f"\n    Conductivity statistics:")
        print(f"      Min: {np.min(values):.4f} S/m")
        print(f"      Max: {np.max(values):.4f} S/m")
        print(f"      Mean: {np.mean(values):.4f} S/m")
        print(f"      Std: {np.std(values):.4f} S/m")


# ТЕСТ СЛУЧАЙНОЙ МОДЕЛИ ГРУНТА
if __name__ == "__main__":

    # Создаем домен
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([20.0, 8.0])],
        [40, 16],  # Умеренное количество элементов
        mesh.CellType.triangle
    )
    
    # Тестовые параметры
    test_params = [4.5, 0.2, 0.9, 7.5, 9.0, 0.5, 0.0, 0.92]
    
    print(f"Parameters: R_sigma={test_params[0]}, wetness={test_params[5]}")
    print('='*70)
    
    soil = SoilModel(domain, test_params, 8.0, 4.0, True)
    
    # Выбираем несколько тестовых точек
    test_points_indices = np.random.choice(soil.n_dofs, 10, replace=False)
    
    # Собираем значения во времени
    time_points = [0, 5, 10, 15, 20, 25, 30]
    values_over_time = {idx: [] for idx in test_points_indices}
    
    for t in time_points:
        sigma_func = soil.get_conductivity(t)
        sigma_values = sigma_func.x.array
        
        print(f"\nTime t = {t:2d} years:")
        print(f"  Global: min={np.min(sigma_values):.4f}, max={np.max(sigma_values):.4f}")
        