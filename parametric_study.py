import numpy as np
import json
import os
import h5py
from datetime import datetime
from dolfinx import mesh, fem
from mpi4py import MPI

from simulator import CPS_DegradationSimulator
from soil import SoilModel


class CPS_DatasetGenerator:
    """Генератор датасета с управлением моделями грунта"""
    
    def __init__(self, save_dir="./dataset"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.data_file = os.path.join(save_dir, "cp_dataset_full.h5")
        self.metadata_file = os.path.join(save_dir, "metadata.json")
        self.soil_models_file = os.path.join(save_dir, "soil_models.json")
        
        # Создаем общий домен ОДИН РАЗ для всех случаев
        print("Creating shared domain for all cases...")
        self.shared_domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([20.0, 8.0])],
            [80, 32],
            mesh.CellType.triangle
        )
        
        # Кэш для моделей грунта
        self.soil_models_cache = {}
    
    def generate_base_parameters(self, num_cases):
        """Генерация базовых параметров для случаев"""
        np.random.seed(42)
        
        combinations = []
        for i in range(num_cases):
            if i == 0:  # Хороший случай
                params = [
                    4.5,  # R_sigma (низкое)
                    0.2,  # Шероховатость
                    0.9,  # Качество покрытия (высокое)
                    7.5,  # Кислотность
                    9.0,  # V_app (высокое)
                    0.5,  # Влажность
                    0.0,  # Возраст
                    0.92  # КПД анода
                ]
            elif i == 1:  # Средний случай
                params = [
                    6.0,  # R_sigma
                    0.3,  # Шероховатость
                    0.75, # Качество покрытия
                    7.0,  # Кислотность
                    6.5,  # V_app
                    0.4,  # Влажность
                    0.0,  # Возраст
                    0.85  # КПД анода
                ]
            else:  # Плохой случай
                params = [
                    7.5,  # R_sigma (высокое)
                    0.4,  # Шероховатость
                    0.6,  # Качество покрытия (низкое)
                    6.5,  # Кислотность
                    4.5,  # V_app (низкое)
                    0.7,  # Влажность
                    0.0,  # Возраст
                    0.78  # КПД анода
                ]
            
            combinations.append(params)
        
        return combinations
    
    def generate_case_with_soil_model(self, base_params, case_idx, time_points):
        """Генерация данных для одного случая с привязанной моделью грунта"""
        print(f"\nГенерация данных для случая {case_idx}...")
        print(f"  V_app: {base_params[4]:.1f} В, Покрытие: {base_params[2]:.2f}")
        
        # Создаем или получаем модель грунта из кэша
        soil_key = tuple(base_params[:6])  # Используем только релевантные параметры
        
        if soil_key not in self.soil_models_cache:
            print(f"  Создание новой модели грунта для случая {case_idx}...")
            soil_model = SoilModel(
                self.shared_domain, base_params, 
                domain_height=8.0, pipe_y=4.0,
                #enable_plotting=True
            )
            self.soil_models_cache[soil_key] = soil_model
        else:
            print(f"  Использование существующей модели грунта для случая {case_idx}")
            soil_model = self.soil_models_cache[soil_key]
        
        # Создаем симулятор для этого случая
        simulator = CPS_DegradationSimulator()
        simulator.domain = self.shared_domain  # Используем общий домен
        
        # ВАЖНО: Нужно создать функциональные пространства
        simulator.create_mesh_and_function_space()
        
        sequence_results = []
        
        # Для каждого времени используем ОДНУ И ТУ ЖЕ модель грунта
        for t in time_points:
            print(f"\n  Время: {t} лет")
            
            # Настраиваем симулятор с правильной проводимостью
            simulator.setup_soil_model(base_params, t, soil_model)
            
            # Решаем модель
            results = simulator.solve_full_physics_model(base_params, t, soil_model)
            sequence_results.append(results)
        
        # Сохраняем сводку
        if sequence_results:
            initial = sequence_results[0]
            final = sequence_results[-1]
            print(f"\n  📊 ИТОГИ для случая {case_idx}:")
            print(f"    Начало (0 лет): coverage={initial['coverage']:.1f}%, "
                  f"потенциал={initial['avg_potential']:.3f} В")
            print(f"    Конец (30 лет): coverage={final['coverage']:.1f}%, "
                  f"потенциал={final['avg_potential']:.3f} В")
            if 'coverage' in initial and 'coverage' in final:
                print(f"    Деградация coverage: {initial['coverage'] - final['coverage']:.1f}%")
        
        return (base_params, sequence_results, soil_model.get_state_dict())
    
    # def generate_case_with_advanced_physics(self, base_params, case_idx, time_points, 
    #                                        model_type='nonlinear'):
    #     """
    #     Генерация с разными типами физических моделей
        
    #     model_type:
    #     - 'linear': линейные граничные условия Дирихле (текущая)
    #     - 'mixed': смешанные граничные условия
    #     - 'nonlinear': нелинейные граничные условия
    #     """
    #     print(f"\nГенерация данных для случая {case_idx} ({model_type} модель)...")
        
    #     # Создаем или получаем модель грунта
    #     soil_key = tuple(base_params[:6])
        
    #     if soil_key not in self.soil_models_cache:
    #         soil_model = SoilModel(
    #             self.shared_domain, base_params, 
    #             domain_height=8.0, pipe_y=4.0
    #         )
    #         self.soil_models_cache[soil_key] = soil_model
    #     else:
    #         soil_model = self.soil_models_cache[soil_key]
        
    #     # Создаем симулятор
    #     simulator = CPS_DegradationSimulator()
    #     simulator.domain = self.shared_domain
    #     simulator.create_mesh_and_function_space()
        
    #     sequence_results = []
        
    #     for t in time_points:
    #         print(f"\n  Время: {t} лет")
            
    #         # Настраиваем проводимость
    #         simulator.setup_soil_model(base_params, t, soil_model)
            
    #         # Выбираем тип модели
    #         if model_type == 'linear':
    #             results = simulator.solve_full_physics_model(base_params, t, soil_model)
    #         elif model_type == 'mixed':
    #             results = simulator.solve_mixed_boundary_model(base_params, t, soil_model)
    #         elif model_type == 'nonlinear':
    #             results = simulator.solve_nonlinear_physics_model(base_params, t, soil_model)
    #         else:
    #             results = simulator.solve_full_physics_model(base_params, t, soil_model)
            
    #         sequence_results.append(results)
        
    #     return (base_params, sequence_results, soil_model.get_state_dict(), model_type)

    def save_dataset(self, all_results):
        """Сохранение датасета"""
        print(f"\n💾 Сохранение датасета...")
        
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        with h5py.File(self.data_file, 'w') as f:
            params_group = f.create_group('parameters')
            results_group = f.create_group('results')
            fields_group = f.create_group('fields')
            
            for case_idx, (base_params, time_sequence, _) in enumerate(all_results):
                params_group.create_dataset(f'case_{case_idx:04d}', 
                                          data=np.array(base_params, dtype=np.float32))
                
                case_group = results_group.create_group(f'case_{case_idx:04d}')
                case_fields_group = fields_group.create_group(f'case_{case_idx:04d}')
                
                for time_result in time_sequence:
                    t = int(time_result['time_years'])
                    
                    # Сохраняем результаты
                    time_results_group = case_group.create_group(f't_{t:03d}')
                    for key, value in time_result.items():
                        if key != 'time_years' and key != 'field_data':
                            time_results_group.attrs[key] = float(value)
                    
                    # Сохраняем поля
                    if 'field_data' in time_result:
                        field_data = time_result['field_data']
                        time_fields_group = case_fields_group.create_group(f't_{t:03d}')
                        
                        time_fields_group.create_dataset('X', data=field_data['X_grid'])
                        time_fields_group.create_dataset('Y', data=field_data['Y_grid'])
                        time_fields_group.create_dataset('phi', data=field_data['phi_grid'])
                        time_fields_group.create_dataset('sigma', data=field_data['sigma_grid'])
                        
                        for key in ['domain_width', 'domain_height', 'pipe_y', 
                                  'resolution_x', 'resolution_y', 'pipe_radius',
                                  'pipe_start', 'pipe_end']:
                            if key in field_data:
                                time_fields_group.attrs[key] = field_data[key]
                        time_fields_group.attrs['time_years'] = t
                        
                        # Сохраняем полное решение FEM (если есть)
                        if 'phi_solution' in field_data and field_data['phi_solution'] is not None:
                            time_fields_group.create_dataset('phi_fem', 
                                                            data=field_data['phi_solution'])
        
        # Метаданные
        metadata = {
            'total_cases': len(all_results),
            'time_points': [0, 5, 10, 15, 20, 25, 30],
            'generation_date': datetime.now().isoformat(),
            'model_type': 'Полноценная физическая модель FEM',
            'physics': 'Уравнение Лапласа с нелинейными граничными условиями',
            'soil_models_count': len(self.soil_models_cache),
            'note': 'Модели грунта сохранены отдельно в soil_models.json'
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Датсет сохранен: {self.data_file}")
        
        # Проверка
        print(f"\n🔍 ПРОВЕРКА ДАННЫХ:")
        with h5py.File(self.data_file, 'r') as f:
            for case_idx in range(len(all_results)):
                case_key = f'case_{case_idx:04d}'
                if case_key in f['fields']:
                    t_key = list(f['fields'][case_key].keys())[0]
                    phi_data = f['fields'][case_key][t_key]['phi'][:]
                    sigma_data = f['fields'][case_key][t_key]['sigma'][:]
                    
                    print(f"  Случай {case_idx} ({t_key}):")
                    print(f"    Потенциал: [{np.min(phi_data):.3f}, {np.max(phi_data):.3f}] В")
                    print(f"    Проводимость: [{np.min(sigma_data):.4f}, {np.max(sigma_data):.4f}] См/м")
    
    def save_soil_models(self):
        """Сохранение моделей грунта"""
        print(f"\n💾 Сохранение моделей грунта...")
        
        soil_models_data = {}
        for i, (soil_key, soil_model) in enumerate(self.soil_models_cache.items()):
            soil_models_data[f"soil_model_{i:04d}"] = {
                'params': soil_model.params,
                'seed': soil_model.seed,
                'base_factors_mean': float(np.mean(soil_model.base_factors)) if hasattr(soil_model, 'base_factors') else 0.0,
                'base_factors_std': float(np.std(soil_model.base_factors)) if hasattr(soil_model, 'base_factors') else 0.0,
                'key': list(soil_key)
            }
        
        with open(self.soil_models_file, 'w') as f:
            json.dump(soil_models_data, f, indent=2)
        
        print(f"  Сохранено {len(soil_models_data)} моделей грунта в {self.soil_models_file}")
    
    def generate_and_save(self, num_cases=3):
        """Генерация и сохранение датасета"""
        print(f"\n{'='*70}")
        print("ГЕНЕРАЦИЯ ДАТАСЕТА С УПРАВЛЕНИЕМ МОДЕЛЯМИ ГРУНТА")
        print(f"{'='*70}")
        
        time_points = [0, 5, 10, 15, 20, 25, 30]
        base_params_list = self.generate_base_parameters(num_cases)
        
        all_results = []
        
        for i, base_params in enumerate(base_params_list):
            scenario = ["Отличная защита", "Средняя защита", "Плохая защита"][i]
            
            print(f"\n{'='*50}")
            print(f"СЦЕНАРИЙ {i+1}: {scenario}")
            print(f"{'='*50}")
            
            # Генерируем данные с привязанной моделью грунта
            case_data = self.generate_case_with_soil_model(base_params, i, time_points)
            all_results.append(case_data)
        
        # Сохраняем датасет
        self.save_dataset(all_results)
        
        # Сохраняем модели грунта
        self.save_soil_models()
        
        print(f"\n{'='*70}")
        print("✅ ДАТАСЕТ УСПЕШНО СОЗДАН")
        print(f"{'='*70}")
        print(f"\nСоздано: {len(all_results)} случаев")
        print(f"Использовано уникальных моделей грунта: {len(self.soil_models_cache)}")
        print(f"\nФайлы:")
        print(f"  Основные данные: {self.data_file}")
        print(f"  Метаданные: {self.metadata_file}")
        print(f"  Модели грунта: {self.soil_models_file}")

    # def compare_physics_models(self):
    #     """Сравнение разных физических моделей"""
    #     print("="*70)
    #     print("СРАВНЕНИЕ ФИЗИЧЕСКИХ МОДЕЛЕЙ")
    #     print("="*70)
        
    #     from dolfinx import mesh
    #     from mpi4py import MPI
    #     import numpy as np
        
    #     # Создаем домен
    #     domain = mesh.create_rectangle(
    #         MPI.COMM_WORLD,
    #         [np.array([0.0, 0.0]), np.array([20.0, 8.0])],
    #         [40, 16],
    #         mesh.CellType.triangle
    #     )
        
    #     # Тестовые параметры
    #     test_params = [4.5, 0.2, 0.9, 7.5, 9.0, 0.5, 0.0, 0.92]
        
    #     # Создаем симуляторы
    #     sim_linear = CPS_DegradationSimulator()
    #     sim_mixed = CPS_DegradationSimulator()
    #     sim_nonlinear = CPS_DegradationSimulator()
        
    #     sim_linear.domain = domain
    #     sim_mixed.domain = domain
    #     sim_nonlinear.domain = domain
        
    #     sim_linear.create_mesh_and_function_space()
    #     sim_mixed.create_mesh_and_function_space()
    #     sim_nonlinear.create_mesh_and_function_space()
        
    #     # Модель грунта
    #     soil = SoilModel(domain, test_params, 8.0, 4.0)
        
    #     print(f"\nПараметры: V_app={test_params[4]} В, coating={test_params[2]}")
    #     print(f"Время: t=0 лет")
        
    #     # Настраиваем проводимость
    #     sigma_func = soil.get_conductivity(0)
        
    #     sim_linear.sigma.x.array[:] = sigma_func.x.array[:]
    #     sim_mixed.sigma.x.array[:] = sigma_func.x.array[:]
    #     sim_nonlinear.sigma.x.array[:] = sigma_func.x.array[:]
        
    #     # Решаем разными методами
    #     print(f"\n{'='*50}")
    #     print("1. ЛИНЕЙНАЯ МОДЕЛЬ (Дирихле)")
    #     print('='*50)
    #     results_linear = sim_linear.solve_full_physics_model(test_params, 0, soil)
        
    #     print(f"\n{'='*50}")
    #     print("2. СМЕШАННАЯ МОДЕЛЬ (Дирихле + Нейман)")
    #     print('='*50)
    #     results_mixed = sim_mixed.solve_mixed_boundary_model(test_params, 0, soil)
        
    #     print(f"\n{'='*50}")
    #     print("3. НЕЛИНЕЙНАЯ МОДЕЛЬ (итерационная)")
    #     print('='*50)
    #     results_nonlinear = sim_nonlinear.solve_nonlinear_physics_model(test_params, 0, soil)
        
    #     # Сравнение
    #     print(f"\n{'='*70}")
    #     print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    #     print('='*70)
        
    #     print(f"\n{'Модель':<20} {'Потенциал (В)':<15} {'Coverage (%)':<15} {'Ток трубы (А/м)':<15}")
    #     print(f"{'-'*70}")
        
    #     print(f"{'Линейная':<20} {results_linear['avg_potential']:<15.3f} "
    #         f"{results_linear['coverage']:<15.1f} {results_linear.get('pipe_current', 'N/A'):<15}")
        
    #     print(f"{'Смешанная':<20} {results_mixed['avg_potential']:<15.3f} "
    #         f"{results_mixed['coverage']:<15.1f} {results_mixed.get('pipe_current', 'N/A'):<15}")
        
    #     print(f"{'Нелинейная':<20} {results_nonlinear['avg_potential']:<15.3f} "
    #         f"{results_nonlinear['coverage']:<15.1f} {results_nonlinear.get('pipe_current', 'N/A'):<15}")
        
    #     # Визуализация
    #     try:
    #         import matplotlib.pyplot as plt
            
    #         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
    #         # Распределение потенциала для каждой модели
    #         models = [
    #             (sim_linear.phi.x.array, "Линейная"),
    #             (sim_mixed.phi.x.array, "Смешанная"),
    #             (sim_nonlinear.phi.x.array, "Нелинейная")
    #         ]
            
    #         dof_coords = sim_linear.V.tabulate_dof_coordinates()
            
    #         for idx, (phi_values, title) in enumerate(models[:3]):
    #             ax = axes[idx // 2, idx % 2]
    #             scatter = ax.scatter(dof_coords[:, 0], dof_coords[:, 1], 
    #                                 c=phi_values, cmap='coolwarm', s=10, alpha=0.8)
    #             ax.set_title(f'{title} модель')
    #             ax.set_xlabel('X (м)')
    #             ax.set_ylabel('Y (м)')
    #             ax.set_aspect('equal')
    #             plt.colorbar(scatter, ax=ax, label='Потенциал (В)')
            
    #         # График сравнения
    #         ax = axes[1, 1]
    #         models_names = ['Линейная', 'Смешанная', 'Нелинейная']
    #         potentials = [r['avg_potential'] for r in [results_linear, results_mixed, results_nonlinear]]
    #         coverages = [r['coverage'] for r in [results_linear, results_mixed, results_nonlinear]]
            
    #         x = np.arange(len(models_names))
    #         width = 0.35
            
    #         ax.bar(x - width/2, potentials, width, label='Потенциал (В)', color='skyblue')
    #         ax.bar(x + width/2, coverages, width, label='Coverage (%)', color='lightcoral')
            
    #         ax.set_xlabel('Модель')
    #         ax.set_ylabel('Значения')
    #         ax.set_title('Сравнение моделей')
    #         ax.set_xticks(x)
    #         ax.set_xticklabels(models_names)
    #         ax.legend()
            
    #         plt.tight_layout()
    #         plt.savefig('physics_models_comparison.png', dpi=150)
    #         plt.close()
            
    #         print(f"\nВизуализация сохранена в physics_models_comparison.png")
            
    #     except Exception as e:
    #         print(f"Визуализация не удалась: {e}")
        
    #     print(f"\n{'='*70}")
    #     print("СРАВНЕНИЕ ЗАВЕРШЕНО")
    #     print('='*70)

if __name__ == "__main__":
    generator = CPS_DatasetGenerator()
    generator.generate_and_save(num_cases=3)
    # generator.compare_physics_models()