import numpy as np
import json
import os
import h5py
import signal
import sys
from datetime import datetime
from dolfinx import mesh, fem
from mpi4py import MPI
from tqdm import tqdm
from scipy.interpolate import griddata

from simulator import CPS_DegradationSimulator
from soil import SoilModel

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global _shutdown_requested
    if not _shutdown_requested:
        print("\n\n⚠️  Interrupt received. Finishing current case and saving...")
        print("    Press Ctrl+C again to force quit (may lose current case)")
        _shutdown_requested = True
    else:
        print("\n❌ Force quit. Current case may be incomplete.")
        sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

import shutil

# Before starting large generation
cache_dir = os.path.expanduser("~/.cache/fenics")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)


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

        # Create shared function space ONCE (reuse across all cases to prevent memory leak)
        print("Creating shared function space...")
        self.shared_V = fem.functionspace(self.shared_domain, ("Lagrange", 1))

        # Create a temporary simulator to generate shared facet markers
        print("Creating shared boundary markers...")
        from simulator import CPS_DegradationSimulator
        temp_sim = CPS_DegradationSimulator(verbose=False)
        temp_sim.domain = self.shared_domain
        temp_sim.V = self.shared_V
        temp_sim.mark_boundaries()  # Create facet markers once
        self.shared_facet_markers = temp_sim.facet_markers
        del temp_sim

        # Create shared solution functions (reuse across all cases)
        print("Creating shared solution functions...")
        self.shared_phi = fem.Function(self.shared_V, name="Potential")
        self.shared_sigma = fem.Function(self.shared_V, name="Conductivity")

        # Кэш для моделей грунта
        self.soil_models_cache = {}
    
    def generate_base_parameters(self, num_cases):
        """Генерация базовых параметров для случаев"""
        np.random.seed(42)
        
        combinations = []
        for i in range(num_cases):
            params = [
                np.random.uniform(3, 8),  # R_sigma
                np.random.uniform(0.1, 0.5),  # Шероховатость
                np.random.uniform(0.7, 0.95),  # Качество покрытия
                np.random.uniform(6.5, 8),  # Кислотность
                np.random.uniform(3, 7),  # V_app
                np.random.uniform(0.3, 0.8),  # Влажность
                0.0,  # Возраст (начало отсчета)
                np.random.uniform(0.9, 0.95)  # КПД анода
            ]
            combinations.append(params)
        
        return combinations
    
    def _generate_field_data(self, simulator, t_years):
        """Генерация данных полей для визуализации"""
        try:
            # Получаем данные с разрешением для визуализации
            resolution_x, resolution_y = 40, 20
            x_coords = np.linspace(0, simulator.domain_width, resolution_x)
            y_coords = np.linspace(0, simulator.domain_height, resolution_y)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            phi_grid = np.zeros_like(X)
            sigma_grid = np.zeros_like(X)
            
            # Получаем DOF координаты для интерполяции
            dof_coords = simulator.V.tabulate_dof_coordinates()
            phi_values = simulator.phi.x.array
            sigma_values = simulator.sigma.x.array

            # Vectorized linear interpolation (5-10x faster than nearest-neighbor loops)
            points = dof_coords[:, :2]  # (x, y) coordinates
            phi_grid = griddata(points, phi_values, (X, Y), method='linear', fill_value=np.nan)
            sigma_grid = griddata(points, sigma_values, (X, Y), method='linear', fill_value=np.nan)

            # Fill any NaN values with nearest neighbor as fallback
            if np.any(np.isnan(phi_grid)):
                mask = np.isnan(phi_grid)
                phi_grid[mask] = griddata(points, phi_values, (X[mask], Y[mask]), method='nearest')
            if np.any(np.isnan(sigma_grid)):
                mask = np.isnan(sigma_grid)
                sigma_grid[mask] = griddata(points, sigma_values, (X[mask], Y[mask]), method='nearest')
            
            # Параметры трубы
            pipe_start, pipe_end, pipe_y, pipe_radius = simulator.pipe.get_pipe_segment(simulator.domain_width)
            
            field_data = {
                'X_grid': X,
                'Y_grid': Y,
                'phi_grid': phi_grid,
                'sigma_grid': sigma_grid,
                'domain_width': simulator.domain_width,
                'domain_height': simulator.domain_height,
                'pipe_y': pipe_y,
                'pipe_radius': pipe_radius,
                'pipe_start': float(pipe_start),
                'pipe_end': float(pipe_end),
                'resolution_x': resolution_x,
                'resolution_y': resolution_y,
                'time_years': t_years
            }
            
            return field_data
            
        except Exception as e:
            print(f"      ⚠️  Ошибка при генерации данных полей: {e}")
            # Возвращаем минимальные данные
            return {
                'X_grid': np.zeros((1, 1)),
                'Y_grid': np.zeros((1, 1)),
                'phi_grid': np.zeros((1, 1)),
                'sigma_grid': np.zeros((1, 1)),
                'domain_width': 20.0,
                'domain_height': 8.0,
                'pipe_y': 4.0,
                'pipe_radius': 0.1,
                'pipe_start': 5.0,
                'pipe_end': 15.0,
                'resolution_x': 1,
                'resolution_y': 1,
                'time_years': t_years
            }

    def generate_case_with_soil_model(self, base_params, case_idx, time_points, verbose=False):
        """Генерация данных для одного случая с привязанной моделью грунта"""
        if verbose:
            print(f"\nГенерация данных для случая {case_idx}...")
            print(f"  V_app: {base_params[4]:.1f} В, Покрытие: {base_params[2]:.2f}")
        
        # Create fresh soil model for each case (no caching with random parameters)
        # Caching disabled: with random params, reuse rate is 0% and cache just leaks memory
        if verbose:
            print(f"  Создание модели грунта для случая {case_idx}...")
        soil_model = SoilModel(
            self.shared_domain, base_params,
            domain_height=8.0, pipe_y=4.0,
            enable_plotting=False,
            function_space=self.shared_V  # Reuse shared function space (prevents leak)
        )

        # Store metadata for later saving (but not the full model object)
        soil_key = tuple(base_params[:6])
        if soil_key not in self.soil_models_cache:
            self.soil_models_cache[soil_key] = {
                'params': base_params,
                'seed': soil_model.seed,
                'case_idx': case_idx
            }
        
        # Создаем симулятор для этого случая
        simulator = CPS_DegradationSimulator(verbose=False)
        simulator.domain = self.shared_domain  # Используем общий домен
        simulator.V = self.shared_V  # Reuse shared function space (prevents memory leak)
        simulator.facet_markers = self.shared_facet_markers  # Reuse shared boundary markers

        # Reuse shared solution functions (just update arrays, don't create new objects)
        simulator.phi = self.shared_phi
        simulator.sigma = self.shared_sigma

        # Reset solution arrays for this case
        simulator.phi.x.array[:] = 0.0
        simulator.sigma.x.array[:] = 0.0
        
        sequence_results = []
        
        # Кэшируем проводимость для каждого времени (will be cleaned up at end)
        conductivity_cache = {}

        try:
            for t in time_points:
                if verbose:
                    print(f"\n  Время: {t} лет")

                # Получаем проводимость (вычисляем только один раз)
                if t not in conductivity_cache:
                    sigma_func = soil_model.get_conductivity(t)
                    conductivity_cache[t] = sigma_func.x.array.copy()

                # Устанавливаем проводимость в симулятор
                simulator.sigma.x.array[:] = conductivity_cache[t]

                # Проверка проводимости
                if verbose:
                    sigma_values = simulator.sigma.x.array
                    print(f"    Проводимость: min={np.min(sigma_values):.4f}, "
                        f"max={np.max(sigma_values):.4f} S/m")

                # Solve with Robin BC (includes degradation internally)
                results = simulator.solve_with_robin_bc(base_params, t, soil_model)

                # Добавляем данные полей для визуализации
                if results is not None:
                    results['field_data'] = self._generate_field_data(simulator, t)
                    results['time_years'] = t
                    sequence_results.append(results)
        finally:
            # Cleanup memory for this case
            del conductivity_cache
            del simulator
            del soil_model  # Free soil model after use (no longer cached)

        # Сохраняем сводку
        if sequence_results and verbose:
            initial = sequence_results[0]
            final = sequence_results[-1]
            print(f"\n  📊 ИТОГИ для случая {case_idx}:")
            print(f"    Начало (0 лет): coverage={initial['coverage']:.1f}%, "
                f"потенциал={initial['avg_potential']:.3f} В")
            print(f"    Конец (30 лет): coverage={final['coverage']:.1f}%, "
                f"потенциал={final['avg_potential']:.3f} В")
            if 'coverage' in initial and 'coverage' in final:
                print(f"    Деградация coverage: {initial['coverage'] - final['coverage']:.1f}%")

        # Note: soil_model has been deleted in finally block (no longer cached)
        return (base_params, sequence_results, None)

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
        
    def save_soil_models(self):
        """Сохранение моделей грунта (metadata only, models not cached)"""
        print(f"\n💾 Сохранение метаданных моделей грунта...")

        soil_models_data = {}
        for i, (soil_key, metadata) in enumerate(self.soil_models_cache.items()):
            soil_models_data[f"soil_model_{i:04d}"] = {
                'params': metadata['params'],
                'seed': metadata['seed'],
                'case_idx': metadata['case_idx'],
                'key': list(soil_key)
            }
        
        with open(self.soil_models_file, 'w') as f:
            json.dump(soil_models_data, f, indent=2)
        
        print(f"  Сохранено {len(soil_models_data)} моделей грунта в {self.soil_models_file}")
    
    def get_completed_cases(self):
        """
        Check which cases are already completed in the HDF5 file.

        Returns:
            set: Set of completed case indices
        """
        if not os.path.exists(self.data_file):
            return set()

        completed = set()
        try:
            with h5py.File(self.data_file, 'r') as f:
                if 'parameters' in f:
                    for case_name in f['parameters'].keys():
                        # Extract case index from 'case_0000' format
                        case_idx = int(case_name.split('_')[1])
                        completed.add(case_idx)
        except Exception as e:
            print(f"Warning: Error reading HDF5 file for resume detection: {e}")
            return set()

        return completed

    def save_case_incremental(self, case_idx, base_params, time_sequence):
        """
        Incremental saving of a single case to HDF5 (memory-efficient).

        If the case already exists, it will be overwritten (handles resume after partial save).
        """
        with h5py.File(self.data_file, 'a') as f:
            # Ensure groups exist
            if 'parameters' not in f:
                f.create_group('parameters')
            if 'results' not in f:
                f.create_group('results')
            if 'fields' not in f:
                f.create_group('fields')

            params_group = f['parameters']
            results_group = f['results']
            fields_group = f['fields']

            case_name = f'case_{case_idx:04d}'

            # Remove existing case if present (handles resume after interruption)
            if case_name in params_group:
                del params_group[case_name]
            if case_name in results_group:
                del results_group[case_name]
            if case_name in fields_group:
                del fields_group[case_name]

            # Save parameters
            params_group.create_dataset(case_name, data=np.array(base_params, dtype=np.float32))

            case_group = results_group.create_group(case_name)
            case_fields_group = fields_group.create_group(case_name)

            for time_result in time_sequence:
                t = int(time_result['time_years'])

                # Save results
                time_results_group = case_group.create_group(f't_{t:03d}')
                for key, value in time_result.items():
                    if key != 'time_years' and key != 'field_data':
                        time_results_group.attrs[key] = float(value)

                # Save fields
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

            # Force flush to disk (helps reduce memory buffering)
            f.flush()

    def generate_and_save(self, num_cases=3, batch_size=200, start_from=None, auto_resume=True):
        """
        Генерация и сохранение датасета (memory-efficient incremental saving)

        Args:
            num_cases: Total number of cases to generate
            batch_size: Max cases per batch (restart needed after this to prevent OOM)
            start_from: Case index to start from (None = auto-detect from HDF5 file)
            auto_resume: If True, automatically skip completed cases
        """
        global _shutdown_requested
        _shutdown_requested = False

        print(f"\n{'='*70}")
        print("ГЕНЕРАЦИЯ ДАТАСЕТА С УПРАВЛЕНИЕМ МОДЕЛЯМИ ГРУНТА")
        print(f"{'='*70}")

        time_points = [0, 5, 10, 15, 20, 25, 30]
        base_params_list = self.generate_base_parameters(num_cases)

        # Auto-detect resume point if not specified
        completed_cases = self.get_completed_cases() if auto_resume else set()

        if start_from is None:
            if completed_cases:
                start_from = max(completed_cases) + 1
                print(f"📁 Found existing dataset with {len(completed_cases)} cases")
                print(f"   Auto-resuming from case {start_from}")
            else:
                start_from = 0
                print("   Starting fresh dataset")

        if start_from == 0 and not completed_cases and os.path.exists(self.data_file):
            print("   Removing old dataset file")
            os.remove(self.data_file)

        # Determine end of this batch
        end_idx = min(start_from + batch_size, num_cases)

        print(f"Total cases: {num_cases}, Batch size: {batch_size}")
        print(f"Processing cases {start_from} to {end_idx-1} (batch of {end_idx-start_from})")
        if completed_cases:
            print(f"Skipping {len([i for i in range(start_from, end_idx) if i in completed_cases])} already completed cases in this batch")
        print()

        # Generate and save incrementally (no memory accumulation)
        cases_generated = 0
        for i in tqdm(range(start_from, end_idx), desc="Generating cases"):
            # Check for graceful shutdown
            if _shutdown_requested:
                print(f"\n\n💾 Graceful shutdown at case {i}")
                print(f"   Generated {cases_generated} new cases in this session")
                print(f"   Total cases in dataset: {len(self.get_completed_cases())}")
                print(f"\n   To resume, run:")
                print(f"   generator.generate_and_save(num_cases={num_cases}, batch_size={batch_size})")
                return

            # Skip if already completed
            if auto_resume and i in completed_cases:
                continue

            base_params = base_params_list[i]

            # Generate data for this case
            case_data = self.generate_case_with_soil_model(base_params, i, time_points)
            base_params_result, time_sequence, _ = case_data

            # Save immediately (incremental)
            self.save_case_incremental(i, base_params_result, time_sequence)
            cases_generated += 1

            # Explicitly free memory for this case
            del case_data, time_sequence

        print(f"\n✅ Batch complete! Generated {cases_generated} new cases")

        # Check if more batches needed
        if end_idx < num_cases:
            print(f"\n⚠️  Batch complete. {num_cases - end_idx} cases remaining.")
            print(f"    To continue, restart Python and run:")
            print(f"    generator.generate_and_save(num_cases={num_cases}, batch_size={batch_size}, start_from={end_idx})")
            return  # Don't save metadata yet

        # Save metadata
        metadata = {
            'total_cases': num_cases,
            'time_points': time_points,
            'generation_date': datetime.now().isoformat(),
            'model_type': 'Полноценная физическая модель FEM',
            'physics': 'Уравнение Лапласа с нелинейными граничными условиями',
            'soil_models_count': len(self.soil_models_cache),
            'note': 'Модели грунта сохранены отдельно в soil_models.json'
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save soil models
        self.save_soil_models()

        print(f"\n{'='*70}")
        print("✅ ДАТАСЕТ УСПЕШНО СОЗДАН")
        print(f"{'='*70}")
        print(f"\nСоздано: {num_cases} случаев")
        print(f"Использовано уникальных моделей грунта: {len(self.soil_models_cache)}")
        print(f"\nФайлы:")
        print(f"  Основные данные: {self.data_file}")
        print(f"  Метаданные: {self.metadata_file}")
        print(f"  Модели грунта: {self.soil_models_file}")


if __name__ == "__main__":
    generator = CPS_DatasetGenerator()

    # Generate in batches of 200 to avoid OOM (30 MB leak per case)
    # Auto-resumes from where it left off (checks HDF5 file)
    # Use Ctrl+C to interrupt gracefully (waits for current case to finish)
    generator.generate_and_save(num_cases=1000, batch_size=200)
