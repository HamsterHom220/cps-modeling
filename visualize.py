import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime
import matplotlib.cm as cm

class CPS_Visualizer:
    """Визуализатор результатов моделирования катодной защиты"""
    
    def __init__(self, data_dir="./dataset"):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "cp_dataset.h5")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        
    def load_dataset(self):
        """Загрузка датасета"""
        print(f"📊 Загрузка датасета из {self.data_file}")
        
        with h5py.File(self.data_file, 'r') as f:
            # Загружаем параметры
            parameters = {}
            for case_name in f['parameters'].keys():
                case_idx = int(case_name.split('_')[1])
                parameters[case_idx] = f['parameters'][case_name][:]
            
            # Загружаем результаты
            results = {}
            for case_name in f['results'].keys():
                case_idx = int(case_name.split('_')[1])
                case_results = {}
                
                for time_name in f['results'][case_name].keys():
                    t = int(time_name.split('_')[1])
                    time_group = f['results'][case_name][time_name]
                    
                    # Читаем все атрибуты временной точки
                    time_results = {}
                    for key in time_group.attrs.keys():
                        time_results[key] = time_group.attrs[key]
                    time_results['time_years'] = t
                    
                    case_results[t] = time_results
                
                results[case_idx] = case_results
        
        # Загружаем метаданные
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return parameters, results, metadata
    
    def plot_case_timeline(self, case_idx=0, save_fig=False):
        """Визуализация временной последовательности для одного набора параметров"""
        parameters, results, metadata = self.load_dataset()
        
        if case_idx not in results:
            print(f"❌ Набор параметров {case_idx} не найден")
            return
        
        case_results = results[case_idx]
        time_points = sorted(case_results.keys())
        
        # Извлекаем данные для графика
        times = []
        coverages = []
        avg_potentials = []
        min_potentials = []
        max_potentials = []
        voltage_drops = []
        
        for t in time_points:
            res = case_results[t]
            times.append(t)
            coverages.append(res['coverage'])
            avg_potentials.append(res['avg_potential'])
            min_potentials.append(res['min_potential'])
            max_potentials.append(res['max_potential'])
            voltage_drops.append(res['voltage_drop'])
        
        # Получаем базовые параметры
        base_params = parameters[case_idx]
        param_names = [
            'R_sigma (Ω·m)',
            'pipe_roughness',
            'coating_quality',
            'soil_acidity',
            'V_app (V)',
            'wetness',
            'system_age',
            'anode_efficiency'
        ]
        
        # Создаем график
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Динамика параметров катодной защиты\nНабор параметров {case_idx}', 
                    fontsize=14, fontweight='bold')
        
        # Используем GridSpec для сложной компоновки
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. График coverage
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, coverages, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Время (годы)', fontsize=10)
        ax1.set_ylabel('Процент защищенных точек (%)', fontsize=10)
        ax1.set_title('Динамика защищенности трубы', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # Добавляем аннотации для начального и конечного значений
        ax1.annotate(f'Начало: {coverages[0]:.1f}%', 
                    xy=(times[0], coverages[0]), 
                    xytext=(5, 15),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
        
        if len(coverages) > 1:
            ax1.annotate(f'Конец: {coverages[-1]:.1f}%', 
                        xy=(times[-1], coverages[-1]), 
                        xytext=(-60, 15),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))
        
        # 2. График потенциалов
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, avg_potentials, 'go-', linewidth=2, markersize=8, label='Средний')
        ax2.fill_between(times, min_potentials, max_potentials, alpha=0.2, color='green', label='Диапазон')
        ax2.axhline(y=-0.85, color='r', linestyle='--', linewidth=1.5, label='Критерий защиты (-0.85 В)')
        ax2.set_xlabel('Время (годы)', fontsize=10)
        ax2.set_ylabel('Потенциал трубы (В)', fontsize=10)
        ax2.set_title('Динамика потенциала трубы', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.invert_yaxis()  # Более отрицательные значения выше на графике
        
        # 3. График падения напряжения
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times, voltage_drops, 'mo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Время (годы)', fontsize=10)
        ax3.set_ylabel('Падение напряжения (В)', fontsize=10)
        ax3.set_title('Падение напряжения анод-труба', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Гистограмма базовых параметров
        ax4 = fig.add_subplot(gs[1, :])
        bars = ax4.bar(range(len(param_names)), base_params, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(param_names))))
        ax4.set_xlabel('Параметр', fontsize=10)
        ax4.set_ylabel('Значение', fontsize=10)
        ax4.set_title('Базовые параметры системы', fontsize=11, fontweight='bold')
        ax4.set_xticks(range(len(param_names)))
        ax4.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for i, (bar, val) in enumerate(zip(bars, base_params)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Матрица корреляции между временем и ключевыми параметрами
        ax5 = fig.add_subplot(gs[2, 0])
        correlation_data = np.column_stack([
            times,
            coverages,
            avg_potentials,
            voltage_drops
        ])
        
        corr_matrix = np.corrcoef(correlation_data, rowvar=False)
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax5.set_title('Корреляционная матрица', fontsize=11, fontweight='bold')
        ax5.set_xticks(range(4))
        ax5.set_yticks(range(4))
        ax5.set_xticklabels(['Время', 'Coverage', 'Потенциал', 'ΔV'], fontsize=9, rotation=45)
        ax5.set_yticklabels(['Время', 'Coverage', 'Потенциал', 'ΔV'], fontsize=9)
        
        # Добавляем значения корреляции
        for i in range(4):
            for j in range(4):
                text = ax5.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        
        plt.colorbar(im, ax=ax5)
        
        # 6. Диаграмма рассеяния: Coverage vs Потенциал
        ax6 = fig.add_subplot(gs[2, 1])
        scatter = ax6.scatter(avg_potentials, coverages, c=times, 
                             cmap='plasma', s=100, alpha=0.7, edgecolors='black')
        ax6.set_xlabel('Средний потенциал (В)', fontsize=10)
        ax6.set_ylabel('Coverage (%)', fontsize=10)
        ax6.set_title('Coverage vs Потенциал', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.invert_xaxis()  # Более отрицательные значения правее
        
        # Добавляем цветовую шкалу для времени
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Время (годы)', fontsize=9)
        
        # 7. Информационная панель
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        if len(coverages) > 1:
            total_degradation = coverages[0] - coverages[-1]
            yearly_degradation = total_degradation / times[-1]
            final_coverage = coverages[-1]
            
            info_text = (
                f"СТАТИСТИКА НАБОРА ПАРАМЕТРОВ {case_idx}\n\n"
                f"Начальное coverage: {coverages[0]:.1f}%\n"
                f"Конечное coverage: {final_coverage:.1f}%\n"
                f"Общая деградация: {total_degradation:.1f}%\n"
                f"Скорость деградации: {yearly_degradation:.2f}%/год\n\n"
                f"Средний потенциал (начало): {avg_potentials[0]:.3f} В\n"
                f"Средний потенциал (конец): {avg_potentials[-1]:.3f} В\n"
                f"Изменение потенциала: {avg_potentials[-1] - avg_potentials[0]:.3f} В\n\n"
            )
            
            ax7.text(0.1, 0.95, info_text, transform=ax7.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=1", 
                            facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, f"case_{case_idx:04d}_timeline.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"💾 График сохранен: {fig_path}")
        
        plt.show()
    
    def plot_dataset_summary(self, max_cases=10, save_fig=False):
        """Сводная визуализация всего датасета"""
        parameters, results, metadata = self.load_dataset()
        
        case_indices = sorted(list(results.keys()))[:max_cases]
        n_cases = len(case_indices)
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Сводная статистика датасета катодной защиты\n'
                    f'{n_cases} наборов параметров из {metadata["total_cases"]}', 
                    fontsize=14, fontweight='bold')
        
        # Собираем данные по всем случаям
        all_initial_coverages = []
        all_final_coverages = []
        all_degradation_rates = []
        all_avg_potentials = []
        all_V_app = []
        
        for case_idx in case_indices:
            case_results = results[case_idx]
            times = sorted(case_results.keys())
            
            if len(times) >= 2:
                initial = case_results[times[0]]['coverage']
                final = case_results[times[-1]]['coverage']
                degradation = (initial - final) / times[-1] if final < initial else 0
                avg_potential = case_results[times[0]]['avg_potential']
                
                all_initial_coverages.append(initial)
                all_final_coverages.append(final)
                all_degradation_rates.append(degradation)
                all_avg_potentials.append(avg_potential)
                all_V_app.append(parameters[case_idx][4])  # V_app
        
        # 1. Гистограмма начального и конечного coverage
        ax1 = plt.subplot(2, 3, 1)
        width = 0.35
        x = np.arange(n_cases)
        ax1.bar(x - width/2, all_initial_coverages, width, 
                label='Начальное', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, all_final_coverages, width, 
                label='Конечное', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Набор параметров', fontsize=10)
        ax1.set_ylabel('Coverage (%)', fontsize=10)
        ax1.set_title('Начальное и конечное coverage', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'C{i}' for i in case_indices], fontsize=8)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 105])
        
        # 2. Гистограмма скорости деградации
        ax2 = plt.subplot(2, 3, 2)
        colors = ['green' if rate < 0.5 else 'orange' if rate < 1.5 else 'red' 
                 for rate in all_degradation_rates]
        ax2.bar(range(n_cases), all_degradation_rates, color=colors, alpha=0.7)
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Критический порог (1%/год)')
        ax2.set_xlabel('Набор параметров', fontsize=10)
        ax2.set_ylabel('Скорость деградации (%/год)', fontsize=10)
        ax2.set_title('Скорость деградации системы', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(n_cases))
        ax2.set_xticklabels([f'C{i}' for i in case_indices], fontsize=8)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Диаграмма рассеяния: V_app vs Начальный потенциал
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(all_V_app, all_avg_potentials, 
                             c=all_initial_coverages, s=100, 
                             cmap='viridis', alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Приложенное напряжение (V_app)', fontsize=10)
        ax3.set_ylabel('Начальный потенциал трубы (В)', fontsize=10)
        ax3.set_title('V_app vs Потенциал (цвет = начальное coverage)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()
        plt.colorbar(scatter, ax=ax3).set_label('Начальное coverage (%)', fontsize=9)
        
        # 4. Распределение конечного coverage
        ax4 = plt.subplot(2, 3, 4)
        bins = np.arange(0, 101, 10)
        ax4.hist(all_final_coverages, bins=bins, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Конечное coverage (%)', fontsize=10)
        ax4.set_ylabel('Количество случаев', fontsize=10)
        ax4.set_title('Распределение конечного coverage', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axvline(x=70, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Минимальный стандарт')
        ax4.legend(fontsize=9)
        
        # 5. Матрица корреляции между параметрами
        ax5 = plt.subplot(2, 3, 5)
        
        # Собираем параметры для корреляции
        param_data = []
        for case_idx in case_indices:
            params = parameters[case_idx]
            param_data.append([
                params[0],  # R_sigma
                params[2],  # coating_quality
                params[4],  # V_app
                params[7],  # anode_efficiency
                all_initial_coverages[case_idx],
                all_degradation_rates[case_idx]
            ])
        
        param_data = np.array(param_data)
        corr_matrix = np.corrcoef(param_data, rowvar=False)
        
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax5.set_title('Корреляция параметров', fontsize=11, fontweight='bold')
        labels = ['R_sigma', 'Покрытие', 'V_app', 'КПД анода', 'Coverage нач.', 'Деградация']
        ax5.set_xticks(range(len(labels)))
        ax5.set_yticks(range(len(labels)))
        ax5.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax5.set_yticklabels(labels, fontsize=8)
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax5.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))
        
        plt.colorbar(im, ax=ax5)
        
        # 6. Тепловая карта coverage по времени для всех случаев
        ax6 = plt.subplot(2, 3, 6)
        
        # Создаем матрицу coverage по времени
        time_points = sorted(results[case_indices[0]].keys())
        coverage_matrix = np.zeros((n_cases, len(time_points)))
        
        for i, case_idx in enumerate(case_indices):
            case_results = results[case_idx]
            for j, t in enumerate(time_points):
                coverage_matrix[i, j] = case_results[t]['coverage']
        
        im = ax6.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', 
                       vmin=50, vmax=100)
        ax6.set_xlabel('Время (годы)', fontsize=10)
        ax6.set_ylabel('Набор параметров', fontsize=10)
        ax6.set_title('Динамика coverage для всех случаев', fontsize=11, fontweight='bold')
        ax6.set_xticks(range(len(time_points)))
        ax6.set_xticklabels(time_points, fontsize=8)
        ax6.set_yticks(range(n_cases))
        ax6.set_yticklabels([f'C{i}' for i in case_indices], fontsize=8)
        
        plt.colorbar(im, ax=ax6).set_label('Coverage (%)', fontsize=9)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, "dataset_summary.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"💾 Сводный график сохранен: {fig_path}")
        
        plt.show()
        
        # Выводим статистику
        print(f"\n📊 СТАТИСТИКА ДАТАСЕТА:")
        print(f"   Всего случаев: {metadata['total_cases']}")
        print(f"   Временные точки: {metadata['time_points']}")
        print(f"   Дата генерации: {metadata['generation_date']}")
        print(f"\n📈 СТАТИСТИКА ПО {n_cases} СЛУЧАЯМ:")
        print(f"   Среднее начальное coverage: {np.mean(all_initial_coverages):.1f}%")
        print(f"   Среднее конечное coverage: {np.mean(all_final_coverages):.1f}%")
        print(f"   Средняя скорость деградации: {np.mean(all_degradation_rates):.2f}%/год")
        print(f"   Минимальное конечное coverage: {np.min(all_final_coverages):.1f}%")
        print(f"   Максимальное конечное coverage: {np.max(all_final_coverages):.1f}%")
        
        # Анализ рисков
        critical_cases = sum(1 for cov in all_final_coverages if cov < 70)
        print(f"\n⚠️  АНАЛИЗ РИСКОВ:")
        print(f"   Критических случаев (coverage < 70%): {critical_cases} из {n_cases} ({critical_cases/n_cases*100:.1f}%)")
        print(f"   Случаев с деградацией > 1%/год: {sum(1 for rate in all_degradation_rates if rate > 1)}")
    
    def plot_parameter_distributions(self, save_fig=False):
        """Визуализация распределений параметров"""
        parameters, _, metadata = self.load_dataset()
        
        param_names = [
            'R_sigma (Ω·m)',
            'pipe_roughness',
            'coating_quality',
            'soil_acidity',
            'V_app (V)',
            'wetness',
            'anode_efficiency'
        ]
        
        # Собираем данные по параметрам (исключаем system_age)
        param_data = []
        for case_idx in parameters:
            params = parameters[case_idx]
            param_data.append(params[:6] + [params[7]])  # Исключаем system_age
        
        param_data = np.array(param_data)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Распределение параметров в датасете', fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, (ax, name, data) in enumerate(zip(axes, param_names, param_data.T)):
            if i >= len(param_names):
                ax.axis('off')
                continue
                
            # Гистограмма
            n, bins, patches = ax.hist(data, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Линия плотности
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 200)
            ax.plot(x_range, kde(x_range) * len(data) * (bins[1] - bins[0]), 
                   'r-', linewidth=2, alpha=0.8)
            
            # Статистика
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Среднее: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Медиана: {median_val:.2f}')
            
            ax.set_xlabel(name, fontsize=10)
            ax.set_ylabel('Частота', fontsize=10)
            ax.set_title(f'{name}\nσ={std_val:.2f}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Добавляем информацию о диапазоне
            ax.text(0.02, 0.98, f'Min: {min(data):.2f}\nMax: {max(data):.2f}',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Удаляем лишние оси
        for i in range(len(param_names), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, "parameter_distributions.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"💾 График распределений сохранен: {fig_path}")
        
        plt.show()
    
    def plot_degradation_trajectories(self, num_cases=5, save_fig=False):
        """Визуализация траекторий деградации для нескольких случаев"""
        parameters, results, _ = self.load_dataset()
        
        case_indices = sorted(list(results.keys()))[:num_cases]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.plasma(np.linspace(0, 1, num_cases))
        
        # График траекторий coverage
        for i, case_idx in enumerate(case_indices):
            case_results = results[case_idx]
            times = sorted(case_results.keys())
            coverages = [case_results[t]['coverage'] for t in times]
            
            ax1.plot(times, coverages, 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=f'Случай {case_idx}')
            
            # Добавляем конечную точку
            ax1.annotate(f'{coverages[-1]:.0f}%', 
                        xy=(times[-1], coverages[-1]),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        
        ax1.axhline(y=70, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Критический порог (70%)')
        ax1.set_xlabel('Время (годы)', fontsize=11)
        ax1.set_ylabel('Coverage (%)', fontsize=11)
        ax1.set_title('Траектории деградации покрытия', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        ax1.set_ylim([0, 105])
        
        # График траекторий потенциала
        for i, case_idx in enumerate(case_indices):
            case_results = results[case_idx]
            times = sorted(case_results.keys())
            potentials = [case_results[t]['avg_potential'] for t in times]
            
            ax2.plot(times, potentials, 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=f'Случай {case_idx}')
        
        ax2.axhline(y=-0.85, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Критерий защиты (-0.85 В)')
        ax2.set_xlabel('Время (годы)', fontsize=11)
        ax2.set_ylabel('Потенциал трубы (В)', fontsize=11)
        ax2.set_title('Траектории изменения потенциала', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, "degradation_trajectories.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"💾 График траекторий сохранен: {fig_path}")
        
        plt.show()


if __name__ == "__main__":
    visualizer = CPS_Visualizer()
    parameters, results, metadata = visualizer.load_dataset()
    visualizer.plot_case_timeline(case_idx=0, save_fig=True)
    visualizer.plot_dataset_summary(max_cases=min(10, len(parameters)), save_fig=True)
    visualizer.plot_parameter_distributions(save_fig=True)
    visualizer.plot_degradation_trajectories(num_cases=5, save_fig=True)