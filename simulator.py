import numpy as np
from dolfinx import mesh, fem, default_scalar_type
from mpi4py import MPI
import ufl
from dolfinx.fem import form
from dolfinx.fem.petsc import PETSc

from soil import SoilModel
from pipe import Pipe
from eage import ExtendedAnode


class CPS_DegradationSimulator:
    def __init__(self, domain_width=20.0, domain_height=8.0, mesh_resolution=(80, 32), verbose=False):
        self.domain_width = domain_width
        self.domain_height = domain_height
        self.mesh_resolution = mesh_resolution
        self.verbose = verbose
        
        # Инициализация компонентов системы
        self.pipe = Pipe(y_position=4.0)
        self.anode = ExtendedAnode(y_position=1.5)
        
        # Физические параметры
        self.time_years = 0.0
        
        # FEM-объекты
        self.domain = None
        self.V = None
        self.phi = None
        self.sigma_function = None

    def create_mesh_and_function_space(self):
        """Создание сетки и функционального пространства"""
        
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([self.domain_width, self.domain_height])],
            self.mesh_resolution,
            mesh.CellType.triangle
        )
        
        # Создание функционального пространства для потенциала
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.phi = fem.Function(self.V, name="Potential")
        
        # Функция для проводимости грунта
        self.sigma = fem.Function(self.V, name="Conductivity")
        
        return self.domain, self.V
    
    def setup_soil_model(self, base_params, t_years, soil_model):
        """Настройка с существующей моделью грунта - ПОПРАВЛЕННАЯ"""
        # Получаем проводимость из модели грунта
        self.sigma_function = soil_model.get_conductivity(t_years)

        # ВАЖНО: копируем значения в self.sigma, который используется в FEM решателях
        if self.sigma is not None:
            self.sigma.x.array[:] = self.sigma_function.x.array[:]

        # Проверка
        if self.verbose:
            sigma_values = self.sigma.x.array
            print(f"      Проводимость: min={np.min(sigma_values):.4f}, "
                  f"max={np.max(sigma_values):.4f} S/m")

        return soil_model

    def _calculate_anode_resistance(self, params, T, humidity):
        """Расчет сопротивления анода - делегирует в класс Anode"""
        return self.anode.calculate_anode_resistance(T=T, humidity=humidity)

    def _calculate_pipe_resistance(self, params, T, humidity):
        """Расчет сопротивления трубы - делегирует в класс Pipe"""
        return self.pipe.calculate_effective_resistance(T=T, humidity=humidity)

    def _get_boundary_potentials(self):
        """Получение средних потенциалов на границах"""
        anode_dofs = self._get_boundary_dofs('anode')
        pipe_dofs = self._get_boundary_dofs('pipe')

        if len(anode_dofs) > 0:
            anode_potential = np.mean(self.phi.x.array[anode_dofs])
        else:
            anode_potential = 0.0

        if len(pipe_dofs) > 0:
            pipe_potential = self.phi.x.array[pipe_dofs]
        else:
            pipe_potential = np.array([-0.85])

        return anode_potential, pipe_potential

    def _get_average_potential_on_boundary(self, boundary_type):
        """Get average potential on anode or pipe boundary."""
        dofs = self._get_boundary_dofs(boundary_type)
        if len(dofs) > 0:
            return np.mean(self.phi.x.array[dofs])
        return 1.2 if boundary_type == 'anode' else -0.85

    def _calculate_boundary_currents(self):
        """Расчет токов на границах"""
        anode_dofs = self._get_boundary_dofs('anode')
        pipe_dofs = self._get_boundary_dofs('pipe')

        # Простая оценка тока на основе градиента потенциала
        if len(anode_dofs) > 0 and len(pipe_dofs) > 0:
            phi_anode = np.mean(self.phi.x.array[anode_dofs])
            phi_pipe = np.mean(self.phi.x.array[pipe_dofs])
            delta_phi = phi_anode - phi_pipe

            R_anode = self.anode.calculate_anode_resistance()
            R_pipe = self.pipe.calculate_effective_resistance()
            R_total = R_anode + R_pipe

            if R_total > 0:
                current = delta_phi / R_total
            else:
                current = 0.0

            anode_current = current * self.anode.get_area()
            pipe_current = -current * self.pipe.get_area(self.domain_width)
        else:
            anode_current = 0.0
            pipe_current = 0.0

        return anode_current, pipe_current

    def _calculate_protection_coverage(self, pipe_potential):
        """Расчет покрытия защиты"""
        if isinstance(pipe_potential, np.ndarray):
            protected = pipe_potential <= -0.85
            coverage = 100.0 * np.sum(protected) / len(pipe_potential)
        else:
            coverage = 100.0 if pipe_potential <= -0.85 else 0.0
        return coverage

    def _estimate_corrosion_rate(self, pipe_potential, current_density, T):
        """Оценка скорости коррозии"""
        if isinstance(pipe_potential, np.ndarray):
            avg_potential = np.mean(pipe_potential)
        else:
            avg_potential = pipe_potential

        return self._estimate_corrosion_rate_from_potential(avg_potential, current_density, T)

    def _calculate_anode_current(self, J):
        """Расчет тока на аноде"""
        anode_y = self.anode.y_position
        dof_coords = self.V.tabulate_dof_coordinates()
        anode_dofs = []

        for i in range(dof_coords.shape[0]):
            x, y = dof_coords[i, 0], dof_coords[i, 1]
            if (9.0 <= x <= 11.0) and (anode_y - 0.5 <= y <= anode_y + 0.5):
                anode_dofs.append(i)

        if len(anode_dofs) > 0 and hasattr(J.x, 'array') and J.x.array.ndim > 1:
            avg_current = np.mean(np.abs(J.x.array[anode_dofs, 1]))
            return avg_current * 2.0  # Ширина анода
        return 0.0

    def mark_boundaries(self):
        """Разметка границ расчетной области и создание meshtags"""
        
        # Создаем связь граней с ячейками (нужно для exterior_facet_indices)
        self.domain.topology.create_connectivity(self.domain.topology.dim-1, self.domain.topology.dim)
        
        # Находим все граничные грани
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        
        # Создаем массив меток
        facet_markers = np.zeros(len(boundary_facets), dtype=np.int32)
        
        # Получаем координаты центров граней
        boundary_facet_centroids = mesh.compute_midpoints(self.domain, 
                                                         self.domain.topology.dim-1, 
                                                         boundary_facets)
        
        for i, centroid in enumerate(boundary_facet_centroids):
            x, y = centroid[0], centroid[1]

            # Проверяем, к какой границе принадлежит (pipe и anode имеют приоритет)
            if self._is_on_pipe(np.array([[x], [y]]))[0]:
                facet_markers[i] = 5  # Труба (on TOP boundary)
            elif self._is_on_anode(np.array([[x], [y]]))[0]:
                facet_markers[i] = 6  # Анод (on BOTTOM boundary)
            elif np.isclose(x, 0.0):
                facet_markers[i] = 1  # Левая граница
            elif np.isclose(x, self.domain_width):
                facet_markers[i] = 2  # Правая граница
            elif np.isclose(y, 0.0):
                facet_markers[i] = 3  # Нижняя граница (excluding anode)
            elif np.isclose(y, self.domain_height):
                facet_markers[i] = 4  # Верхняя граница (excluding pipe)
        
        # Создаем meshtags
        self.facet_markers = mesh.meshtags(self.domain, 
                                          self.domain.topology.dim-1, 
                                          boundary_facets, 
                                          facet_markers)
        
        return self.facet_markers
    
    def _is_on_pipe(self, x):
        """Проверка, находятся ли точки на поверхности трубы.

        For 2D FEM with exterior BCs only, the pipe is represented as a
        segment on the TOP boundary (y=domain_height) in the x range [5, 15].
        """
        pipe_start, pipe_end, _, _ = self.pipe.get_pipe_segment(self.domain_width)

        # x имеет размерность (2, n_points)
        # Pipe is on TOP boundary (y=domain_height), x in [pipe_start, pipe_end]
        in_x_range = np.logical_and(x[0] >= pipe_start, x[0] <= pipe_end)
        on_top_boundary = np.abs(x[1] - self.domain_height) < 0.1

        return np.logical_and(in_x_range, on_top_boundary)
    
    def _is_on_anode(self, x):
        """Проверка, находятся ли точки на поверхности анода.

        For 2D FEM with exterior BCs only, the anode is represented as a
        segment on the BOTTOM boundary (y=0) in the x range [9, 11].
        """
        anode_x_min, anode_x_max, _, _ = self.anode.get_anode_region(self.domain_width)

        # x имеет размерность (2, n_points)
        # Anode is on BOTTOM boundary (y=0), x in [anode_x_min, anode_x_max]
        in_anode_x = np.logical_and(x[0] >= anode_x_min, x[0] <= anode_x_max)
        on_bottom_boundary = np.abs(x[1]) < 0.1

        return np.logical_and(in_anode_x, on_bottom_boundary)

    def solve_with_robin_bc(self, base_params, t_years, soil_model=None):
        """
        Unified solver with Robin boundary conditions and proper Newton iteration.

        This is the authoritative solver that:
        1. Uses Robin BC (computes pipe potential) instead of Dirichlet (prescribes it)
        2. Implements proper Newton iteration (12 max, tol=1e-5, damping=0.8)
        3. Returns physically meaningful coverage metrics

        Args:
            base_params: Base system parameters [R_sigma, roughness, coating_quality, pH, V_app, humidity, age, anode_eff]
            t_years: Time in years
            soil_model: SoilModel instance (if None, creates new one)

        Returns:
            dict: Results including coverage, potentials, currents, etc.
        """
        # 1. SETUP DOMAIN AND MESH
        if self.domain is None:
            self.create_mesh_and_function_space()

        # 2. APPLY DEGRADATION
        degraded_params = self.apply_degradation(base_params, t_years)

        # 3. SETUP SOIL MODEL
        if soil_model is not None:
            self.setup_soil_model(degraded_params, t_years, soil_model)
        else:
            # Create temporary soil model if not provided
            from soil import SoilModel
            temp_soil_model = SoilModel(
                self.domain, degraded_params, self.domain_height, self.pipe.y_position,
                enable_plotting=False
            )
            self.setup_soil_model(degraded_params, t_years, temp_soil_model)

        # 4. CREATE BOUNDARY MARKERS (for Robin BC)
        if not hasattr(self, 'facet_markers') or self.facet_markers is None:
            self.mark_boundaries()

        # Physical parameters
        T = 15.0  # Temperature [°C]
        humidity = degraded_params[5] if len(degraded_params) > 5 else 0.8

        # Electrochemical equilibrium potentials
        E_anode = 1.2   # Mg anode equilibrium [V]
        E_pipe = -0.65  # Steel equilibrium [V]

        # 5. INITIAL GUESS
        self.phi.x.array[:] = -0.5  # Start with intermediate value

        # Get initial boundary potentials
        phi_anode_avg = self._get_average_potential_on_boundary('anode')
        phi_pipe_avg = self._get_average_potential_on_boundary('pipe')

        # 6. NEWTON ITERATION WITH ROBIN BC
        max_iterations = 12
        tolerance = 1e-5
        damping = 0.8
        converged = False

        for iteration in range(max_iterations):
            # Get potential-dependent resistances from Pipe and Anode classes
            R_anode = self.anode.calculate_anode_resistance(phi_surface=phi_anode_avg, T=T, humidity=humidity)
            R_pipe = self.pipe.calculate_effective_resistance(phi_surface=phi_pipe_avg, T=T, humidity=humidity)

            # Ensure minimum resistances to avoid division by zero
            R_anode = max(R_anode, 0.01)
            R_pipe = max(R_pipe, 0.01)

            # Solve Robin system with current R values
            phi_new = self._solve_robin_linear_system(R_anode, R_pipe, E_anode, E_pipe)

            # Check convergence
            delta = np.max(np.abs(phi_new.x.array - self.phi.x.array))

            if self.verbose:
                print(f"      Newton iter {iteration + 1}: delta={delta:.2e}, R_anode={R_anode:.3f}, R_pipe={R_pipe:.3f}")

            if delta < tolerance:
                converged = True
                if self.verbose:
                    print(f"      Converged in {iteration + 1} iterations")
                break

            # Damped update: phi = (1-damping)*phi_old + damping*phi_new
            self.phi.x.array[:] = (1.0 - damping) * self.phi.x.array + damping * phi_new.x.array

            # Update boundary potentials for next iteration
            phi_anode_avg = self._get_average_potential_on_boundary('anode')
            phi_pipe_avg = self._get_average_potential_on_boundary('pipe')

        if not converged and self.verbose:
            print(f"      Warning: Newton did not converge in {max_iterations} iterations (final delta={delta:.2e})")

        # 7. CALCULATE RESULTS
        results = self._calculate_simplified_results(degraded_params, T, humidity)

        # Add required fields
        results['time_years'] = t_years
        results['V_app'] = float(degraded_params[4])
        results['anode_efficiency'] = float(degraded_params[7])
        results['coating_quality'] = float(degraded_params[2])
        results['soil_resistivity'] = float(degraded_params[0])
        results['newton_converged'] = converged
        results['newton_iterations'] = iteration + 1

        return results

    def _calculate_nonlinear_results(self, params, T, humidity):
        """
        Расчет результатов с учетом нелинейных эффектов
        """
        # Получаем стандартные результаты
        results = self._calculate_simplified_results(params, T, humidity)
        
        # Добавляем нелинейные поправки
        
        # 1. Поправка на электрохимическую нелинейность
        phi_pipe = results['avg_potential']
        if phi_pipe > -0.65:  # Если выше равновесного потенциала стали
            # Увеличивается скорость коррозии
            results['corrosion_rate'] *= np.exp(2.0 * (phi_pipe + 0.65))
        
        # 2. Поправка на локальные токи (эффект "короткого замыкания")
        current_density = results['current_density']
        if current_density > 0.05:  # Высокая плотность тока
            # Снижается эффективность защиты
            overprotection_factor = min(1.0, 0.05 / current_density)
            results['coverage'] *= overprotection_factor
            results['corrosion_rate'] *= 2.0
        
        # 3. Поправка на распределение потенциала
        potential_range = results['max_potential'] - results['min_potential']
        if potential_range > 0.3:  # Большой разброс потенциалов
            # Снижается покрытие защиты на основе стандартного отклонения
            std_factor = max(0.5, 1.0 - results['std_potential'] / 0.2)
            results['coverage'] *= std_factor
        
        # 4. Метрики
        results['nonlinear_correction'] = True
        results['protection_efficiency'] = results['coverage'] / 100.0 * (
            1.0 - 0.2 * max(0, results['current_density'] - 0.02) / 0.08
        )
        
        return results

    def _calculate_simplified_results(self, params, T, humidity):
        """
        Расчет результатов с использованием методов из pipe.py и eage.py
        """
        try:
            # Получаем геометрические параметры
            pipe_start, pipe_end, pipe_y, pipe_radius = self.pipe.get_pipe_segment(self.domain_width)
            anode_x_min, anode_x_max, anode_y_min, anode_y_max = self.anode.get_anode_region(self.domain_width)
            
            # Находим DOF на границах
            anode_dofs = self._get_boundary_dofs('anode')
            pipe_dofs = self._get_boundary_dofs('pipe')
            
            # Расчет потенциалов
            E_pipe_eq = -0.65  # Steel equilibrium potential [V]

            if len(pipe_dofs) > 0:
                pipe_potentials = self.phi.x.array[pipe_dofs]
                avg_pipe_potential = np.mean(pipe_potentials)
                min_pipe_potential = np.min(pipe_potentials)
                max_pipe_potential = np.max(pipe_potentials)
                std_pipe_potential = np.std(pipe_potentials)

                # With Robin BC, phi is electrolyte potential at pipe surface
                # Current flows INTO pipe (cathodic protection) when phi > E_pipe_eq
                # Coverage = % of points receiving cathodic current
                R_pipe_approx = self.pipe.calculate_effective_resistance(
                    phi_surface=avg_pipe_potential, T=T, humidity=humidity
                )

                # Calculate cathodic current density at each point
                # j = (phi - E_pipe_eq) / R_pipe, positive = cathodic (protecting)
                current_densities = (pipe_potentials - E_pipe_eq) / max(R_pipe_approx, 0.01)

                # Protection threshold: minimum cathodic current for protection
                # ~0.0001 A/m² is a typical threshold for mild steel
                protection_threshold = 1e-4
                protection_mask = current_densities > protection_threshold
                coverage = 100.0 * np.sum(protection_mask) / len(pipe_potentials)
            else:
                # Значения по умолчанию
                avg_pipe_potential = -0.85
                min_pipe_potential = -0.9
                max_pipe_potential = -0.8
                std_pipe_potential = 0.02
                coverage = 100.0
            
            if len(anode_dofs) > 0:
                anode_potentials = self.phi.x.array[anode_dofs]
                avg_anode_potential = np.mean(anode_potentials)
            else:
                avg_anode_potential = params[4] * params[7]
            
            # Используем методы из классов для расчетов
            R_anode = self.anode.calculate_anode_resistance(
                phi_surface=avg_anode_potential, T=T, humidity=humidity
            )
            R_pipe = self.pipe.calculate_effective_resistance(
                phi_surface=avg_pipe_potential, T=T, humidity=humidity
            )
            R_total = R_anode + R_pipe
            
            # Расчет тока
            delta_V = avg_anode_potential - avg_pipe_potential
            if R_total > 0:
                current_density = delta_V / R_total
                anode_area = self.anode.get_area()
                pipe_area = self.pipe.get_area(self.domain_width)
                current = current_density * min(anode_area, pipe_area)
            else:
                current_density = 0.0
                current = 0.0
            
            # Расчет скорости коррозии
            corrosion_rate = self._estimate_corrosion_rate_from_potential(
                avg_pipe_potential, current_density, T
            )
            
            results = {
                'coverage': float(coverage),
                'avg_potential': float(avg_pipe_potential),
                'min_potential': float(min_pipe_potential),
                'max_potential': float(max_pipe_potential),
                'std_potential': float(std_pipe_potential),
                'anode_potential': float(avg_anode_potential),
                'current': float(current),
                'current_density': float(current_density),
                'corrosion_rate': float(corrosion_rate),
                'coating_resistance': float(self.pipe.calculate_coating_resistance_detailed(
                    t_years=params[6], humidity=humidity
                )),
                'anode_resistance': float(R_anode),
                'pipe_resistance': float(R_pipe),
                'voltage_drop': float(delta_V),
                'pipe_points': int(len(pipe_dofs)),
                'anode_points': int(len(anode_dofs))
            }
            
            return results
            
        except Exception as e:
            print(f"      ⚠️  Ошибка в расчете результатов: {e}")
            return self._get_default_results(params)

    def _get_default_results(self, params):
        """Возвращает результаты по умолчанию при ошибке"""
        V_app = params[4] if len(params) > 4 else 5.0
        anode_efficiency = params[7] if len(params) > 7 else 0.9
        return {
            'coverage': 100.0,
            'avg_potential': -0.85,
            'min_potential': -0.9,
            'max_potential': -0.8,
            'std_potential': 0.02,
            'anode_potential': float(V_app * anode_efficiency),
            'current': 0.01,
            'current_density': 0.001,
            'corrosion_rate': 0.001,
            'coating_resistance': 10000.0,
            'anode_resistance': 0.1,
            'pipe_resistance': 100.0,
            'voltage_drop': 1.0,
            'pipe_points': 100,
            'anode_points': 20
        }

    def _solve_robin_linear_system(self, R_anode, R_pipe, E_anode=1.2, E_pipe=-0.65):
        """
        Solve the Robin boundary condition linear system.

        Uses meshtags for boundaries:
        - ds_anode with ID=6
        - ds_pipe with ID=5

        Bilinear form: a = inner(sigma*grad(u), grad(v))*dx + (1/R_anode)*u*v*ds_anode + (1/R_pipe)*u*v*ds_pipe
        Linear form: L = (E_anode/R_anode)*v*ds_anode + (E_pipe/R_pipe)*v*ds_pipe

        Args:
            R_anode: Anode resistance [Ohm*m^2]
            R_pipe: Pipe effective resistance [Ohm*m^2]
            E_anode: Anode equilibrium potential [V] (default 1.2V for Mg)
            E_pipe: Pipe equilibrium potential [V] (default -0.65V for steel)

        Returns:
            phi_new: Solution function
        """
        # Ensure facet_markers exist
        if not hasattr(self, 'facet_markers') or self.facet_markers is None:
            self.mark_boundaries()

        # Create measures for boundaries using meshtags
        ds_anode = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_markers, subdomain_id=6)
        ds_pipe = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_markers, subdomain_id=5)

        # Trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Bilinear form: volume term + Robin BC terms
        a = ufl.inner(self.sigma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        a += (1.0 / R_anode) * u * v * ds_anode
        a += (1.0 / R_pipe) * u * v * ds_pipe

        # Linear form: Robin BC source terms
        L = (E_anode / R_anode) * v * ds_anode + (E_pipe / R_pipe) * v * ds_pipe

        # Assemble and solve
        phi_new = fem.Function(self.V)

        A = fem.petsc.assemble_matrix(fem.form(a))
        A.assemble()

        b = fem.petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.CG)
        solver.setTolerances(rtol=1e-10, max_it=2000)

        solver.solve(b, phi_new.x.petsc_vec)
        phi_new.x.scatter_forward()

        return phi_new

    def _estimate_corrosion_rate_from_potential(self, phi, current_density, T):
        """
        Оценка скорости коррозии на основе потенциала и плотности тока
        (должен быть в pipe.py, но временно здесь)
        """
        if phi > -0.85:
            # Линейная аппроксимация при недостаточной защите
            base_rate = 0.1 * max(0, (phi + 0.85) / 0.3)
            # Влияние тока
            current_factor = 1.0 + 0.5 * max(0, current_density - 0.02) / 0.08
            # Влияние температуры (Q10 ≈ 2)
            T_factor = 2.0 ** ((T - 20.0) / 10.0)
            
            corrosion_rate = base_rate * current_factor * T_factor
            return min(corrosion_rate, 0.5)
        else:
            return 0.001  # Низкая скорость при защите

    def _get_boundary_dofs(self, boundary_type=None):
        """
        Получение DOF на границах трубы и анода.

        For 2D FEM with exterior BCs only:
        - Pipe is on TOP boundary (y=domain_height) in x range [5, 15]
        - Anode is on BOTTOM boundary (y=0) in x range [9, 11]

        Args:
            boundary_type: 'anode', 'pipe', или None для обоих

        Returns:
            Если boundary_type указан: список DOF
            Иначе: (anode_dofs, pipe_dofs)
        """
        # Получаем геометрические параметры
        pipe_start, pipe_end, _, _ = self.pipe.get_pipe_segment(self.domain_width)
        anode_x_min, anode_x_max, _, _ = self.anode.get_anode_region(self.domain_width)

        # Определяем функции для границ - anode on BOTTOM boundary
        def anode_boundary(x):
            in_x = (anode_x_min <= x[0]) & (x[0] <= anode_x_max)
            on_bottom = np.abs(x[1]) < 0.1
            return np.logical_and(in_x, on_bottom)

        # Pipe on TOP boundary
        def pipe_boundary(x):
            in_x = (pipe_start <= x[0]) & (x[0] <= pipe_end)
            on_top = np.abs(x[1] - self.domain_height) < 0.1
            return np.logical_and(in_x, on_top)
        
        try:
            if boundary_type == 'anode':
                return fem.locate_dofs_geometrical(self.V, anode_boundary)
            elif boundary_type == 'pipe':
                return fem.locate_dofs_geometrical(self.V, pipe_boundary)
            else:
                # Возвращаем оба
                anode_dofs = fem.locate_dofs_geometrical(self.V, anode_boundary)
                pipe_dofs = fem.locate_dofs_geometrical(self.V, pipe_boundary)
                return anode_dofs, pipe_dofs
                
        except Exception as e:
            print(f"      ⚠️  Ошибка при поиске DOF: {e}")
            # Возвращаем пустые списки
            if boundary_type == 'anode':
                return []
            elif boundary_type == 'pipe':
                return []
            else:
                return [], []

    def apply_degradation(self, base_params, t_years):
        """Применение износа ко всем компонентам системы"""
        
        # Извлечение базовых параметров
        base_roughness = base_params[1]
        base_coating_quality = base_params[2]
        base_anode_efficiency = base_params[7]
        
        # Применение деградации к трубе
        coating_quality, roughness = self.pipe.apply_degradation(
            base_coating_quality, base_roughness, t_years
        )
        
        # Применение деградации к аноду
        anode_efficiency = self.anode.apply_degradation(
            base_anode_efficiency, t_years
        )
        
        # Создание обновленного вектора параметров
        degraded = base_params.copy()
        degraded[1] = roughness
        degraded[2] = coating_quality
        degraded[6] = t_years
        degraded[7] = anode_efficiency
        
        print(f"  t={t_years} лет: покрытие {base_params[2]:.2f}→{coating_quality:.2f}, "
              f"анод {base_params[7]:.2f}→{anode_efficiency:.2f}") if self.verbose else None
        
        return degraded
    
