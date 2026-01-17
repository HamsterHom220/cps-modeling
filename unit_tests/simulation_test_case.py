import numpy as np
from simulator import CPS_DegradationSimulator
from soil import SoilModel
from dolfinx import mesh
from mpi4py import MPI

# Reproduce case 371 (seed 18682)
np.random.seed(18682)
params = [
    np.random.uniform(3, 8),      # R_sigma
    np.random.uniform(0.1, 0.5),  # Roughness
    np.random.uniform(0.7, 0.95), # Coating quality
    np.random.uniform(6.5, 8),    # pH
    np.random.uniform(3, 7),      # V_app
    np.random.uniform(0.3, 0.8),  # Humidity
    0.0,                          # Age
    np.random.uniform(0.9, 0.95)  # Anode efficiency
]

print("Testing case 371 (seed=18682) with parameters:")
labels = ['R_sigma', 'Roughness', 'Coating_q', 'pH', 'V_app', 'Humidity', 'Age',
'Anode_eff']
for l, p in zip(labels, params):
    print(f"  {l:12s} = {p:.4f}")

# Create shared domain
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([20.0, 8.0])],
    [80, 32],
    mesh.CellType.triangle
)

# Create simulator
sim = CPS_DegradationSimulator(verbose=True)
sim.domain = domain
sim.create_mesh_and_function_space()

# Create soil model
soil_model = SoilModel(domain, params, 8.0, 7.5, enable_plotting=False)

# Try to solve at t=0 (this previously froze)
print("\nAttempting solve at t=0 years...")
import time
start = time.time()
try:
    results = sim.solve_with_robin_bc(params, 0.0, soil_model)
    elapsed = time.time() - start
    print(f"\n✓ Solution completed in {elapsed:.2f} seconds")
    print(f"  Coverage: {results.get('coverage', 'N/A'):.4f}")
    print(f"  Newton converged: {results.get('newton_converged', False)}")
except Exception as e:
    elapsed = time.time() - start
    print(f"\n✗ Failed after {elapsed:.2f} seconds: {e}")