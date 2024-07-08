'''
Imports
'''
from firedrake import *
from irksome import *
import numpy as np
import av_fet_modules.terminal_options as terminal_options



'''
General purpose functions
'''
# Parallelised "print"
print_ = print
def print(x):
    if mesh.comm.rank == 0:
        print_(x, flush = True)

# Sech
def sech(x):
    return 2 / (exp(x) + exp(- x))

# Soliton
def soliton(x, x_0, c):
    return 3 * (c-1) * sech(0.5 * sqrt(1 - 1/c) * (x-x_0))**2


'''
Parameters
'''
# Spatial discretisation
L = terminal_options.get("L", type=float, default=100)  # Grid length
nx = terminal_options.get("L", type=int, default=L/2)  # Mesh number (Per unit length)

# Temporal discretisation
stages = terminal_options.get("stages", type=int, default=2)  # (Max.) temporal degree (Must be >=1 | Equiv. to no. of steps in timestepping scheme)
dur = Constant(terminal_options.get("dur", type=float, default=20000))  # Duration
dt = Constant(terminal_options.get("dt", type=float, default=1))  # Timestep

# Initial conditions
c_max = Constant(terminal_options.get("c_max", type=float, default=(1+sqrt(5))/2))  # Max soliton speed (Must be >=1)



'''
Mesh (and properties)
'''
# Create mesh
mesh = PeriodicIntervalMesh(nx, L)

# Get properties
x, = SpatialCoordinate(mesh)  # Cartesian coordinates



'''
Function spaces
'''
# Individual (spatial) spaces
U = FunctionSpace(mesh, "HER", 3)  # Persistent/intermediate value space



'''
Functions
'''
# Trial functions
u = Function(U)
# u = project(
#     soliton(x, 1/6 * L, c_max)
#   + soliton(x, 2/6 * L, 1 + 2/3 * (c_max - 1))
#   + soliton(x, 5/6 * L, 1 + 1/3 * (c_max - 1)),
#     U
# )
u = project(
    soliton(x, 1/2 * L, c_max),
    U
)

# Test functions
v = TestFunction(U)



'''
Residual definition
'''
F = (
    (  # LHS
        inner(Dt(u), v)
      + inner((Dt(u)).dx(0), v.dx(0))
    )*dx
  + inner(u.dx(0) + u * u.dx(0), v)*dx  # RHS
)



'''
Solver parameters
'''
sp = {
    # Outer (nonlinear) solver
    "snes_atol": 1.0e-10,

    "snes_converged_reason"     : None,
    "snes_linesearch_monitor"   : None,
    "snes_monitor"              : None,

    # Inner (linear) solver
    "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    "pc_type"                   : "lu",
    "pc_factor_mat_solver_type" : "mumps",
    # "ksp_atol"                  : 1e-8,
    # "ksp_rtol"                  : 1e-8,
    # "ksp_max_it"                : 100,

    # "ksp_monitor" : None,
    # "ksp_converged_reason" : None,
    "ksp_monitor_true_residual" : None,
}



'''
Solve
'''
# Record data
u_data = u.dat.data  # u
energy = assemble((1/2 * u**2 + 1/6 * u**3) * dx)  # Energy
print(GREEN % f"Energy: {energy}")
energy_data = energy
h1_norm = assemble((u**2 + u.dx(0)**2) * dx)  # H1 norm
print(GREEN % f"H1 norm: {h1_norm}")
h1_norm_data = h1_norm

# Set up timestepper
time = Constant(0.0)
stepper = TimeStepper(F, GaussLegendre(stages), time, dt, u, solver_parameters=sp)

# Solve
while (float(time) < float(dur) - float(dt)/2):
    # Print timestep
    print(BLUE % f"Solving for t = {float(time) + float(dt)}:")

    # Advance timestepper
    stepper.advance()

    # Record data
    u_data = np.vstack((u_data, u.dat.data))  # u
    energy = assemble((1/2 * u**2 + 1/6 * u**3) * dx)  # Energy
    print(GREEN % f"Energy: {energy}")
    energy_data = np.vstack((energy_data, energy))
    h1_norm = assemble((u**2 + u.dx(0)**2) * dx)  # H1 norm
    print(GREEN % f"H1 norm: {h1_norm}")
    h1_norm_data = np.vstack((h1_norm_data, h1_norm))

    # Increment time
    time.assign(float(time) + float(dt))

# Record data
with open("output/u.txt", "w") as file:
    np.savetxt(file, u_data)  # u
with open("output/energy.txt", "w") as file:
    np.savetxt(file, energy_data)  # Energy
with open("output/h1_norm.txt", "w") as file:
    np.savetxt(file, h1_norm_data)  # H1 norm
