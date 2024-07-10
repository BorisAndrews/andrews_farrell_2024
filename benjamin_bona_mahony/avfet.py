'''
Imports
'''
from firedrake import *
import avfet_modules.cheb_fet as cheb_fet
import numpy as np
import avfet_modules.terminal_options as terminal_options



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
c_max = Constant(terminal_options.get("c_max", type=float, default=float(1+sqrt(5))/2))  # Max soliton speed (Must be >=1)



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
U_ = FunctionSpace(mesh, "HER", 3)  # Persistent/intermediate value space (For e.g. u|_t^n)

# Individual (space-time) spaces (For composition...)
U = cheb_fet.FETFunctionSpace(mesh, "HER", 3, stages-1)  # u_t

# Mixed (space-time) spaces (...as required)
UR  = cheb_fet.FETMixedFunctionSpace([U, U])  # (u_t, r)



'''
Functions
'''
# Trial functions (Space-time)
ur = Function(UR)
(u_t, r) = cheb_fet.FETsplit(ur)

# Test functions (Space-time)
vs = TestFunction(UR)
(v, s) = cheb_fet.FETsplit(vs)

# Persistent value trackers (Spatial only)
# u_ = project(
#     soliton(x, 1/6 * L, c_max)
#   + soliton(x, 2/6 * L, 1 + 2/3 * (c_max - 1))
#   + soliton(x, 5/6 * L, 1 + 1/3 * (c_max - 1)),
#     U_
# )
u_ = project(
    soliton(x, 1/2 * L, c_max),
    U_
)

# Integated trial functions (Space-time)
u = cheb_fet.integrate(u_t, u_, dt)



'''
Residual definition
'''
# Initialise residual
F = 0

# LHS
F = cheb_fet.residual(
    F,
    lambda a, b : - (
        inner(a, b)
      + inner(a.dx(0), b.dx(0))
    )*dx,
    (u_t, v)
)
# RHS
F = cheb_fet.residual(
    F,
    lambda a, b: inner(a - a.dx(0).dx(0), b.dx(0))*dx,
    (r, v)
)

# AV LHS
F = cheb_fet.residual(
    F,
    lambda a, b : - (
        inner(a, b)
      + inner(a.dx(0), b.dx(0))
    )*dx,
    (r, s)
)
# AV RHS
F = cheb_fet.residual(
    F,
    lambda a, b : inner(a, b)*dx,
    (u, s)
)
F = cheb_fet.residual(
    F,
    lambda a, b, c : 0.5 * inner(a * b, c)*dx,
    (u, u, s)
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
u_data = u_.dat.data  # u
energy = assemble((1/2 * u_**2 + 1/6 * u_**3) * dx)  # Energy
print(GREEN % f"Energy: {energy}")
energy_data = energy
h1_norm = assemble((u_**2 + u_.dx(0)**2) * dx)  # H1 norm
print(GREEN % f"H1 norm: {h1_norm}")
h1_norm_data = h1_norm

# Solve
time = Constant(0.0)
while (float(time) < float(dur) - float(dt)/2):
    # Print timestep
    print(BLUE % f"Solving for t = {float(time) + float(dt)}:")

    # Solve
    solve(F == 0, ur, solver_parameters = sp)

    # Update u
    u_.assign(cheb_fet.FETeval(
        (u_, None),
        (ur, 0),
        dt,
        dt
    ))

    # Record data
    u_data = np.vstack((u_data, u_.dat.data))  # u
    energy = assemble((1/2 * u_**2 + 1/6 * u_**3) * dx)  # Energy
    print(GREEN % f"Energy: {energy}")
    energy_data = np.vstack((energy_data, energy))
    h1_norm = assemble((u_**2 + u_.dx(0)**2) * dx)  # H1 norm
    print(GREEN % f"H1 norm: {h1_norm}")
    h1_norm_data = np.vstack((h1_norm_data, h1_norm))

    # Increment time
    time.assign(float(time) + float(dt))

# Record data
with open("output/benjamin_bona_mahony/avfet/u.txt", "w") as file:
    np.savetxt(file, u_data)  # u
with open("output/benjamin_bona_mahony/avfet/energy.txt", "w") as file:
    np.savetxt(file, energy_data)  # Energy
with open("output/benjamin_bona_mahony/avfet/h1_norm.txt", "w") as file:
    np.savetxt(file, h1_norm_data)  # H1 norm
