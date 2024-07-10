'''
Imports
'''
from firedrake import *
import avfet_modules.cheb_fet as cheb_fet
import avfet_modules.project_tools as project_tools
from scipy import special
import math
import gc



'''
General purpose functions
'''
# Parallelised "print"
print_ = print
def print(x):
    if mesh.comm.rank == 0:
        print_(x, flush = True)



'''
Hill vortex functions
'''
# Firedrake-compatible Bessel function
def besselJ(x, alpha, layers=10):
    return sum([
        (-1)**m / math.factorial(m) / special.gamma(m + alpha + 1)
      * (x/2)**(2*m+alpha)
        for m in range(layers)
    ])



# Bessel function parameters
besselJ_root = 5.7634591968945506
besselJ_root_threehalves = besselJ(besselJ_root, 3/2)



# (r, theta, phi) components of Hill vortex
def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (
        besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
      - besselJ_root_threehalves
    ) * cos(theta)

def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        besselJ_root * besselJ(besselJ_root*rho, 5/2) / rho**(1/2)
      + 2 * besselJ_root_threehalves
      - 2 * besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
    ) * sin(theta)

def hill_phi(r, theta, radius):
    rho = r / radius
    return besselJ_root * (
        besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
      - besselJ_root_threehalves
    ) * rho * sin(theta)



# Hill vortex (Cartesian)
def hill(vec, radius):
    (x, y, z) = vec

    # Cylindrical/spherical coordinates
    r_cyl = sqrt(x**2 + y**2)  # Cylindrical radius
    r_sph = sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = conditional(  # Spherical angle
        le(r_cyl, 1e-13),
        0,
        pi/2 - atan(z/r_cyl)
    )

    return conditional(  # If we're outside the vortex...
        ge(r_sph, radius),
        as_vector([0, 0, 0]),
        conditional(  # If we're at the origin...
            le(r_sph, 1e-13),
            as_vector([0, 0, 2*((besselJ_root/2)**(3/2)/special.gamma(5/2) - besselJ_root_threehalves)]),
            conditional(  # If we're on the z axis...
                le(r_cyl, 1e-13),
                as_vector([0, 0, hill_r(r_sph, 0, radius)]),
                as_vector(  # Else...
                    hill_r(r_sph, theta, radius) * np.array([x, y, z]) / r_sph
                  + hill_theta(r_sph, theta, radius) * np.array([x*z, y*z, -r_cyl**2]) / r_sph / r_cyl
                  + hill_phi(r_sph, theta, radius) * np.array([-y, x, 0]) / r_cyl
                )
            )
        )
    )



'''
Parameters
'''
# Spatial discretisation
nx = 8  # Mesh number (Per unit length)
k = 2  # (Max.) spatial degree (Must be >=2)

# Temporal discretisation
s = 3  # (Max.) temporal degree (Must be >=1 | Equiv. to no. of steps in timestepping scheme)
duration = 3*2**(-6)
timestep = Constant(2**(-10))

# Setting
Re_arr = [2**(2*i) for i in range(0,9)]



'''
Mesh (and properties)
'''
# Create mesh
mesh = PeriodicUnitCubeMesh(nx, nx, nx)

# Get properties
(x, y, z) = SpatialCoordinate(mesh)  # Cartesian coordinates



'''
Function spaces
'''
# Individual (spatial) spaces
U_ = VectorFunctionSpace(mesh, "CG", k)  # Persistent/intermediate value space (For e.g. u|_t^n)
P_ = FunctionSpace(mesh, "CG", k-1)
R_ = FunctionSpace(mesh, "R", 0)

# Individual (space-time) spaces (For composition...)
U = cheb_fet.FETVectorFunctionSpace(mesh, "CG", k, s-1)
P = cheb_fet.FETFunctionSpace(mesh, "CG", k-1, s-1)
R = cheb_fet.FETFunctionSpace(mesh, "R", 0, s-1)

# Mixed (space-time) spaces (...as required)
UP  = cheb_fet.FETMixedFunctionSpace([U, P, R, R, R])

# Print number of degrees of freedom
print(RED % f"Degrees of freedom: {UP.dim()} {[UP_.dim() for UP_ in UP]}")



'''
Functions
'''
# Trial functions (Space-time)
up = Function(UP)
(u_t, p, f_1, f_2, f_3) = cheb_fet.FETsplit(up)

# Test functions (Space-time)
v_up = TestFunction(UP)
(v, q, g_1, g_2, g_3) = cheb_fet.FETsplit(v_up)

# Persistent value trackers (Spatial only)
u_ref = project_tools.project_op_free(
    2 * hill([x-0.5, y-0.5, z-0.5], 0.25),
    U_,
    (lambda u, p: inner(div(u), p)*dx, P_),
    *[(lambda u, r: inner(u[i], r)*dx, R_) for i in range(3)]
)
u_ = Function(U_)


# Integated trial functions (Space-time)
u       = cheb_fet.integrate(u_t, u_, timestep)
u_tilde = cheb_fet.project(u)



'''
Residual definition
'''
# Default Reynolds no.
Re = Constant(1)



# Initialise residual
F = 0



# u main
F = cheb_fet.residual(
    F,
    lambda a, b : inner(a, b)*dx,
    (u_t, v)
)
F = cheb_fet.residual(
    F,
    lambda a, b, c : - inner(cross(a, curl(b)), c)*dx,
    (u_tilde, u, v)
)
F = cheb_fet.residual(
    F,
    lambda a, b : 1/Re * inner(grad(a), grad(b))*dx,
    (u_tilde, v)
)

F = cheb_fet.residual(
    F,
    lambda a, b : - inner(a, div(b))*dx,
    (p, v)
)
for (i, f_i) in enumerate([f_1, f_2, f_3]):
    F = cheb_fet.residual(
        F,
        lambda a, b : - inner(a, b[i])*dx,
        (f_i, v)
    )

# u incompressibility
F = cheb_fet.residual(
    F,
    lambda a, b : - inner(div(a), b)*dx,
    (u_tilde, q)
)

# u stationary
for (i, g_i) in enumerate([g_1, g_2, g_3]):
    F = cheb_fet.residual(
        F,
        lambda a, b : - inner(a[i], b)*dx,
        (u_tilde, g_i)
    )



'''
Solver parameters
'''
sp = {
    # # Outer (nonlinear) solver
    # "snes_atol": 1.0e-11,
    # "snes_rtol": 1.0e-11,

    # "snes_converged_reason"     : None,
    # "snes_linesearch_monitor"   : None,
    # "snes_monitor"              : None,

    # # Inner (linear) solver
    # "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    # "pc_type"                   : "lu",
    # "pc_factor_mat_solver_type" : "mumps",
    # # "ksp_atol"                  : 1e-8,
    # # "ksp_rtol"                  : 1e-8,
    # # "ksp_max_it"                : 100,

    # # "ksp_monitor" : None,
    # # "ksp_converged_reason" : None,
    # "ksp_monitor_true_residual" : None,
}



'''
Solve loop
'''
for (i, Re_) in enumerate(Re_arr):
    # Print Reynolds no.
    print(RED % f"Solving for Re = {Re_}:")

    # Reset
    Re.assign(Re_)
    u_.assign(u_ref)



    '''
    Solve setup
    '''
    # Create ParaView file
    pvd = VTKFile("output/incompressible_ns/avfet_nohelicity/re_" + str(Re_) + "/solution.pvd")

    # Write to Paraview file
    u_.rename("Velocity")
    pvd.write(u_)



    # Create text files
    energy_txt   = "output/incompressible_ns/avfet_nohelicity/re_" + str(Re_) + "/energy.txt"
    helicity_txt = "output/incompressible_ns/avfet_nohelicity/re_" + str(Re_) + "/helicity.txt"
    momentum_txt = "output/incompressible_ns/avfet_nohelicity/re_" + str(Re_) + "/momentum.txt"

    # Write to text files
    energy = assemble(1/2 * inner(u_, u_)*dx)
    if mesh.comm.rank == 0:
        open(energy_txt, "w").write(str(energy) + "\n")

    helicity = assemble(1/2 * inner(u_, curl(u_))*dx)
    print(GREEN % f"Helicity: {helicity}")
    if mesh.comm.rank == 0:
        open(helicity_txt, "w").write(str(helicity) + "\n")

    momentum = [float(assemble(u_[i]*dx)) for i in range(3)]
    print(GREEN % f"Momentum: {momentum}")
    if mesh.comm.rank == 0:
        open(momentum_txt, "w").write(str(momentum) + "\n")



    '''
    Solve
    '''
    time = 0.0
    while (time < duration - float(timestep)/2):
        # Print timestep
        print(RED % f"Solving for t = {float(time) + float(timestep)}:")

        # Solve
        solve(F==0, up, solver_parameters=sp)

        # Collect garbage
        gc.collect()
        
        # Record dissipation
        energy_diss = cheb_fet.FETassemble(
            lambda a, b : 1/Re * inner(grad(a), grad(b))*dx,
            (u_tilde, u_tilde),
            timestep
        )
        print(BLUE % f"Energy dissipation: {energy_diss}")

        # Update u
        u_.assign(cheb_fet.FETeval((u_, None), (up, 0), timestep, timestep))

        # Write to Paraview
        pvd.write(u_)

        # Write to text files
        energy = assemble(1/2 * inner(u_, u_)*dx)
        print(GREEN % f"Energy: {energy}")
        if mesh.comm.rank == 0:
            open(energy_txt, "a").write(str(energy) + "\n")

        helicity = assemble(1/2 * inner(u_, curl(u_))*dx)
        print(GREEN % f"Helicity: {helicity}")
        if mesh.comm.rank == 0:
            open(helicity_txt, "a").write(str(helicity) + "\n")

        momentum = [float(assemble(u_[i]*dx)) for i in range(3)]
        print(GREEN % f"Momentum: {momentum}")
        if mesh.comm.rank == 0:
            open(momentum_txt, "a").write(str(momentum) + "\n")

        # Increment time
        time += float(timestep)
