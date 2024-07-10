# Imports
from firedrake import *
import numpy as np
import gc



#####



# Parallelise "print"
_print = print  # Save old serial "print"
def print(x):  # Create new parallel "print"
    if mesh.comm.rank == 0:
        _print(x, flush = True)



#####



# Model parameters
#   Setting
# M = Constant(1 / np.sqrt(2))  # Mach number
# Re = Constant(32)  # Reynolds number

#   Constitutive relations
CV = Constant(2.50)  # Specific heat capacity (Material-dependent, ~2.50 standard for air)

theta = lambda rho, eps : eps / rho / CV  # Non-AV theta (temperature)
p     = lambda rho, eps : 1/CV * eps  # Non-AV p (pressure)
s     = lambda rho, eps : rho * ln(theta(rho, eps)**CV / rho)  # Non-AV s (entropy density)



# Discretisation parameters
#   Space
# nx = round(2 * float(Re))  # Mesh number (Should be greater than ~Re & ~Re*Pr)
nx = 32
k = 1  # Spatial degree (Must be >=1)

#   Time
# timestep = Constant(2 / float(Re))  # Timestep (Should be smaller than ~1/M/Re)
timestep = Constant(1 / 4 / nx)
duration = Constant(16)  # Duration



#####



# Create mesh and coordinates
mesh = PeriodicUnitSquareMesh(nx, nx, quadrilateral = False)
(x, y) = SpatialCoordinate(mesh)



#####



# Create trial/test function spaces
#   Function spaces
Vec = VectorFunctionSpace(mesh, "P", k)
Sca = FunctionSpace(mesh, "P", k)
SME = MixedFunctionSpace([Sca, Vec, Sca])

#   Trial functions (rho_t, u_t, theta_t)
sme = Function(SME)
(sigma_t, mu_t, ln_eps_t) = split(sme)

#   Test functions
(v_sigma, v_mu, v_eps) = split(TestFunction(SME))



# Create IC/persistent value trackers
sme_ = project(as_vector([
    exp(1/2 * sin(2*np.pi*x) * sin(2*np.pi*y)),
    0,
    0,
    (1+1/CV) * sin(2*np.pi*x) * sin(2*np.pi*y)
]), SME)
(sigma_, mu_, ln_eps_) = split(sme_)



# Midpoint values
sigma  = sigma_  + timestep*0.5*sigma_t
mu     = mu_     + timestep*0.5*mu_t
ln_eps = ln_eps_ + timestep*0.5*ln_eps_t



# Useful variables
u   = mu / sigma
mom = mu * sigma
rho = sigma**2
eps = exp(ln_eps)



#####



# Symmetric, trace-free gradient
tau = lambda u : sym(grad(u)) - 1/3 * div(u) * Identity(2)



# Set residual
F = (
    (  # Mass
      + inner(2 * sigma * sigma_t, v_sigma) * dx  # Change
      - inner(mom, grad(v_sigma)) * dx  # Convection
    )
  + (  # Momentum
      + inner(sigma * mu_t, v_mu) * dx  # Change
      - 0.5 * (  # Convection
            inner(outer(u, mom),     grad(v_mu))
          - inner(dot(grad(u), mom), v_mu)
        ) * dx
      + inner(grad(p(rho, eps)), v_mu) * dx  # Pressure
    )
  + (  # Energy
      + inner(eps * ln_eps_t, v_eps) * dx  # Change
      - inner(eps * u, grad(v_eps)) * dx  # Convection
      - (
            inner(dot(grad(p(rho, eps)), u), v_eps)
          + inner(p(rho, eps) * u, grad(v_eps))
        ) * dx  # Pressure
    )
)



#####



# Set solver parameters
sp = {
    # Outer (nonlinear) solver
    "snes_atol": 1e-13,
    "snes_rtol": 1e-13,

    "snes_converged_reason"     : None,
    "snes_linesearch_monitor"   : None,
    "snes_monitor"              : None,

    # Inner (linear) solver
    "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    "pc_type"                   : "lu",
    "pc_factor_mat_solver_type" : "mumps",
    #"ksp_atol"                  : 1e-8,
    #"ksp_rtol"                  : 1e-8,
    #"ksp_max_it"                : 100,

    "ksp_monitor_true_residual" : None,
}


#####



# Record IC values
#   Create ParaView file
pvd = File("output/compressible_ns/euler/im/solution.pvd")

#   Write to ParaView file
(sigma_out, mu_out, eps_out) = sme_.subfunctions
sigma_out.rename("Root density")
mu_out.rename("Root density * velocity")
eps_out.rename("Log internal energy")

pvd.write(sigma_out, mu_out, eps_out)



# Create text files
mass_txt            = "output/compressible_ns/euler/im/mass.txt"
momentum_txt        = "output/compressible_ns/euler/im/momentum.txt"
kinetic_energy_txt  = "output/compressible_ns/euler/im/kinetic_energy.txt"
internal_energy_txt = "output/compressible_ns/euler/im/internal_energy.txt"
energy_txt          = "output/compressible_ns/euler/im/energy.txt"
entropy_txt         = "output/compressible_ns/euler/im/entropy.txt"

# Write to text files
mass = assemble(sigma_**2 * dx)
print(GREEN % f"Mass: {mass}")
if mesh.comm.rank == 0:
    open(mass_txt, "w").write(str(mass) + "\n")

momentum = [float(assemble(sigma_ * mu_[i] * dx)) for i in range(2)]
print(GREEN % f"Momentum: {momentum}")
if mesh.comm.rank == 0:
    open(momentum_txt, "w").write(str(momentum) + "\n")

kinetic_energy = assemble(1/2 * inner(mu_, mu_) * dx)
internal_energy = assemble(exp(ln_eps_) * dx)
energy = kinetic_energy + internal_energy
print(GREEN % f"Energy: {energy}")
if mesh.comm.rank == 0:
    open(kinetic_energy_txt, "w").write(str(kinetic_energy) + "\n")
if mesh.comm.rank == 0:
    open(internal_energy_txt, "w").write(str(internal_energy) + "\n")
if mesh.comm.rank == 0:
    open(energy_txt, "w").write(str(energy) + "\n")

entropy = assemble(s(sigma_**2, exp(ln_eps_)) * dx)
print(GREEN % f"Entropy: {entropy}")
if mesh.comm.rank == 0:
    open(entropy_txt, "w").write(str(entropy) + "\n")



# Solve
time = Constant(0.0)
while (float(time) < float(duration) - float(timestep)/2):
    # Print timestep
    print(RED % f"Solving for t = {float(time) + float(timestep)}:")



    # Solve
    solve(F == 0, sme, solver_parameters = sp)



    # Collect garbage
    gc.collect()



    # Update values
    for i in range(3):
        sme_.sub(i).assign(sme_.sub(i) + timestep*sme.sub(i))



    # Write to ParaView
    pvd.write(sigma_out, mu_out, eps_out)




    # Write to text files
    mass = assemble(sigma_**2 * dx)
    print(GREEN % f"Mass: {mass}")
    if mesh.comm.rank == 0:
        open(mass_txt, "a").write(str(mass) + "\n")

    momentum = [float(assemble(sigma_ * mu_[i] * dx)) for i in range(2)]
    print(GREEN % f"Momentum: {momentum}")
    if mesh.comm.rank == 0:
        open(momentum_txt, "a").write(str(momentum) + "\n")

    kinetic_energy = assemble(1/2 * inner(mu_, mu_) * dx)
    internal_energy = assemble(exp(ln_eps_) * dx)
    energy = kinetic_energy + internal_energy
    print(GREEN % f"Energy: {energy}")
    if mesh.comm.rank == 0:
        open(kinetic_energy_txt, "a").write(str(kinetic_energy) + "\n")
    if mesh.comm.rank == 0:
        open(internal_energy_txt, "a").write(str(internal_energy) + "\n")
    if mesh.comm.rank == 0:
        open(energy_txt, "a").write(str(energy) + "\n")

    entropy = assemble(s(sigma_**2, exp(ln_eps_)) * dx)
    print(GREEN % f"Entropy: {entropy}")
    if mesh.comm.rank == 0:
        open(entropy_txt, "a").write(str(entropy) + "\n")
    


    # Increment time
    time.assign(float(time) + float(timestep))
