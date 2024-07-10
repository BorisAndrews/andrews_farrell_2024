import avfet_modules.terminal_options as terminal_options
import avfet_modules.timestepping as timestepping
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat



'''
Parameters
'''
# Independent
scheme = terminal_options.get("scheme", default="gauss")  # Timestepping scheme to be used
stages = terminal_options.get("stages", int, default=1)  # Number of stages (for multi-stage schemes like Gauss)

dt = terminal_options.get("dt", type=float, default=1e-1)  # Timestep
dur = terminal_options.get("dur", type=float, default=1e2)  # Simulation duration
res = terminal_options.get("res", type=int, default=1)  # Resolution per timestep for recording output data

x_0 = np.array([0.4, 0.0])  # x ICs
v_0 = np.array([0.0, 2.0])  # v ICs



# Dependent
dot = lambda u, v: np.sum(u * v)  # Dot product
cross = lambda v: np.array([-v[1], v[0]])  # Cross with k, v -> k^v
cross_dot = lambda u, v: u[0] * v[1] - u[1] * v[0]  # Cross and dot with k, (u, v) -> {k, u, v}
l2_norm = lambda v: np.sum(v**2)**0.5  # L2 norm

V = lambda x: - 1/l2_norm(x)  # (Gravitational) potential energy
H = lambda x, v: 0.5 * l2_norm(v)**2 + V(x)  # Hamiltonian (Kinetic + Gravitational potential)

F = lambda x: - 1/l2_norm(x)**3 * x  # Gravity

Pois = np.array([  # Poisson matrix
    [0, 0,  -1, 0],
    [0, 0,  0,  -1],
    [1, 0,  0,  0],
    [0, 1,  0,  0]
])
H_grad = lambda x, v: np.hstack((- F(x), v))  # (dH_dx, dH_dv)

L = lambda x, v: cross_dot(x, v)  # Angular momentum
L_grad = lambda x, v: np.hstack((- cross(v), cross(x)))
A = lambda x, v: - L(x, v) * cross(v) - 1/l2_norm(x) * x  # Runge--Lenz
A_grad = lambda x, v: (
  - (
        np.outer(cross(v), L_grad(x, v))
      + L(x, v) * np.array([[0,0,0,-1],[0,0,1,0]])
    )
  - np.hstack(
        (1/l2_norm(x) * np.array([[1,0],[0,1]]) - 1/l2_norm(x)**3 * np.outer(x, x),
        np.zeros((2,2)))
    )
)
theta = lambda x, v: math.atan2(A(x, v)[1], A(x, v)[0])

N = round(dur/dt)  # Number of time steps



'''
Solver
'''
# Define explicit timesteppers
def explicit_euler_timestepper(xv_start):
    xv_end = timestepping.schemes.explicit_euler(
        xv_start,
        dt,
        lambda xv: np.hstack((
            xv[2:4],
            F(xv[0:2])
            ))
        )
    return xv_end

def rk4_timestepper(xv_start):
    xv_end = timestepping.schemes.rk4(
        xv_start,
        dt,
        lambda xv: np.hstack((
            xv[2:4],
            F(xv[0:2])
            ))
        )
    return xv_end

# Define implicit timesteppers
def implicit_euler_timestepper(xv_start):
    xv_end = timestepping.schemes.implicit_euler(
        xv_start,
        dt,
        lambda xv: np.hstack((
            xv[2:4],
            F(xv[0:2])
            ))
        )
    return xv_end

def implicit_midpoint_timestepper(xv_start):
    xv_end = timestepping.schemes.implicit_midpoint(
        xv_start,
        dt,
        lambda xv: np.hstack((
            xv[2:4],
            F(xv[0:2])
            ))
        )
    return xv_end

def labudde_greenspan_timestepper(xv_start):
    xv_end = timestepping.schemes.labudde_greenspan(
        xv_start,
        dt,
        F,
        V
    )
    return xv_end

def crank_nicolson_timestepper(xv_start):
    xv_end = timestepping.schemes.crank_nicolson(
        xv_start,
        dt,
        lambda xv: np.hstack((
            xv[2:4],
            F(xv[0:2])
            ))
        )
    return xv_end

def gauss_timestepper(xv_start):
    xv_end = timestepping.schemes.gauss(
        xv_start,
        dt,
        lambda xv: np.hstack((
            xv[2:4],
            F(xv[0:2])
            )),
        stages,
        res=res
        )
    return xv_end

def cohen_hairer_timestepper(xv_start):
    xv_end = timestepping.schemes.cohen_hairer(
        xv_start,
        dt,
        lambda xv: Pois,
        lambda xv: H_grad(xv[0:2], xv[2:4]),
        stages,
        res=res,
        snes_tol=1e-25
        )
    return xv_end

def andrews_farrell_L_timestepper(xv_start):
    xv_end = timestepping.schemes.andrews_farrell(
        xv_start,
        dt,
        lambda xv: Pois,
        lambda xv: H_grad(xv[0:2], xv[2:4]),
        lambda xv: np.array([L_grad(xv[0:2], xv[2:4])]),
        1,
        stages,
        res=res
        )
    return xv_end

def andrews_farrell_A_timestepper(xv_start):
    xv_end = timestepping.schemes.andrews_farrell(
        xv_start,
        dt,
        lambda xv: Pois,
        lambda xv: H_grad(xv[0:2], xv[2:4]),
        lambda xv: A_grad(xv[0:2], xv[2:4]),
        2,
        stages,
        res=res
        )
    return xv_end



# Define timestepper dictionary
timestepper_dict = {
    "explicit_euler": explicit_euler_timestepper,
    "rk4": rk4_timestepper,

    "implicit_euler": implicit_euler_timestepper,
    "implicit_midpoint": implicit_midpoint_timestepper,
    "labudde_greenspan": labudde_greenspan_timestepper,
    "crank_nicolson": crank_nicolson_timestepper,
    "gauss": gauss_timestepper,
    "cohen_hairer": cohen_hairer_timestepper,
    "andrews_farrell_L": andrews_farrell_L_timestepper,
    "andrews_farrell_A": andrews_farrell_A_timestepper,
    "andrews_farrell": andrews_farrell_A_timestepper,
    "avfet": andrews_farrell_A_timestepper,
}



# Run solve loop
if scheme in timestepper_dict:
    start_time = time.time()
    xv = timestepping.solve_loop(
        np.hstack([x_0, v_0]),
        timestepper_dict[scheme],
        N
    )
    print(f"Solve loop ran in {time.time() - start_time} seconds")
else:
    raise KeyError("Scheme ", scheme, " is not in timestepper_dict")



'''
Data retrieval
'''
#orbit_completion_index = np.where(np.diff(np.sign(xv[:, 1])) > 0)[0][1]  # Time index where y first becomes positive again
#orbit = dt/res * (orbit_completion_index + 1 / (1 - xv[orbit_completion_index + 1, 1]/xv[orbit_completion_index, 1]))
#print(f"Estimated orbit duration:", orbit)
#print(f"Error in orbit duration:", abs(2*np.pi - orbit))

#two_pi_index = res * 2*np.pi / dt
#return_x = (
#    (1 - two_pi_index + math.floor(two_pi_index)) * xv[math.floor(two_pi_index), 0:2]
#  + (two_pi_index - math.floor(two_pi_index)) * xv[math.floor(two_pi_index) + 1, 0:2]
#)
#print(f"Estimated position at 2pi:", return_x)
#print(f"Error in position at 2pi:", np.linalg.norm(return_x - x_0))



'''
Plotting
'''
# Create axes
fig, axes = plt.subplots(2, 2)

# Plot positions
axes[0,0].plot(xv[:,0], xv[:,1], color="#4C72B0", label="Planet")  # Planet
theta_arr = np.arange(0, 2*np.pi * (1 + 1/100), 2*np.pi / 100)  # Ideal orbit
L_0 = L(x_0, v_0)
r_arr = 1 / (1/L_0**2 + (1/x_0[0]  - 1/L_0**2)*np.cos(theta_arr))
axes[0,0].plot(r_arr * np.cos(theta_arr),r_arr * np.sin(theta_arr), linestyle="dashed", color="#4C72B0", label="Exact orbit")
A_arr = np.array([A(xv_[0:2], xv_[2:4]) for xv_ in xv])  # Runge--Lenz vector
axes[0,0].plot(A_arr[:,0], A_arr[:,1], color="#C44E52", label="Runge–Lenz vector")
axes[0,0].add_patch(pat.Circle((0, 0), 0.05, color="#000000", fill=True))
# Edit plot
axes[0,0].set_xlabel("x")
axes[0,0].set_ylabel("y")
axes[0,0].set_title("Position")
axes[0,0].axis("equal")
axes[0,0].legend()

t_arr = dt/res * np.arange(N*res + 1)

# Plot angles
theta_arr = np.array([theta(xv_[0:2], xv_[2:4]) for xv_ in xv])  # Energies
axes[0,1].plot(t_arr, theta_arr, color="#C44E52", label="Orbital angle")
# Edit plot
axes[0,1].set_xlabel("t")
axes[0,1].set_title("Orbital angle")
axes[0,1].legend()

# Plot energies
H_arr = np.array([H(xv_[0:2], xv_[2:4]) for xv_ in xv])  # Energies
axes[1,0].plot(t_arr, H_arr, color="#C44E52", label="Energy")
# Edit plot
axes[1,0].set_xlabel("t")
axes[1,0].set_title("Energy")
axes[1,0].legend()

# Plot angular momenta
L_arr = np.array([L(xv_[0:2], xv_[2:4]) for xv_ in xv])  # Angular momenta
axes[1,1].plot(t_arr, L_arr, color="#C44E52", label="Angular momentum")
axes[1,1].set_xlabel("t")
axes[1,1].set_title("Angular momentum")
axes[1,1].legend()

# Tidy plot
plt.tight_layout()
# Show plot
plt.show()
