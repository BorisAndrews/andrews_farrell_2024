'''
Imports
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import avfet_modules.terminal_options as terminal_options



'''
General purpose functions: Plotting
'''
# Evaluate u at a given x
def u_eval_x(u, x, L):
    u_len = len(u)  # Number of elements in u
    dx = 2 * L / u_len  # Interval length
    x_ind = int(np.floor(x / dx))  # Index of interval in which x lies
    delta_x = x - x_ind * dx  # Displacement of x into interval

    out = 0
    out += u[(2*x_ind + 0) % u_len] * (2*delta_x + dx) * (delta_x - dx)**2
    out += u[(2*x_ind + 1) % u_len] * delta_x * (delta_x - dx)**2
    out -= u[(2*x_ind + 2) % u_len] * (2*delta_x - 3*dx) * delta_x**2
    out += u[(2*x_ind + 3) % u_len] * (delta_x - dx) * delta_x**2

    return out / dx**3

# Evaluate u at a given vector x
def u_eval_xvec(u, x, L):
    out = np.zeros(len(x), dtype=float)

    for (i, x_) in enumerate(x):
        out[i] = u_eval_x(u, x_, L)
    
    return out

# Evaluate u at a given vector x and time t
def u_eval_xvec_t(u, x, L, t, dt):
    t_ind = int(np.floor(t / dt))  # Index of time interval in which t lies
    delta_t = t / dt - t_ind  # Percentage displacement of t into interval
    
    if t_ind >= u.shape[0]-1:
        return u_eval_xvec(u[-1], x, L)
    else:
        return (1 - delta_t) * u_eval_xvec(u[t_ind], x, L) + delta_t * u_eval_xvec(u[t_ind+1], x, L)

# Sech
def sech(x):
    return 2 / (np.exp(x) + np.exp(- x))

# Soliton
def soliton(x, x_0, c):
    return 3 * (c-1) * sech(0.5 * np.sqrt(1 - 1/c) * (x-x_0))**2



'''
Parameters
'''
L = terminal_options.get("L", type=float, default=100)  # Grid length
dt = terminal_options.get("dt", type=float, default=1)  # Timestep (in data) (2 for the bad simulations / 1 for the better simualations)

t_start = terminal_options.get("t_start", type=float, default=0)  # Start time of simulation
t_end = terminal_options.get("t_end", type=float, default=20000)  # End time of simulation
dt_plot = terminal_options.get("dt_plot", type=float, default=dt)  # Timestep (for plotting)
nx_plot = terminal_options.get("nx_plot", type=int, default=500)  # Mesh number (for plotting)

cam_speed = terminal_options.get("cam_speed", type=float, default=0)  # Camera speed (1.618034 is accurate)

save_file = terminal_options.get("save_file", type=str, default=False)  # If/where to save the exported mp4

dir = terminal_options.get("dir", type=str, default="output/bbm/avfet/")  # Directory in which to find data



'''
Animation
'''
# Load data from the files
u = np.loadtxt(dir + "u.txt")
energy = np.loadtxt(dir + "energy.txt")

# Create a figure and axes with different widths for the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

# Create empty line objects for each subplot
if cam_speed != 0:
    line1ghost, = ax1.plot([], [], color="#4C72B0", linestyle="dotted", alpha=0.5)
line1, = ax1.plot([], [], color="#4C72B0")
line2, = ax2.plot([], [], color="#C44E52")  # Set the color of the line to red

# Set the axis limits for the first subplot (u data)
ax1.set_xlim(0, L)
ax1.set_ylim(np.nanmin(u), np.nanmax(u))

# Set the axis limits for the second subplot (energy data)
ax2.set_xlim(0, len(energy) * dt)  # Set x-axis limits based on the length of energy data
ax2.set_ylim(0, 1.1 * energy[0])  # Set y-axis limits based on the energy data

# Function to update the plot for each frame
x_vec = L / nx_plot * np.arange(nx_plot + 1)  # Vector of x coordinates for left plot

def update(frame):
    t = frame * dt_plot  # Time in animation
    ind = round(t / dt)  # Approximate corresponding index

    if cam_speed == 0:
        line1.set_data(x_vec, u_eval_xvec_t(u, x_vec, L, t, dt))
    else:
        line1ghost.set_data(x_vec, soliton(x_vec, L/2, (1+np.sqrt(5))/2))
        line1.set_data(x_vec, u_eval_xvec_t(u, cam_speed*t + x_vec, L, t, dt))
    line2.set_data(dt * np.arange(0, len(energy[:ind+1])), energy[:ind+1])  # Update energy data up to current frame
    
    # Add dashed horizontal lines
    ax1.axhline(y=0, color="#4C72B0", linestyle='--')  # Set color to black and linestyle to dashed
    ax2.axhline(y=energy[0], color="#C44E52", linestyle='--')  # Set color to black and linestyle to dashed
    
    return line1, line2

# Create the animation
if t_end == t_start:
    ani = FuncAnimation(
        fig,
        update,
        frames=range(round(t_start / dt_plot), round(u.shape[0] * dt / dt_plot)),
        interval=20*dt_plot,
        blit=True
    )
else:
    ani = FuncAnimation(
        fig,
        update,
        frames=range(round(t_start / dt_plot), round(t_end / dt_plot)),
        interval=20*dt_plot,
        blit=True
    )

# Set labels and title for each subplot
ax1.set_xlabel('x')
ax1.set_title('u')

ax2.set_xlabel('t')
ax2.set_title('H')

# Save animation
if save_file:
    ani.save(save_file, fps=30, extra_args=["-vcodec", "libx264"])

plt.tight_layout()
plt.show()
