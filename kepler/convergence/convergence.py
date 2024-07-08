import subprocess
import numpy as np

# Arrays
stages_arr = [i for i in range(1, 5)]
dt_arr = [2*np.pi * 2**(-i) for i in range(5, 13)]

# Iterate over the timestep sizes and run the script with each value
for stages in stages_arr:
    for dt in dt_arr:
        subprocess.run([
            "python", "kepler/convergence/kepler_error.py",
            "--dt", str(dt),
            "--stages", str(stages),
        ])
