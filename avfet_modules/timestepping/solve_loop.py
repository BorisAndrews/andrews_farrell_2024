'''
Solve loop
'''
def solve_loop(x_0, timestepper, N):
    import numpy as np

    x = x_0
    x_start = x_0
    for _ in range(N):
        x_data = timestepper(x_start)
        x = np.vstack([x, x_data])
        x_start = x[-1]
    
    return x