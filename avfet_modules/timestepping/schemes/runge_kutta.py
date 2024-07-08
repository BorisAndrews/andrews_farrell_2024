'''
Runge--Kutta timestepping schemes
'''
from .default_initial_guess import *



'''
Actual Runge--Kutta schemes
'''
# (Order 4, but must be defined first for initial guesses...)
def rk4(x_start, dt, f):
    import numpy as np
    return explicit_rk(
        x_start, dt, f,
        np.array([
            [0,   0,   0, 0],
            [1/2, 0,   0, 0],
            [0,   1/2, 0, 0],
            [0,   0,   1, 0]
            ]),
        np.array(
            [1/6, 1/3, 1/3, 1/6]
            )
        )

# Order 1
def explicit_euler(x_start, dt, f):
    import numpy as np
    return explicit_rk(
        x_start, dt, f,
        np.array([[0]]),
        np.array([1]),
        )

def implicit_euler(x_start, dt, f, initial_guess=default_initial_guess):
    import numpy as np
    return implicit_rk(
        x_start, dt, f,
        np.array([[1]]),
        np.array([1]),
        initial_guess
        )

# Order 2
def implicit_midpoint(x_start, dt, f, initial_guess=default_initial_guess):
    import numpy as np
    return implicit_rk(
        x_start, dt, f,
        np.array([[1/2]]),
        np.array([1]),
        initial_guess
        )



'''
Generic Runge--Kutta scheme functions
'''
# Generic Runge--Kutta scheme
def rk(x_start, dt, f, A, b):
    # Check if scheme is explicit, i.e. A is fully lower diagonal
    explicit = True
    for (i, row) in enumerate(A):
        for (j, elem) in enumerate(row):
            if (i <= j) and (elem != 0):
                explicit = False
                break
        if not explicit:
            break
        
    # Run the corresponding function
    if explicit:
        return explicit_rk(x_start, dt, f, A, b)
    else:
        return implicit_rk(x_start, dt, f, A, b)



# Explicit Runge--Kutta scheme
def explicit_rk(x_start, dt, f, A, b):
    import numpy as np

    # Retrieve array sizes
    n = x_start.size
    s = b.size

    # Normalise input
    x_start = x_start.astype(float)
    
    # Find dx/dt-like terms
    dx_dt_mat = np.zeros([s, n], dtype=float)
    for i in range(s):
        x_mid = x_start
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0:
                x_mid = x_mid + a_ij * dt * dx_dt_mat[j, :]
        dx_dt_mat[i, :] = f(x_mid)

    return x_start + dt * np.dot(b, dx_dt_mat)



# Implicit Runge--Kutta scheme (in PETSc)
def implicit_rk(x_start, dt, f, A, b, initial_guess=default_initial_guess):
    import numpy as np

    # Test PETSc is imported
    try:
        PETSc.__spec__
    except NameError:
        try:
            import petsc4py
            import sys
            petsc4py.init(sys.argv)
            from petsc4py import PETSc
        except ImportError:
            raise ImportError("Failed to import PETSc, which is required to run implicit schemes. Either specify a destination from which to import PETSc through '-petsc' (e.g. '-petsc firedrake') or use an explicit scheme (e.g. '-scheme rk4').")

    # Retrieve array sizes
    n = x_start.size
    s = b.size

    # Normalise input
    x_start = x_start.astype(float)

    # Define residual
    def res_func(snes, dx_dt_mat_PETSc_, res_PETSc_):
        dx_dt_mat_arr_ = dx_dt_mat_PETSc_.getArray(readonly=True)
        res_arr = res_PETSc_.getArray()
        # At each collocation point...
        for i in range(s):
            # Evaluate collocation point
            x_mid = x_start
            for j in range(s):
                a_ij = A[i, j]
                if a_ij != 0:
                    x_mid = x_mid + a_ij * dt * dx_dt_mat_arr_[j*n : (j+1)*n]
            # Add residual contribution at that point
            f_arr = f(x_mid)
            for j in range(n):
                res_arr[i*n + j] = f_arr[j] - dx_dt_mat_arr_[i*n + j]

    # Create PETSc vector for dx/dt-like term
    dx_dt_mat_PETSc = PETSc.Vec().create()
    dx_dt_mat_PETSc.setSizes(n * s)
    dx_dt_mat_PETSc.setFromOptions()
    dx_dt_initial_guess = (initial_guess(x_start, dt, f) - x_start) / dt
    dx_dt_mat_PETSc.setArray(np.array([dx_dt_initial_guess for _ in range(s)]))
            
    # Create SNES
    solver = PETSc.SNES().create()
    dx_dt_mat_PETSc_dup = dx_dt_mat_PETSc.duplicate()
    solver.setFunction(res_func, dx_dt_mat_PETSc_dup)
    solver.setFromOptions()
    
    # Run solve
    solver.solve(None, dx_dt_mat_PETSc)
    dx_dt_mat_arr = dx_dt_mat_PETSc.getArray()

    # Destroy PETSc objects
    dx_dt_mat_PETSc.destroy()
    dx_dt_mat_PETSc_dup.destroy()
    solver.destroy()

    # Find output
    x_end = x_start
    for i in range(s):
        x_end += b[i] * dt * dx_dt_mat_arr[(i*n):((i+1)*n)]

    return x_end
