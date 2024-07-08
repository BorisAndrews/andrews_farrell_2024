'''
Collocation Runge--Kutta timestepping schemes

N.B. These are implemented using a basis of lagrange basis functions for x, not for dx/dt as would be required to derive the scheme in its typical Runge--Kutta form --- this is done to improve sparsity in the discretisation for faster solves/easier implementation
'''
from .default_initial_guess import *



'''
Dictionaries
'''
coll_rk_coeffs_dict = {}
lagrange_eval_coeffs_dict = {}
lagrange_eval_coeffs_grid_dict = {}



'''
Actual schemes
'''
# Gauss
def gauss(x_start, dt, f, s, res=1):
    import numpy.polynomial.legendre as legendre
    return coll_rk(x_start, dt, f,
        0.5 * (legendre.leggauss(s)[0] + 1),
        res
    )



# Radau IIA
def radau_iia(x_start, dt, f, s, res=1):
    import numpy as np
    return coll_rk(x_start, dt, f,
        np.roots(np.polyder([*tuple(np.poly([1]*s))] + [0]*(s-1), s-1))[::-1],
        res
    )



'''
General scheme functions
'''
# Check certain collocation points are valid
def check_c_valid(c):
    import numpy as np

    if np.unique(c).size < c.size:
        raise ValueError("There appears to be a duplicate collocation point in c.")
    elif 0 in c:
        raise ValueError("Collocation Runge--Kutta methods not yet implemented for schemes with collocation points at 0.")
    



# Evaluate/retrieve the coefficients for the collocation method
def get_coll_rk_coeffs(c):
    '''
    Returns a matrix (a_ij) with the following property. Let (l_i) be the Lagrange basis at (0, c); a_ij is the derivative of l_i at c_j.
    '''

    # Create hashable c
    c_tuple = tuple(c)

    # If we've already calculated the coefficients for this c...
    if c_tuple in coll_rk_coeffs_dict:
        return coll_rk_coeffs_dict[c_tuple]
    else:
        import numpy as np

        # Retieve order
        s = c.size

        # Check collocation points are valid
        check_c_valid(c)

        # Create matrix
        coll_rk_coeffs = np.zeros([s+1, s+1], dtype=float)
        
        # Diagonal entries
        coll_rk_coeffs[0,0] = - np.sum(1/c)
        for i in range(s):
            c_copy = c.copy()
            c_copy[i] = 0
            coll_rk_coeffs[i+1,i+1] = sum(1/(c[i] - c_copy))
        # Off-diagonal entries
        for i in range(s):
            coll_rk_coeffs[0,i+1] = - np.prod(1 - c[i]/np.delete(c, i))/c[i]
            coll_rk_coeffs[i+1,0] = 1/np.prod(1 - c[i]/np.delete(c, i))/c[i]
            for j in range(s):
                if j != i:
                    coll_rk_coeffs[i+1,j+1] = np.prod(np.delete(c[j] - c, [i,j]))/np.prod(np.delete(c[i] - c, i)) * c[j]/c[i]

        # Save to dictionary
        coll_rk_coeffs_dict[c_tuple] = coll_rk_coeffs

        return coll_rk_coeffs



# Evaluate x(t) given x_0 and x(t_i) for each i
def get_lagrange_eval_coeffs(c, t):
    '''
    Returns a vecotr (b_i) with the following property. Let (l_i) be the Lagrange basis at (0, c); l_i(t) = b_i.
    '''

    # Create hashable c
    c_tuple = tuple(c)

    # If we've already calculated the coefficients for this c at this point...
    if (c_tuple, t) in lagrange_eval_coeffs_dict:
        return lagrange_eval_coeffs_dict[(c_tuple, t)]
    else:
        import numpy as np

        # Retrieve order
        s = c.size

        # Check collocation points are valid
        check_c_valid(c)

        # Create vector
        lagrange_eval_coeffs = np.zeros(s+1, dtype=float)

        # 0 entry
        lagrange_eval_coeffs[0] = np.prod(1 - t/c)
        # Other entries
        for i in range(s):
            c_copy = c.copy()
            c_copy[i] = 0
            lagrange_eval_coeffs[i+1] = np.prod((t - c_copy)/(c[i] - c_copy))

        # Save to dictionary
        lagrange_eval_coeffs_dict[(c_tuple, t)] = lagrange_eval_coeffs

        return lagrange_eval_coeffs



# Evalute/retrieve the coeffs for evaluating the Lagrange basis at a given resolution
def get_lagrange_eval_coeffs_grid(c, res=1):
    '''
    Returns a matrix (b_ij) with the following property. Let (l_i) be the Lagrange basis at (0, c); l_i((j+1)/res) = b_ij.
    '''

    # Create hashable c
    c_tuple = tuple(c)

    # If we've already calculated the coefficients for this c at this resolution...
    if (c_tuple, res) in lagrange_eval_coeffs_grid_dict:
        return lagrange_eval_coeffs_grid_dict[(c_tuple, res)]
    else:
        import numpy as np

        # Retieve order
        s = c.size

        # Check collocation points are valid
        check_c_valid(c)

        # Create array
        lagrange_eval_coeffs_grid = np.zeros([s+1, res], dtype=float)

        # Fill entries
        for i in range(res):
            lagrange_eval_coeffs_grid[:, i] = get_lagrange_eval_coeffs(c, (i+1)/res)

        # Save to dictionary
        lagrange_eval_coeffs_grid_dict[(c_tuple, res)] = lagrange_eval_coeffs_grid

        return lagrange_eval_coeffs_grid
    



# Generic collocation Runge--Kutta method
def coll_rk(x_start, dt, f, c, res=1, initial_guess=default_initial_guess):
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
    s = c.size

    # Normalise input
    x_start = x_start.astype(float)

    # Get coeffs
    coll_rk_coeffs = get_coll_rk_coeffs(c)
    lagrange_eval_coeffs_grid = get_lagrange_eval_coeffs_grid(c, res)

    # Define residual
    def res_func(snes, x_mat_PETSc_, res_PETSc_):
        x_mat_arr_ = x_mat_PETSc_.getArray(readonly=True)
        res_arr = res_PETSc_.getArray()
        # At each collocation point...
        for i in range(s):
            f_arr = f(x_mat_arr_[i*n : (i+1)*n])
            for j in range(n):
                res_arr[i*n + j] = dt * f_arr[j] - (
                    coll_rk_coeffs[0, i+1] * x_start[j]
                  + np.sum([coll_rk_coeffs[k+1, i+1] * x_mat_arr_[k*n+j] for k in range(s)])
                )

    # Create PETSc vector for dx/dt-like term
    x_mat_PETSc = PETSc.Vec().create()
    x_mat_PETSc.setSizes(n * s)
    x_mat_PETSc.setFromOptions()
    dx_initial_guess = initial_guess(x_start, dt, f) - x_start
    x_mat_PETSc.setArray(np.array([x_start + c[i] * dx_initial_guess for i in range(s)]))
            
    # Create SNES
    solver = PETSc.SNES().create()
    x_mat_PETSc_dup = x_mat_PETSc.duplicate()
    solver.setFunction(res_func, x_mat_PETSc_dup)
    solver.setFromOptions()
    
    # Run solve
    solver.solve(None, x_mat_PETSc)
    x_mat_arr = x_mat_PETSc.getArray().reshape(s, n)

    # Destroy PETSc objects
    x_mat_PETSc.destroy()
    x_mat_PETSc_dup.destroy()
    solver.destroy()

    # Find output
    x_out = np.outer(
        lagrange_eval_coeffs_grid[0, :],
        x_start
    ) + np.sum([np.outer(
        lagrange_eval_coeffs_grid[i+1, :],
        x_mat_arr[i, :]
    ) for i in range(s)], axis=0)

    return x_out
