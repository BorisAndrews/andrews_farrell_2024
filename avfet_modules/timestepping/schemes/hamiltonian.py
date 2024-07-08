'''
Specialist integrators for Hamiltonian systems
'''
from .default_initial_guess import *
from .collocation_runge_kutta import *
from .weighted_collocation import implicit_1_stage
import numpy as np



'''
Actual schemes
'''
# Energy-momentum conserving method of LaBudde & Greenspan (1974)
def labudde_greenspan(x_start, dt, F, H, snes_tol=1e-15, initial_guess=default_initial_guess):
    # Retrieve problem size
    n = x_start.size
    if n % 2 != 0:
        raise ValueError("LaBudde--Greenspan only well-defined in canonical coordinates")
    n_half = round(n/2)

    # dx/dt
    f = lambda x: np.hstack((
        x[n_half:n],
        F(x[0:n_half])
        ))
    
    # LaBudde--Greenspan approximation to dx/dt
    f_approx = lambda x_start, x_end: np.hstack((
        1/2 * (x_start[n_half:n] + x_end[n_half:n]),
        - (H(x_end[0:n_half]) - H(x_start[0:n_half])) / (sum(x_end[0:n_half]**2) - sum(x_start[0:n_half]**2)) * (x_end[0:n_half] + x_start[0:n_half])
        ))
    
    return implicit_1_stage(x_start, dt, f, f_approx)

    



# Energy-conserving scheme of Cohen & Hairer (2011) (using Gauss-Legendre quadrature points)
def cohen_hairer(x_start, dt, Pois, H_grad, s, res=1, quad_pts_min=10, snes_tol=1e-15, initial_guess=default_initial_guess):
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

    # Retrieve problem size
    n = x_start.size

    # Normalise input
    x_start = x_start.astype(float)

    # Set collocation points as GL points
    (c_nodes, c_weights) = np.polynomial.legendre.leggauss(s)
    c_nodes = 0.5 * (c_nodes + 1)
    c_weights = 0.5 * c_weights

    # Get GL quadrature points
    (quad_nodes, quad_weights) = np.polynomial.legendre.leggauss(max(2*s, quad_pts_min))
    quad_nodes = 0.5 * (quad_nodes + 1)
    quad_weights = 0.5 * quad_weights

    # Get coeffs
    coll_rk_coeffs = get_coll_rk_coeffs(c_nodes)
    lagrange_eval_coeffs_grid = get_lagrange_eval_coeffs_grid(c_nodes, res)

    # Define evaluation function
    def x(t, x_mat_arr_):
        lagrange_eval_coeffs = get_lagrange_eval_coeffs(c_nodes, t)
        x = lagrange_eval_coeffs[0] * x_start + sum([
            lagrange_eval_coeff * x_mat_arr_[i*n : (i+1)*n]
            for (i, lagrange_eval_coeff) in enumerate(lagrange_eval_coeffs[1:])
        ])
        return x

    # Define residual
    def res_func(snes, x_mat_PETSc_, res_PETSc_):
        x_mat_arr_ = x_mat_PETSc_.getArray(readonly=True)
        res_arr = res_PETSc_.getArray()
        # At each collocation point...
        for i in range(s):
            Pois_arr = Pois(x_mat_arr_[i*n : (i+1)*n])

            c_nodes_red = np.delete(c_nodes, i)  # Collocation nodes with node_i removed
            l_i = lambda t: np.prod([t - c_nodes_red])/np.prod([c_nodes[i] - c_nodes_red])  # l_i(t)
            H_grad_arr = 1/c_weights[i] * sum([
                quad_weight  # quad_weight
              * l_i(quad_node)  # l_i(quad_node)
              * H_grad(x(quad_node, x_mat_arr_))  # H_grad(quad_node)
                for (quad_node, quad_weight) in zip(quad_nodes, quad_weights)
            ])

            for j in range(n):
                res_arr[i*n + j] = dt * np.dot(H_grad_arr, Pois_arr[:, j]) - (
                    coll_rk_coeffs[0, i+1] * x_start[j]
                  + np.sum([coll_rk_coeffs[k+1, i+1] * x_mat_arr_[k*n+j] for k in range(s)])
                )

    # Create PETSc vector for dx/dt-like term
    x_mat_PETSc = PETSc.Vec().create()
    x_mat_PETSc.setSizes(n*s)
    x_mat_PETSc.setFromOptions()
    dx_initial_guess = initial_guess(x_start, dt, lambda x: np.dot(H_grad(x), Pois(x))) - x_start
    x_mat_PETSc.setArray(np.array([x_start + c_nodes[i] * dx_initial_guess for i in range(s)]))
            
    # Create SNES
    solver = PETSc.SNES().create()
    x_mat_PETSc_dup = x_mat_PETSc.duplicate()
    solver.setFunction(res_func, x_mat_PETSc_dup)
    solver.setTolerances(atol=snes_tol, rtol=snes_tol)
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



# Fully conservative scheme of Andrews & Farrell (2024) (as a modification of Gauss--Legendre)
def andrews_farrell(x_start, dt, Pois, H_grad, N_grad, P, s, res=1, quad_pts_min=10, snes_tol=1e-15, initial_guess=default_initial_guess):
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

    # Retrieve problem size
    n = x_start.size

    # Normalise input
    x_start = x_start.astype(float)

    # Set collocation points as GL points
    (c_nodes, c_weights) = np.polynomial.legendre.leggauss(s)
    c_nodes = 0.5 * (c_nodes + 1)
    c_weights = 0.5 * c_weights

    # Get GL quadrature points
    (quad_nodes, quad_weights) = np.polynomial.legendre.leggauss(max(2*s, quad_pts_min))
    quad_nodes = 0.5 * (quad_nodes + 1)
    quad_weights = 0.5 * quad_weights

    # Get coeffs
    coll_rk_coeffs = get_coll_rk_coeffs(c_nodes)
    lagrange_eval_coeffs_grid = get_lagrange_eval_coeffs_grid(c_nodes, res)

    # Define evaluation function
    def x(t, x_mat_arr_):
        lagrange_eval_coeffs = get_lagrange_eval_coeffs(c_nodes, t)
        x = lagrange_eval_coeffs[0] * x_start + sum([
            lagrange_eval_coeff * x_mat_arr_[i*n : (i+1)*n]
            for (i, lagrange_eval_coeff) in enumerate(lagrange_eval_coeffs[1:])
        ])
        return x

    # Define residual
    def res_func(snes, x_mat_PETSc_, res_PETSc_):
        x_mat_arr_ = x_mat_PETSc_.getArray(readonly=True)
        res_arr = res_PETSc_.getArray()
        # At each collocation point...
        for i in range(s):
            Pois_arr = Pois(x_mat_arr_[i*n : (i+1)*n])  # Poisson matrix at c_i (B(c_i))

            c_nodes_red = np.delete(c_nodes, i)  # c with c_i removed
            l_i = lambda t: np.prod([t - c_nodes_red])/np.prod([c_nodes[i] - c_nodes_red])  # l_i(t)
            
            H_grad_arr = 1/c_weights[i] * sum([  # \tilde{H_grad} at c_i
                quad_weight  # quad_weight
              * l_i(quad_node)  # l_i(quad_node)
              * H_grad(x(quad_node, x_mat_arr_))  # H_grad(quad_node)
                for (quad_node, quad_weight) in zip(quad_nodes, quad_weights)
            ])

            N_grad_arr = 1/c_weights[i] * sum([  # \tilde{N_grad} at c_i
                quad_weight  # quad_weight
              * l_i(quad_node)  # l_i(quad_node)
              * N_grad(x(quad_node, x_mat_arr_))  # N_grad(quad_node)
                for (quad_node, quad_weight) in zip(quad_nodes, quad_weights)
            ])

            mat_to_inv = np.array([  # Matrix to invert to solve for lambda's
                np.array([
                    np.dot(N_grad_arr[p, :], N_grad_arr[q, :]) * sum(H_grad_arr**2)
                  - np.dot(N_grad_arr[p, :], H_grad_arr) * np.dot(N_grad_arr[q, :], H_grad_arr)
                    for p in range(P)
                ])
                for q in range(P)
            ])
            vec_to_inv = np.array([  # Vector to invert to solve for lambda's
                np.dot(np.dot(H_grad_arr, Pois_arr), N_grad_arr[p, :])
                for p in range(P)
            ])
            lamb_arr = np.linalg.solve(mat_to_inv, vec_to_inv)  # lambda's
            Pois_pert_arr = sum([  # Perturbation to Poisson matrix at c_i (\tilde{delta B}(c_i))
                lamb_arr[q] * (
                    np.outer(H_grad_arr, N_grad_arr[q, :])
                  - np.outer(N_grad_arr[q, :], H_grad_arr) 
                )
                for q in range(P)
            ])

            for j in range(n):
                res_arr[i*n + j] = dt * np.dot(H_grad_arr, Pois_arr[:, j] - Pois_pert_arr[:, j]) - (
                    coll_rk_coeffs[0, i+1] * x_start[j]
                  + np.sum([coll_rk_coeffs[k+1, i+1] * x_mat_arr_[k*n+j] for k in range(s)])
                )

    # Create PETSc vector for dx/dt-like term
    x_mat_PETSc = PETSc.Vec().create()
    x_mat_PETSc.setSizes(n*s)
    x_mat_PETSc.setFromOptions()
    dx_initial_guess = initial_guess(x_start, dt, lambda x: np.dot(H_grad(x), Pois(x))) - x_start
    x_mat_PETSc.setArray(np.array([x_start + c_nodes[i] * dx_initial_guess for i in range(s)]))
            
    # Create SNES
    solver = PETSc.SNES().create()
    x_mat_PETSc_dup = x_mat_PETSc.duplicate()
    solver.setFunction(res_func, x_mat_PETSc_dup)
    solver.setTolerances(atol=snes_tol, rtol=snes_tol)
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
