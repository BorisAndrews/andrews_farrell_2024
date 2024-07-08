'''
Weight collocation timestepping schemes
'''
from .default_initial_guess import *



'''
Actual schemes
'''
# Crank--Nicolson
def crank_nicolson(x_start, dt, f, initial_guess=default_initial_guess):
    return implicit_1_stage(
        x_start, dt, f,
        lambda x_start_, x_end: 0.5*f(x_start_) + 0.5*f(x_end)
        )



'''
General scheme functions
'''
# 1-stage implicit schemes
def implicit_1_stage(x_start, dt, f, f_approx, initial_guess=default_initial_guess):
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
    
    # Retrieve number of variables
    n = x_start.size

    # Create PETSc vector for x_end
    x_end_PETSc = PETSc.Vec().create()
    x_end_PETSc.setSizes(n)
    x_end_PETSc.setFromOptions()
    x_end_PETSc.setArray(initial_guess(x_start, dt, f))

    # Create SNES
    def res_func(snes, x_end_PETSc_, res_PETSc_):
        x_end_arr = x_end_PETSc_.getArray(readonly=True)
        res_arr = res_PETSc_.getArray()
        f_arr = f_approx(x_start, x_end_arr)
        for i in range(len(x_end_arr)):
            res_arr[i] = (x_end_arr[i] - x_start[i]) - dt * f_arr[i]
            
    solver = PETSc.SNES().create()
    x_end_PETSc_dup = x_end_PETSc.duplicate()
    solver.setFunction(res_func, x_end_PETSc_dup)
    solver.setFromOptions()
    
    # Run solve
    solver.solve(None, x_end_PETSc)

    # Retrieve output
    x_end = x_end_PETSc.getArray() 

    # Destroy PETSc objects
    x_end_PETSc.destroy()
    x_end_PETSc_dup.destroy()
    solver.destroy()

    return x_end
