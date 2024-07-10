'''
PETSc import
'''
# Get chosen PETSc source
import avfet_modules.terminal_options as terminal_options
petsc_source = terminal_options.get("petsc", type=str, default="none")

# If PETSc requested...
if petsc_source != "none":
    # Import (#1) from desired location...
    if petsc_source == "petsc4py":
        import petsc4py
        import sys
        petsc4py.init(sys.argv)
    elif petsc_source == "firedrake":
        import firedrake.petsc as petsc4py
    elif petsc_source != None:
        raise KeyError("Unknown PETSc source.")
    # Import (#2)
    from petsc4py import PETSc
    # Initialise
    PETSc.Sys.popErrorHandler()



'''
Imports
'''
from .solve_loop import *



'''
Sub-module imports
'''
import avfet_modules.timestepping.schemes as schemes
