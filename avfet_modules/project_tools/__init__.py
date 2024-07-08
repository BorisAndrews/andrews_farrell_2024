'''
Need to tidy this whole file up:
- Don't import all of firedrake
- Add documentation
'''

from firedrake import *

def project_op_free(target, U, *op_tup, **kwargs):
    # Create Lagrange multiplier space
    UP = MixedFunctionSpace([U, *[op[1] for op in op_tup]])

    # Create functions for projection
    up = Function(UP)
    (u, *p_arr) = split(up)
    (v, *q_arr) = split(TestFunction(UP))

    # Create Lagrange multiplier residual
    F = (
        inner(u - target, v)*dx
      - sum([
            op[0](v, p) + op[0](u, q)
            for (op, p, q) in zip(op_tup, p_arr, q_arr)
        ])
    )

    # Solve
    if "solver_parameters" in kwargs:
        solve(F == 0, up, solver_parameters=kwargs["solver_parameters"])
    else:
        solve(F == 0, up)
        

    # Assign solution
    u_ = Function(U)
    u_.assign(up.sub(0))

    return u_
