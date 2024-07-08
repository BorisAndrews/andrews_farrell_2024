'''
Default initial guess for implicit schemes
'''
def default_initial_guess(x_start, dt, f):
    from .runge_kutta import rk4
    return rk4(x_start, dt, f)
