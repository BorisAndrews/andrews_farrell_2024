'''
numpy.polynomial.chebyshev, with the missing ingredients: projection and recursive multiplication
'''



from numpy.polynomial.chebyshev import *

import numpy as np
import numpy.polynomial.polyutils as pu



chebdualmat_dict = {}
chebprojvec_dict = {}



def basis_vec(order):
    """
    Simple basis vector
    
    Parameters
    ----------
    order : int
        Order of output basis vector
    
    Returns
    -------
    ... : ndarray
        With "order" 0's, succeeded by a 1
    
    Examples
    --------
    >>> cheb.basis_vec(3)
    array([0., 0., 0., 1.])
    """
    return np.append(np.zeros(order), 1)



def chebmulrec(*c_array):
    """
    Multiply an array of Chebyshev series by each other, recursively.

    Returns the product of an array of Chebyshev series `c_array`.

    Parameters
    ----------
    *c_array : tuple of array_likes
        Arbitrarily long tuple of 1-D arrays of Chebyshev series
        coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    Examples
    --------
    >>> cheb.chebmulrec([0, 1], [0, 1], [0, 1])
    array([0.  , 0.75, 0.  , 0.25])
    """
    # Base case: If the array has length, unpack...
    if len(c_array) == 1:
        out = c_array[0]
    # Recursive case: ...otherwise, define product recursively via chebmul
    else:
        out = chebmul(c_array[0], chebmulrec(*c_array[1:]))

    # Remove any trailing 0's
    out = pu.trimseq(out)

    return out
    



def chebproj(p, c):
    """
    Project a Chebyshev series into P^p, the space of degree-p polynomials,
    under the L^2 inner product.

    Parameters
    ----------
    p : integer
        Polynomial degree of target space.
    c : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Chebyshev series of the projection.

    Examples
    --------
    >>> c = (1,2,3)
    >>> chep.chebproj(1,c)
    array([0., 2.])
    """

    # Check if the polynomial already lies in P^p
    if len(c) - 1 <= p:
        # Make no changes (except removing trailing 0's)
        out = c
    else:
        # Create vector
        out = c[:(p+1)]
        # Project high-order terms
        for n, coeff in enumerate(c[(p+1):]):
            out += coeff*chebprojvec(p, n+p+1)

    # Remove any trailing 0's
    out = pu.trimseq(out)

    return out



def chebprojvec(p, n):
    """
    Evaluates a Chebyshev series representing the L^2 projection of T_n, the degree-n
    Chebyshev polynomial, into P^p, the space of degree-p polnoymials.

    Results are stored in the dictionary chebprojvec_dict for quicker access
    once computed.

    Parameters
    ----------
    p : integer
        Degree of space into which we are projecting.
    n : integer
        Degree of Chebyshev polnomial to be projected.

    Returns
    -------
    chebprojvec_dict[p][n] : ndarray
        Array representing the L^2 projection.
    """
    
    # Check if any vector at p has already been computed 
    if p not in chebprojvec_dict:
        # Create dictionary at p
        chebprojvec_dict[p] = {}

    # Check if vector at p, n has already been computed
    if n not in chebprojvec_dict[p]:
        # Create vector
        vec = np.zeros(p+1)
        
        # Check if projection is trivial
        if p >= n:
            vec[n] = 1.0
        else:
            # Evaluate
            chebdualmat_p = chebdualmat(p)
            for m in range(p+1):
                if (m + n) % 2 == 0:
                    inner_prod = 1/(1 - (m+n)**2) + 1/(1 - (m-n)**2)
                    for l in range(p+1):
                        vec[l] += inner_prod*chebdualmat_p[l, m]
        
        # Store
        chebprojvec_dict[p][n] = vec
        
    return chebprojvec_dict[p][n]



def chebdualmat(p):
    """
    Evaluates the inverse of the L^2 mass matrix of Chebyshev functions,
    up to degree p.

    Results are stored in the dictionary chebdualmat_dict for quicker access
    once computed.

    Parameters
    ----------
    p : integer
        Highest degree of Chebyshev polnomials.

    Returns
    -------
    chebdualmat_dict[p] : ndarray
        Array representing L^2 mass matrix inverse.
    """

    # Check if matrix has already been computed 
    if p not in chebdualmat_dict:
        # Create non-inverse matrix
        mat = np.zeros([p+1, p+1])
        
        # Evaluate
        for m in range(p+1):
            for n in range(p+1):
                if (m + n) % 2 == 0:
                    if m <= n:
                        mat[m, n] = 1/(1 - (m+n)**2) + 1/(1 - (m-n)**2)
                    else:
                        mat[m, n] = mat[n, m]
        
        # Invert and store
        chebdualmat_dict[p] = np.linalg.inv(mat)

    return chebdualmat_dict[p]
