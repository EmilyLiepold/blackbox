import sys
import multiprocessing as mp
import numpy as np
import scipy.optimize as op


def get_default_executor():
    """
    Provide a default executor (a context manager
    returning an object with a map method).

    This is the multiprocessing Pool object () for python3.

    The multiprocessing Pool in python2 does not have an __enter__
    and __exit__ method, this function provides a backport of the python3 Pool
    context manager.

    Returns
    -------
    Pool : executor-like object
        An object with context manager (__enter__, __exit__) and map method.
    """
    if (sys.version_info > (3, 0)):
        Pool = mp.Pool
        return Pool
    else:
        from contextlib import contextmanager
        from functools import wraps

        @wraps(mp.Pool)
        @contextmanager
        def Pool(*args, **kwargs):
            pool = mp.Pool(*args, **kwargs)
            yield pool
            pool.terminate()
        return Pool

def getInitialPoints(box,n):
    points = latin(n,len(box))
    return unScalePoints(box,points)

def unScalePoint(box,point):
    return [box[i][0]+(box[i][1]-box[i][0])*point[i] for i in range(d)]

def unScalePoints(box,points):
    return [unScalePoint(box, point) for point in points]

def ScalePoint(box,point):
    return [(point[i] - box[i][0]) / (box[i][1]-box[i][0])  for i in range(d)]

def ScalePoints(box,points):
    return [ScalePoint(box, point) for point in points]


def default_break_checker(*args):
    return False


def getFit(points,nrand=10000,nrand_frac=0.05):
    # Get the dimension of the data
    d = len(points[0]) - 1

    # make sure that the input data is a numpy array
    points = np.asarray(points)

    # Construct the space-rescaling matrix
    T = np.identity(d)

    if d > 1:
        # Perform the initial fit
        fit_noscale = rbf(points, np.identity(d))

        # Construct a space of random points and calculate the fit at those points.
        population = np.zeros((nrand, d+1))
        population[:, 0:-1] = np.random.rand(nrand, d)
        population[:, -1] = list(map(fit_noscale, population[:, 0:-1]))

        # Grab ths points with the smallest fit values.
        cloud = population[population[:, -1].argsort()][0:int(nrand*nrand_frac), 0:-1]

        # Construct the covariance matrix and find its eigensystem
        eigval, eigvec = np.linalg.eig(np.cov(np.transpose(cloud)))

        # Use that eigensystem to construct the space-rescaling vector
        T = [eigvec[:, j]/np.sqrt(eigval[j]) for j in range(d)]
        T = T/np.linalg.norm(T)

    # Fit given the spatial rescaling
    return(rbf(points,T))

def getNewPoints(fit,currentPoints,batch,rho0):

    N  = len(currentPoints)
    d  = len(currentPoints[0]) - 1

    currentPoints = np.append(currentPoints, np.zeros((batch, d+1)), axis=0)

    v1 = getBallVolume(d)

    for j in range(batch):
        r = ((rho0*((j + 1.) / (batch))**p)/(v1*(N+j)))**(1./d)
        cons = [{'type': 'ineq', 'fun': lambda x, localk=k: np.linalg.norm(np.subtract(x, currentPoints[localk, 0:-1])) - r}
                for k in range(N+j)]
        while True:
            minfit = op.minimize(fit, np.random.rand(d), method='SLSQP', bounds=[[0., 1.]]*d, constraints=cons)
            if np.isnan(minfit.x)[0] == False:
                break
        currentPoints[N+j, 0:-1] = np.copy(minfit.x)

    newPoints = currentPoints[N:,0:-1]

    return(currentPoints,newPoints)

def getBallVolume(d):
    # volume of d-dimensional ball (r = 1)
    if d % 2 == 0:
        return(np.pi**(d/2)/np.math.factorial(d/2))
    else:
        return(2*(4*np.pi)**((d-1)/2)*np.math.factorial((d-1)/2)/np.math.factorial(d))




def search(f, box, n, m, batch, resfile,
           rho0=0.5, p=1.0, nrand=10000, nrand_frac=0.05,
           executor=get_default_executor(), breakCheckFn=default_break_checker):
    """
    Minimize given expensive black-box function and save results into text file.

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    box : list of lists
        List of ranges for each parameter.
    n : int
        Number of initial function calls.
    m : int
        Number of subsequent function calls.
    batch : int
        Number of function calls evaluated simultaneously (in parallel).
    resfile : str
        Text file to save results.
    rho0 : float, optional
        Initial "balls density".
    p : float, optional
        Rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower).
    nrand : int, optional
        Number of random samples that is generated for space rescaling.
    nrand_frac : float, optional
        Fraction of nrand that is actually used for space rescaling.
    executor : callable, optional
        Should have a map method and behave as a context manager.
        Allows the user to use various parallelisation tools
        as dask.distributed or pathos.
    """
    # space size
    d = len(box)

    # adjusting the number of function calls to the batch size
    if n % batch != 0:
        n = n - n % batch + batch

    if m % batch != 0:
        m = m - m % batch + batch

    # go from normalized values (unit cube) to absolute values (box)
    def cubetobox(x):
        return [box[i][0]+(box[i][1]-box[i][0])*x[i] for i in range(d)]

    # generating latin hypercube
    points = np.zeros((n, d+1))
    points[:, 0:-1] = latin(n, d)

    # initial sampling
    for i in range(n//batch):
        with executor() as e:
            points[batch*i:batch*(i+1), -1] = list(e.map(f, list(map(cubetobox, points[batch*i:batch*(i+1), 0:-1]))))

    # normalizing function values
    fmax = max(abs(points[:, -1]))
    points[:, -1] = points[:, -1]/fmax

    # volume of d-dimensional ball (r = 1)
    v1 = getBallVolume(d)

    # subsequent iterations (current subsequent iteration = i*batch+j)

    for i in range(m//batch):

        fit = getFit(points,nrand=nrand,nrand_frac=nrand_frac)

        # check if the current fit is sufficiently converged.
        if i > 0:
            if breakCheckFn(fit, prevFit, fmax, prevFmax):
                break

        # store the current fit for use in the next iteration
        prevFit = fit
        prevFmax = fmax

        points, newpoints = getNewPoints(fit,points,batch,rho0)

        with executor() as e:
            points[n+batch*i:n+batch*(i+1), -1] = list(e.map(f, list(map(cubetobox, points[n+batch*i:n+batch*(i+1), 0:-1]))))/fmax

    # saving results into text file
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1]*fmax
    points = points[points[:, -1].argsort()]

    labels = [' par_'+str(i+1)+(7-len(str(i+1)))*' '+',' for i in range(d)]+[' f_value    ']
    np.savetxt(resfile, points, delimiter=',', fmt=' %+1.4e', header=''.join(labels), comments='')

    def returnedScaleFit(x):
        return fmax * fit(cubetobox(x))

    return points, returnedScaleFit

def latin(n, d):
    """
    Build latin hypercube.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Size of space.

    Returns
    -------
    lh : ndarray
        Array of points uniformly placed in d-dimensional unit cube.
    """
    # spread function
    def spread(points):
        return sum(1./np.linalg.norm(np.subtract(points[i], points[j])) for i in range(n) for j in range(n) if i > j)

    # starting with diagonal shape
    lh = [[i/(n-1.)]*d for i in range(n)]

    # minimizing spread function by shuffling
    minspread = spread(lh)

    for i in range(1000):
        point1 = np.random.randint(n)
        point2 = np.random.randint(n)
        dim = np.random.randint(d)

        newlh = np.copy(lh)
        newlh[point1, dim], newlh[point2, dim] = newlh[point2, dim], newlh[point1, dim]
        newspread = spread(newlh)

        if newspread < minspread:
            lh = np.copy(newlh)
            minspread = newspread

    return lh


def rbf(points, T):
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details) using scaling matrix.

    Parameters
    ----------
    points : ndarray
        Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].
    T : ndarray
        Scaling matrix.

    Returns
    -------
    fit : callable
        Function that returns the value of the RBF-fit at a given point.
    """
    n = len(points)
    d = len(points[0])-1

    def phi(r):
        return r*r*r

    Phi = [[phi(np.linalg.norm(np.dot(T, np.subtract(points[i, 0:-1], points[j, 0:-1])))) for j in range(n)] for i in range(n)]

    P = np.ones((n, d+1))
    P[:, 0:-1] = points[:, 0:-1]

    F = points[:, -1]

    M = np.zeros((n+d+1, n+d+1))
    M[0:n, 0:n] = Phi
    M[0:n, n:n+d+1] = P
    M[n:n+d+1, 0:n] = np.transpose(P)

    v = np.zeros(n+d+1)
    v[0:n] = F

    sol = np.linalg.solve(M, v)
    lam, b, a = sol[0:n], sol[n:n+d], sol[n+d]

    def fit(x):
        return sum(lam[i]*phi(np.linalg.norm(np.dot(T, np.subtract(x, points[i, 0:-1])))) for i in range(n)) + np.dot(b, x) + a

    return fit
