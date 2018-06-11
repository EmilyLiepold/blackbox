import sys
import multiprocessing as mp
import numpy as np
import scipy.optimize as op
import copy
import blackboxhelper as bbh
import utils as u
from skopt import Optimizer

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
    """
    This function will return a list of points with shape (n,d).
    Those points will form a latin hypercube such that the 
    positions minimize a 1/r potential energy.

    Parameters
    ----------
    box : List of lists of floats
          It should contain [minimumValue, maximumValue] for each dimension.
          It should have shape (d,2)
    n   : int
          Number of initial points which you would like.

    Returns
    -------
    points : list of lists of floats
             Array of points uniformly placed in the initial box.
             In should have shape (n,d)
    """
    
    points = latin(n,len(box))
    return np.asarray(unScalePoints(box,points))

def unScalePoint(box,point):
    # This function takes a list with shape (d) describing a single point 
    # which lies in a unit cube and unscales it to lie within the given box.
    # We are usually interested in unscaled quantities for doing analysis.
    
    return [box[i][0]+(box[i][1]-box[i][0])*point[i] for i in range(len(box))]

def unScalePoints(box,points):
    # This function applies unScalePoint() on a list of points. See 
    # unScalePoint() for more details

    return [unScalePoint(box, point) for point in points]

def ScalePoint(box,point):
    # This function takes a list with shape (d) describing a point within
    # the given box and scales it down so that it lies within a unit cube.
    # Most of the fitting functions and all of the internals of blackbox 
    # prefer to use these unit-cube quantities.

    return [(point[i] - box[i][0]) / (box[i][1]-box[i][0])  for i in range(len(box))]

def ScalePoints(box,points):
    # This function applies ScalePoint() on a list of point. See 
    # ScalePoint() for more details.

    return [ScalePoint(box, point) for point in points]


def default_break_checker(*args):
    # This function ignores the inputs (the fit function, box, etc) and always
    # returns False. This is used to turn off convergence checking.

    return False

def getBox(points):
    # This function takes a list of points (either unScaled or Scaled) and 
    # returns the smallest box which bounds those points. That box will have shape (d,2)

    return(np.asarray([[np.min(points[:,i]),np.max(points[:,i])] for i in range(len(points[0]))]))


def getFit(inpoints,nrand=10000,nrand_frac=0.05,scaled=False,method='rbf',returnStd=False):
    # This function will take a list of points with shape (n,d+1), as well as other parameters
    # and return an rbf fit to that data.

    # nrand describes the number of points generated for the spatial rescaling
    # nrand_frac describes the fraction of nrand which is actually used for that rescaling.
    # scaled is a Boolean which describes whether or not the input points lie in a unit cube.

    # If the data is not scaled, then we will rescale the data and perform getFit on that data

    if method == 'bayes':
        return(getFitBayes(copy.deepcopy(inpoints),returnStd=returnStd))
    if not scaled:
        # Find the box and Scale the points
        box = getBox(inpoints[:,:-1])
        ScaledPoints = copy.deepcopy(inpoints)
        ScaledPoints[:,:-1] = ScalePoints(box, inpoints[:,:-1])

        # Scale the values at those points
        fmax = max(abs(ScaledPoints[:, -1]))
        ScaledPoints[:, -1] = ScaledPoints[:, -1]/fmax

        # Perform the fit on this reduced data
        fit = getFit(ScaledPoints,nrand=nrand, nrand_frac=nrand_frac,scaled=True)


        def boxtocube(x):
            return np.divide(np.subtract(x,box[:,0]),np.subtract(box[:,1],box[:,0]))
        
        def returnedScaleFit(x):
            return fmax * fit(boxtocube(x))

        # Return a fit with dimensions
        return returnedScaleFit

    points = copy.deepcopy(inpoints)

    nrand = int(nrand)

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
        population = np.random.rand(nrand, d+1)
        population[:, -1] = fit_noscale(population[:,:-1])

        # Grab ths points with the smallest fit values.
        cloud = population[population[:, -1].argsort()][0:int(nrand*nrand_frac), 0:-1]

        # Construct the covariance matrix and find its eigensystem
        eigval, eigvec = np.linalg.eig(np.cov(np.transpose(cloud)))

        # Use that eigensystem to construct the space-rescaling vector
        T = [eigvec[:, j]/np.sqrt(eigval[j]) for j in range(d)]
        T = T/np.linalg.norm(T)

    # Fit given the spatial rescaling
    return(rbf(points,T))

def getFitBayes(inpoints,returnStd=False):


    # Rescale the data into the unit cube
    box = getBox(inpoints[:,:-1])
    inpoints[:,:-1] = ScalePoints(box, inpoints[:,:-1])

    # Rescale the data to be order unity
    fmax = np.max(np.abs(inpoints[:,-1]))
    inpoints[:,-1] = inpoints[:,-1] / fmax

    # Construct the bounds of the unit box
    dimensions = [(0.,1) for i in range(len(box))]

    # Construct the GP optimizer with the LCB acquisition function
    opt = Optimizer(dimensions, "gp",acq_func='LCB')

    # Tell the optimizer abount inpoints
    opt.tell(inpoints[:,:-1].tolist(),inpoints[:,-1].tolist())

    # See if we constructed a model
    if len(opt.models) == 0:
        print('Could not construct a model for that data! Oh well.')
        return -1
    else:

        # If we did, get that model and construct wrappers which can
        # be returned

        model = opt.models[-1]

        def outFit(x):
            x = np.asarray(x)
            if len(x.shape) == 1:
                x = np.asarray([x])
            x = np.asarray(ScalePoints(box,x))
            x_model = opt.space.transform(x.tolist())
            y_pred, sigma = model.predict(x_model, return_std=True)
            return y_pred * fmax
        if returnStd:
            def outFitSigma(x):
                x = np.asarray(x)
                if len(x.shape) == 1:
                    x = np.asarray([x])
                x = np.asarray(ScalePoints(box,x))
                x_model = opt.space.transform(x.tolist())
                y_pred, sigma = model.predict(x_model, return_std=True)
                return sigma * fmax

            return outFit, outFitSigma
        else:
            return outFit




def getNextPoints(inpoints,N, fitkwargs = {}, ptkwargs = {},method='rbf',plot=False,plotfn='next'): #optParams = {'p': None, 'rho': None, 'nrand': None, 'randfrac': None}):
    ## This function implements the logic required to grab the next set of points using the standard rbf method.
    ## The required inputs are 
    ##      inpoints, a list of points with shape (n,d+1), which are the parameters and measured function at n sample points
    ##      N, which is the number of requested points.
    ## The optional parameters for the fit should be placed in fitkwargs.
    ## The optional parameters for the acquisition function should be placed in ptkwargs

    ## The optional parameters are
    ##      p, which is a float which sets the shape of the search pattern.
    ##      rho, which is a float which sets the initial ball density
    ##      nrand, which is an integer which sets the number of points used for the spatial rescaling.
    ##      randfrac, which is a float which sets the fraction of nrand points which are used for the rescaling.


    # Get the shape of the box from the extent of the current points.
    box = getBox(inpoints[:,:-1])
    # Scale the points into a unit cube
    

    if method == 'rbf':
        inpoints[:,:-1] = ScalePoints(box, inpoints[:,:-1])
        # Accumulate the parameters for the fit and perform the fit
        fit = getFit(inpoints,method=method, **fitkwargs)

        # Accumulate the keywords for the getNewPoints function and run that.
        points, newpoints = getNewPoints(fit,inpoints,N, **ptkwargs)
        inpoints[:,:-1] = unScalePoints(box, inpoints[:,:-1])
        
        newpoints = np.asarray(unScalePoints(box, newpoints))

        if plot:
            u.plotNewPointsRBF(inpoints,newpoints,plotfn)


    elif method == 'bayes':
        # print inpoints
        inpoints[:,:-1] = ScalePoints(box, inpoints[:,:-1])
        newpoints = getNewPointsBayes(inpoints,N)
        # print inpoints
        inpoints[:,:-1] = unScalePoints(box, inpoints[:,:-1])
        # print newpoints
        newpoints = np.asarray(unScalePoints(box, newpoints))
        # print newpoints.shape
        if plot:
            u.plotNewPointsBayes(inpoints,newpoints,plotfn)


    # Return the unScaled points (with dimensions)
    # return(unScalePoints(box,newpoints))
    return(newpoints)

def getNewPointsBayes(inpoints,N):

    # Rescale the data into the unit cube
    box = getBox(inpoints[:,:-1])
    inpoints[:,:-1] = ScalePoints(box, inpoints[:,:-1])

    # Rescale the data to be order unity
    fmax = np.max(np.abs(inpoints[:,-1]))
    inpoints[:,-1] = inpoints[:,-1] / fmax

    # Construct the bounds of the unit box
    dimensions = [(0.,1) for i in range(len(box))]

    # Construct the GP optimizer with the LCB acquisition function
    opt = Optimizer(dimensions, "gp",acq_func='LCB')

    # Tell the optimizer abount inpoints
    opt.tell(inpoints[:,:-1].tolist(),inpoints[:,-1].tolist())

    # If we have a model constructed, then do an ask-tell loop
    if len(opt.models) != 0:
        
        # Start a list of the points
        newpoints = []

        # Loop through the desired number of points
        for i in range(N):
            # Ask for a point
            newpoint = np.asarray([opt.ask()])

            # Ask the most recent model about that point
            newpointVal = opt.models[-1].predict(newpoint)

            # Tell the optimizer about that point
            opt.tell(newpoint.tolist(),newpointVal.tolist())

            # Add the point to the list.
            newpoints.append(newpoint)

        # Turn the list into an ndarray and slice off the extraneous dimension
        newpoints = np.asarray(newpoints)[:,0,:]
    else:
        # If we don't have a model, just use the standard method.
        newpoints = opt.ask(n_points=N)

    inpoints[:,:-1] = unScalePoints(box, inpoints[:,:-1])
    inpoints[:,-1] = inpoints[:,-1] * fmax
    return newpoints



def getNewPoints(fit,currentPoints,batch,rho0=0.5,p=1.0):
    ## This function performs the logic of getting new points from the provided fit and existing points.
    ## The basic logic here follows CORS.
    ## fit should be a function which will take in a list of parameters and return a number.
    ## currentPoints should be a numpy array with shape (N,d+1) where N is the number of currently existing points.
    ## batch should be an integer which is the number of new points to choose.

    ## We will return a pair of arrays.
    ## currentPoints will be identical to the input currentPoints, but its shape is (N+batch,d+1) and the last column 
    ##      for the new points will be 0.
    ## newPoints will be (batch,d) and will be the submatrix of currentPoints describing the new positions.

    N  = len(currentPoints)
    d  = len(currentPoints[0]) - 1

    # Make space for the new points.
    currentPoints = np.append(currentPoints, np.zeros((batch, d+1)), axis=0)

    # Get the volume of a d-dimensional unit sphere.
    v1 = getBallVolume(d)

    # Loop through the new points
    for j in range(batch):

        # Calculate the minimum distance between the new point and any other existing point.
        r = ((rho0*((j + 1.) / (batch))**p)/(v1*(N+j)))**(1./d)

        # Establish constraints on the new position by requiring it to be greater than 'r' away from all existing points.
        cons = [{'type': 'ineq', 'fun': lambda x, localk=k: np.linalg.norm(np.subtract(x, currentPoints[localk, 0:-1])) - r}
                for k in range(N+j)]

        # Minimize the fit function under those constraints.
        while True:
            minfit = op.minimize(fit, np.random.rand(d), method='SLSQP', bounds=[[0., 1.]]*d, constraints=cons)
            if np.isnan(minfit.x)[0] == False:
                break

        # Move the minimized locations into the currentPoints array.
        currentPoints[N+j, 0:-1] = np.copy(minfit.x)

    # Split off the newPoints array
    newPoints = currentPoints[N:,0:-1]

    # Return those arrays
    return(currentPoints,newPoints)

def getBallVolume(d):
    # volume of d-dimensional ball (r = 1)
    if d % 2 == 0:
        return(np.pi**(d/2)/np.math.factorial(d/2))
    else:
        return(2*(4*np.pi)**((d-1)/2)*np.math.factorial((d-1)/2)/np.math.factorial(d))




def search(f, box, n, m, batch, resfile,
           rho0=0.5, p=1.0, nrand=10000, nrand_frac=0.05,
           executor=get_default_executor(), breakCheckFn=default_break_checker,plot=False):
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

    def boxtocube(x):
        return [(x[i] - box[i][0])/(box[i][1]-box[i][0]) for i in range(d)]

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

        ## Plot if you want to
        if plot:
            bbh.plotFit(fit,points,fmax,resfile + '.' + str(i) +  '.png')

        # check if the current fit is sufficiently converged.
        if i > 0:
            if breakCheckFn(fit, prevFit, fmax, prevFmax,d):
                break

        # store the current fit for use in the next iteration
        prevFit = fit
        prevFmax = fmax

        points, newpoints = getNewPoints(fit,points,batch,rho0=rho0, p=p)

        with executor() as e:
            points[n+batch*i:n+batch*(i+1), -1] = list(e.map(f, list(map(cubetobox, points[n+batch*i:n+batch*(i+1), 0:-1]))))/fmax

    # saving results into text file
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1]*fmax
    points = points[points[:, -1].argsort()]

    labels = [' par_'+str(i+1)+(7-len(str(i+1)))*' '+',' for i in range(d)]+[' f_value    ']
    np.savetxt(resfile, points, delimiter=',', fmt=' %+1.4e', header=''.join(labels), comments='')

    def returnedScaleFit(x):
        return fmax * fit(boxtocube(x))

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

    return np.asarray(lh)


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

    sub = np.diagonal(np.subtract.outer(points[:,0:-1],points[:,0:-1]),axis1=1,axis2=3)

    A = np.einsum('ji,kli',T,sub)
    ## A[d,N,N] 

    S = np.einsum('ijk,ijk->jk',A,A)
    ## S[N,N]

    Phi = np.sqrt(np.multiply(S,np.multiply(S,S)))

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
        if x is not np.ndarray:
            x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[np.newaxis,:]
        ## x[N,d]
        sub = np.empty((x.shape[0],points.shape[0],points.shape[1]-1))
        ## sub[N,M,d]
        for ii in range(len(points[:,0])):
            sub[:,ii,:] = np.subtract(x[:,:],points[ii,0:-1])

        ## T[d,d]
        A = np.einsum('ji,kli',T,sub)
        ## A[d,N,M]
        S = np.einsum('ijk,ijk->jk',A,A)
        ## S[N,M]
        P = np.sqrt(np.multiply(S,np.multiply(S,S)))
        ## lam[M]
        Q = np.einsum('i,ji',lam,P)
        ## Q[N]
        return Q+ np.einsum('i,ji',b,x) + a

    return fit

def runInit(args):
    if len(args) < 6:
        print("Not enough arguments! The command should look like:")
        print("blackbox.py init N out_filename xmin xmax (ymin ymax)...") 
        exit(1)
    if len(args) % 2 == 1:
        print("Not the right number of arguments! The command should look like:")
        print("blackbox.py init N out_filename xmin xmax (ymin ymax)...") 
        print("Make sure that you have both a lower and upper bound for each dimension.")
        sys.exit(1)

    rawbox = []
    for i in range(len(args) - 4):
        try:
            rawbox.append(float(args[i+4]))
        except ValueError:
            print(args[i+4] +" doesn't look like a float to me...")
            exit(1)

    box = [[rawbox[2 * i], rawbox[2 * i + 1]] for i in range(len(rawbox) / 2)]

    try:
        N = int(args[2])
    except ValueError:
        print(args[2] + "doesn't look like an integer to me.")
        exit(1)

    points = getInitialPoints(box, N)
    fname = args[3]

    header = " ".join(["Param" + str(i+1) for i in range(len(box))])
    np.savetxt(fname,points,header=header)

    pass

def runNext(args):
    p = None
    rho0 = None
    nrand = None
    nrand_frac = None
    plot = False
    plotfn = ''
    method = 'rbf'
    allowedParams = ['p', 'rho', 'nrand', 'randfrac', 'method','plot']
    # optParams = {'p': p, 'rho': rho0, 'nrand': nrand, 'randfrac': nrand_frac, 'method': method}
    optParams = {}


    if len(args) < 5:
        print("Not enough arguments! The command should look like ")
        print("blackbox.py next N in_filename out_filename (params)")
        exit(1)

    try:
        N = int(args[2])
    except ValueError:
        print(args[2] + "doesn't look like an integer to me.")
        exit(1)

    infname = args[3]
    outfname = args[4]

    if len(args) % 2 == 0:
        print("Incorrect number of arguments! Each optional param should have both a name and a value.")
        exit(1)


    for i in range((len(args) - 5) / 2):
        if args[5 + 2 *i] in allowedParams:
            optParams[args[5 + 2 *i]] = (args[6 + 2 *i])
        else:
            print(args[5 + 2 * i] + "isn't a parameter that I recognize! Please try something else.")
            exit(1)

    inpoints = np.loadtxt(infname)

    ptkwargs = {}
    fitkwargs = {}
    if 'p' in optParams: ptkwargs['p'] = float(optParams['p'])
    if 'rho' in optParams: ptkwargs['rho0'] = float(optParams['rho'])
    if 'nrand' in optParams: ptkwargs['nrand_frac'] = float(optParams['nrand'])
    if 'randfrac' in optParams: ptkwargs['nrand'] = float(optParams['randfrac'])
    if 'plot' in optParams:
        plot = True
        plotfn = optParams['plot']

    method = optParams['method'] if 'method' in optParams else 'rbf'


    newpoints =  getNextPoints(inpoints, N,fitkwargs=fitkwargs,ptkwargs=ptkwargs, method=method,plot=plot,plotfn=plotfn)

    header = " ".join(["Param" + str(i+1) for i in range(len(newpoints[0]))])
    np.savetxt(outfname, newpoints,header=header)

    pass

def runAnalysis(args):

    if len(args) < 3:
        print("Not enough arguments! The command should look like ")
        print("blackbox.py analyze in_filename (params)")
        exit(1)

    infname = args[2]
    inpoints = np.loadtxt(infname)
    box = getBox(inpoints[:,:-1])
    
    plot = False
    plotfn = infname + ".png"
    method = 'rbf'
    foundLabel = False

    d = len(inpoints[0]) - 1

    argList = args[3:]
    for i in range(len(argList)):
        if argList[i] == 'plot':
            plot = True
            continue
        if argList[i] == 'plotfn':
            plotfn = argList[i+1]
            continue
        if argList[i] == "label":
            foundLabel = True
            labels = argList[(i + 1):(i + d + 1)]
            continue
        if argList[i] == "bayes":
            method = 'bayes'
        if argList[i] == "rbf":
            method = 'rbf'

    print('Performing a '+method+' style fit')
    if method == 'rbf':
        fit = getFit(inpoints)
        
    elif method == 'bayes':
        fit = getFitBayes(inpoints)

    # unitbox = np.asarray([[0.,1.],[0.,1.]])
    if foundLabel:
        BF = u.analyzeFit(fit,box,plot=plot,plotfn=plotfn,labels=labels)#,extent=box)
    else:
        BF = u.analyzeFit(fit,box,plot=plot,plotfn=plotfn)

    BF = np.asarray(BF)
    
    print BF
    return BF

    pass



if __name__ == '__main__':

    commands = {'init': runInit, 'next': runNext, 'analyze' : runAnalysis}

    if len(sys.argv) == 1:
        print("No arguments provided, so I'm not sure what you want me to do.")
        exit(1)
    else:
        command = sys.argv[1]

    if command in commands:
        commands[command](sys.argv)
    else:
        print("I don't recognize that command! Please try something else.")
        exit(1)
