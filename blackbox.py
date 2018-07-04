#!/usr/bin/env python
import sys
import multiprocessing as mp
import numpy as np
import scipy.optimize as op
import copy
import blackboxhelper as bbh
import utils as u
from skopt import Optimizer
import skopt.acquisition as acq
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel, Matern


VERSION = 180625

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

def expandBox(box,frac):

    for i in range(len(box)):
        span = box[i][1] - box[i][0]
        box[i][0] -= span * frac
        box[i][1] += span * frac
    box = np.asarray(box)

    return box



def getFit(inpoints,fitkwargs={}, method='rbf'):
    # This function will take a list of points with shape (n,d+1), as well as other parameters
    # and return an rbf fit to that data.

    # nrand describes the number of points generated for the spatial rescaling
    # nrand_frac describes the fraction of nrand which is actually used for that rescaling.
    # scaled is a Boolean which describes whether or not the input points lie in a unit cube.

    # If the data is not scaled, then we will rescale the data and perform getFit on that data

    methods = {'bayes':getFitBayes,'rbf':getFitRBF}

    if method in methods:
        return(methods[method](inpoints,**fitkwargs))
    else:
        print("I don't recognize that fit method!")
        return(-1)


def getFitRBF(inpoints,nrand=10000,nrand_frac=0.05):

    # Copy the data and scale it down.
    box = getBox(inpoints[:,:-1])
    points = copy.deepcopy(inpoints)
    points[:,:-1] = ScalePoints(box, inpoints[:,:-1])

    # Scale the values at those points
    fmax = max(abs(points[:, -1]))
    points[:, -1] = points[:, -1]/fmax

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

    fit = rbf(points,T)

    def boxtocube(x):
        return np.divide(np.subtract(x,box[:,0]),np.subtract(box[:,1],box[:,0]))
    
    def returnedScaleFit(x):
        return fmax * fit(boxtocube(x))

    # Return a fit with dimensions
    return returnedScaleFit

def getFitBayes(inpoints,returnStd=False,scale=None):

    # Copy the input data.
    points = copy.deepcopy(inpoints)

    ## Mark down the boundaries of the box
    box = getBox(points[:,:-1])

    ## Find the minimum in the data and invert the data into a cube
    ## NB: We are not rescaling into a cube here.
    MIN = -np.min(points[:,-1])
    points[:,-1] = np.divide(MIN,points[:,-1])
    
    ## Scale the parameters into the unit cube
    points[:,:-1] = ScalePoints(box,points[:,:-1])

    ## Get the boundaries of the scaled data (0 and 1)
    dimensions = getBox(points[:,:-1])

    ## Construct the Gaussian Kernel with some white noise
    kernel = RBF([0.1 * (d[1] - d[0]) for d in dimensions], [(1e-5, d[1] - d[0]) for d in dimensions]) * ConstantKernel(1.0, (1e-5, 1e8))  + WhiteKernel(noise_level_bounds = (1e-5,1e1))

    ## Construct the Regressor object using that kernel.
    model = GaussianProcessRegressor(alpha=1e-10, kernel=kernel,n_restarts_optimizer=2,normalize_y=True)

    ## Fit the model to the data    
    model.fit(points[:,:-1],points[:,-1])

    ## Construct the functions which will be returned.
    def outFit(x):
        
        ## Make sure that the input is an array with the proper dimensions.
        if type(x) is not np.ndarray:
            x = np.asarray(x)
        if len(x.shape) == 1:
            x = np.asarray([x])
        
        ## Scale the asked point and make it an array
        x = ScalePoints(box,x)
        x = np.asarray(x)

        ## Get the prediction from the model
        y_pred = model.predict(x,return_std=False)

        ## If a predicted point is greater than 0 
        ## (overflow, since we've inverted the objective function),
        ## make it an arbitrarily small number

        y_pred[y_pred > 0] = -1e-30

        ## Return the objective function.
        y_pred = np.divide(MIN,y_pred)

        return y_pred

    if returnStd:
        def outFitSigma(x):
            ## Make sure that the input is an array with the proper dimensions.
            if type(x) is not np.ndarray:
                x = np.asarray(x)
            if len(x.shape) == 1:
                x = np.asarray([x])

            ## Scale the asked point and make it an array
            x = ScalePoints(box,x)
            x = np.asarray(x)

            ## Get the prediction from the model
            y_pred, sigma = model.predict(x, return_std=True)

            ## If a predicted point is less than 0 
            ## (overflow, since we've inverted the objective function),
            ## make it an arbitrarily small number
            y_pred[y_pred > 0] = -1e-30
            sigma[y_pred > 0] = 0.

            ## Calculate the error in the objective function.
            sigma = np.abs(np.multiply(np.divide(MIN,np.multiply(y_pred,y_pred)),sigma))

            return sigma

        return outFit, outFitSigma
    else:
        return outFit





def getNextPoints(inpoints,N, fitkwargs = {}, ptkwargs = {},method='rbf',plot=False,plotfn='next'):
    #optParams = {'p': None, 'rho': None, 'nrand': None, 'randfrac': None}):
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
        fit = getFit(inpoints,fitkwargs=fitkwargs,method=method)

        # Accumulate the keywords for the getNextPointsRBF function and run that.
        points, newpoints = getNextPointsRBF(fit,inpoints,N, **ptkwargs)
        inpoints[:,:-1] = unScalePoints(box, inpoints[:,:-1])
        
        newpoints = np.asarray(unScalePoints(box, newpoints))

        if plot:
            u.plotNewPointsRBF(inpoints,newpoints,plotfn)


    elif method == 'bayes':

        newpoints = getNextPointsBayes(inpoints,N, **fitkwargs)

        if plot:
            u.plotNewPointsBayes(inpoints,newpoints,plotfn)


    ## Return (with dimensions) the new points
    return(newpoints)

def getNextPointsBayes(inpoints,N,regrid=False,scale=None,kappa=1.96,rho0 = 1.0, p = 0.8):

    # Make a copy of the input data
    points = copy.deepcopy(inpoints)

    ###### RESCALE THE DATA TO A UNIT HYPERCUBE
    box = getBox(points[:,:-1])

    points[:,:-1] = ScalePoints(box,points[:,:-1])

    ## Find the minimum in the data and invert the data into a cube
    ## NB: We are not rescaling into a cube here.
    MIN = -np.min(points[:,-1])
    points[:,-1] = np.divide(MIN,points[:,-1])

    ###### GRAB THE NEW BOUNDS
    dimensions = getBox(points[:,:-1])


    fit, err = getFitBayes(points,returnStd=True)

    ###### Build a guess for the kernel which has length scale 1/10 of the length of the box and white noise up to 10
    kernel = RBF([0.1 * (d[1] - d[0]) for d in dimensions], [(1e-5, d[1] - d[0]) for d in dimensions]) * ConstantKernel(1.0, (1e-5, 1e8))  + WhiteKernel(noise_level_bounds = (1e-5,1e1))

    ###### Construct the GPR with that kernel
    model = GaussianProcessRegressor(alpha=1e-10, kernel=kernel,n_restarts_optimizer=2,normalize_y=True)
    
    def acq_func(X):
        return(np.subtract(fit(X),np.power(err(X),2) / (2 * len(dimensions))))


    
    newpoints = []

    ###### Get the volume constant for a d-dimensional ball
    v1 = getBallVolume(len(dimensions)) 

    ###### The volume of the unit cube is 1.    
    ###### The volume of each ball is v1 * rr^d 
    ###### We want to find rr such that the balls occupy the fraction rho0 of the box

    rr = ((rho0)/(v1*(len(points))))**(1./len(dimensions))

    ###### Fit the model to the currently known points.
    # model.fit(points[:,:-1],points[:,-1])

    ###### Just to be safe, we're going to generate N^2 test points
    n_points = N * N

    ###### We'll also stick with the 99% CI for the LCB.
    # acq_func_kwargs={'kappa':1.96}



    ###### Start a list of test points
    X = []

    ###### run though our N^2 points
    for i in range(n_points):

        ###### Try to find a point up to 1000 times
        for j in range(1000):

            ###### Assume that the point will work until proven otherwise
            goodPt = True

            ###### Grab a random point in the space
            pt = np.random.rand(1,len(dimensions))
            xx = np.asarray(unScalePoints(dimensions,pt))

            ###### Loop through the old points
            for k in range(len(points)):
                ###### Find the distance between the test point and the preexisting point.
                c = np.linalg.norm(np.subtract(xx, points[k, :-1])) - rr

                ###### If that distance is less than rr, then drop that point
                if c < 0:
                    goodPt = False
                    break
            ###### If the point is far enough from all of the preexisting points, break out of the loop
            if goodPt == True:
                break

        ###### Add the point to list of trial points.
        X.append(xx)

        ###### If the point (after 1000 samples) was too close to any other points, 
        ###### throw a warning and reduce the cutoff distance slightly.
        if goodPt == False:

            print "Couldn't find enough points. Reducing the search distance."

            rho0 *= 0.9
            rr = ((rho0)/(v1*(len(points))))**(1./len(dimensions))

    ###### Slice the list of test points appropriately
    X = np.asarray(X)[:,0,:]

    ###### Start the list of newpoints
    next_xs_ = []
    
    ###### Find the LCB value at each of the test points and sort the test points by those values
    # values = acq._gaussian_acquisition(
        # X=X, model=model,acq_func="LCB",acq_func_kwargs=acq_func_kwargs)

    values = acq_func(X)

    x0 = X[np.argsort(values)[:n_points]]
    v0 = values[np.argsort(values)[:n_points]]




    ###### Loop through all of the new points
    ###### Hang on to an index over x0
    trialIndex = 0
    for n in range(N):
        print n

        ###### Keep track of the number of attempts we've made on a particular n
        attempts = 0        

        ###### Update the target distance

        rr = ((rho0*((n + 1.) / (N))**p)/(v1*(len(points))))**(1./len(dimensions))

        ###### Construct a list of constraints on the new points.
        cons = [{'type': 'ineq', 'fun': lambda x, localk=k: np.linalg.norm(np.subtract(x, points[localk, :-1])) - rr}
                for k in range(len(points))]

        for i in range(N):          

            ###### Try to minimize the LCB given a particular test point.
            try:
                # results = op.minimize(acq.gaussian_acquisition_1D,x0[trialIndex],args=(model, None, "LCB", acq_func_kwargs, False),method="SLSQP",bounds=dimensions,constraints = cons, options={'maxiter':100})
                results = op.minimize(acq_func,x0[trialIndex],method="SLSQP",bounds=dimensions,constraints = cons, options={'maxiter':100})
                trialIndex += 1
                break
            ###### If the fit didn't work, move onto the next trial point.
            except:
                attempts += 1
                trialIndex += 1

                ###### If we've run through quite a few attempts, pull the target distance down and retry the previous test points
                if attempts >= N / 2:
                    print "I ran into an issue while minimizing. I'm reducing the search distance and trying again!"
                    rho0 *= 0.9
                    trialIndex -= attempts
                    attempts = 0
                    
                    rr = ((rho0*((n + 1.) / (N))**p)/(v1*(len(points))))**(1./len(dimensions))
                    cons = [{'type': 'ineq', 'fun': lambda x, localk=k: np.linalg.norm(np.subtract(x, points[localk, :-1])) - rr}
                        for k in range(len(points))]


                

        ###### Pull the results from the minimization.
        cand_xs = np.array(results.x)
        cand_acqs = np.array(results.fun)

        ###### Add the minimizing point to the list of points.
        newpoint = np.r_[cand_xs, fit(cand_xs.reshape(1,-1))]
        points = np.r_[points,newpoint.reshape(1,-1)]

        ###### Add the new point to the list of new points
        newpoints.append(cand_xs)

    ###### Return the original dimensions to the point.
    newpoints = np.asarray(newpoints)
    newpoints = unScalePoints(box,newpoints)
    newpoints = np.asarray(newpoints)

    return newpoints



def getNextPointsRBF(fit,currentPoints,batch,rho0=0.5,p=1.0):
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

        points, newpoints = getNextPointsRBF(fit,points,batch,rho0=rho0, p=p)

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

def runInit(Output, Bounds, N, fmt = None, log = [],fst=None):

    if len(Bounds) % 2 == 1:
        print("Found an odd number of bounds! Make sure that you've included both a lower and an upper bound for each dimension!")
        sys.exit(1)

    if fmt is not None and len(fmt) != len(Bounds) / 2:
        print("The number of labels doesn't seem to match the number of dimensions!")
        sys.exit(1)


    box = np.asarray([[Bounds[2 * i], Bounds[2 * i + 1]] for i in range(len(Bounds) / 2)])

    box[log] = np.log10(box[log])

    points = getInitialPoints(box, N)

    points[:,log] = np.power(10,points[:,log])

    if fmt == None:
        header = " ".join(["Param" + str(i+1) for i in range(len(box))])
        np.savetxt(Output,points,header=header)
    else:
        if fst == None:
            def formatLine(line):
                out = "".join([fmt[i] + ",%.2e " % line[i] for i in range(len(line))]) + "\n"
                return(out)
        else:
            def formatLine(line):
                out = "".join([fmt[i] + ","+fst[i]+" " % line[i] for i in range(len(line))]) + "\n"
                return(out)

        header = "# " + " ".join(fmt) + "\n"
        out = open(Output,'w')
        out.write(header)
        for f in points:
            out.write(formatLine(f))
        out.close()

    pass

def runNext(N, Input, Output, method = 'bayes', plot = None, args = [],fmt = None, log=[],fst=None):

    allowedParams = ['p', 'rho', 'nrand', 'randfrac', 'method']
    optParams = {}

    if len(args) % 2 != 0:
        print("Incorrect number of arguments! Each optional param should have both a name and a value.")
        exit(1)


    for i in range(len(args) / 2):
        if args[2 *i] in allowedParams:
            optParams[args[2 *i]] = (args[1 +  2 *i])
        else:
            print(args[2 * i] + "isn't a parameter that I recognize! Please try something else.")
            exit(1)

    inpoints = u.loadFile(Input)
    print inpoints
    if fmt is not None and len(fmt) != len(inpoints[0]) - 1:
        print("The number of labels doesn't match the number of parameters!")
        exit(1)


    ptkwargs = {}
    fitkwargs = {}
    if 'p' in optParams: ptkwargs['p'] = float(optParams['p'])
    if 'rho' in optParams: ptkwargs['rho0'] = float(optParams['rho'])
    if 'nrand' in optParams: ptkwargs['nrand_frac'] = float(optParams['nrand'])
    if 'randfrac' in optParams: ptkwargs['nrand'] = float(optParams['randfrac'])
    
    if plot is not None:
        plotfn = str(plot)
        plot = True
    else:
        plotfn = None
        plot = False

    inpoints[:,log] = np.log10(inpoints[:,log])

    newpoints =  getNextPoints(inpoints, N,fitkwargs=fitkwargs,ptkwargs=ptkwargs, method=method,plot=plot,plotfn=plotfn)

    inpoints[:,log] = np.power(10,inpoints[:,log])

    if fmt == None:
        header = " ".join(["Param" + str(i+1) for i in range(len(newpoints[0]))])
        np.savetxt(Output,newpoints,header=header)
    else:
        if fst == None:
            def formatLine(line):
                out = "".join([(fmt[i] + ",%.2e ") % line[i] for i in range(len(line))]) + "\n"
                return(out)
        else:
            def formatLine(line):
                out = "".join([(fmt[i] + ","+fst[i]+" ") % line[i] for i in range(len(line))]) + "\n"
                return(out)

        header = "# " + " ".join(fmt) + "\n"
        out = open(Output,'w')
        out.write(header)
        for f in newpoints:
            out.write(formatLine(f))
        out.close()


    pass

def runAnalysis(Input, plot=None, method = 'bayes', err = False,labels = None,log=[],box=None):


    inpoints = u.loadFile(Input)
    


    if plot == None:
        plotfn = Input + ".png"
        plot = False
    else:
        plotfn = plot + ".png"
        plot = True


    inpoints[:,log] = np.log10(inpoints[:,log])    

    if box is None:
        box = getBox(inpoints[:,:-1])
    else:
        newBox = [[float(box[2*i]),float(box[2*i+1])] for i in range(len(box) / 2)]
        box = np.asarray(newBox)

    d = len(inpoints[0]) - 1

    print('Performing a '+method+' style fit')

    if err:
        fit, errFit = getFit(inpoints,fitkwargs={'returnStd':err},method=method)
    else:
        fit = getFit(inpoints,fitkwargs={},method=method)
        errFit = None


    if labels is not None:
        BF = u.analyzeFit(fit,box,plot=plot,plotfn=plotfn,labels=labels,errFit=errFit)#,extent=box)
    else:
        BF = u.analyzeFit(fit,box,plot=plot,plotfn=plotfn,errFit=errFit)

    BF = np.asarray(BF)

    A = np.power(10,BF[log,0] + BF[log,1])
    B = np.power(10,BF[log,0] - BF[log,1])
    BF[log,0] = 0.5 * np.add(A,B)
    BF[log,1] = 0.5 * np.subtract(A,B)
    
    print BF
    return BF



if __name__ == '__main__':

    commands = {'init': runInit, 'next': runNext, 'analyze' : runAnalysis}

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser')

    parser_init = subparsers.add_parser('init', help='Get an initial set of points')

    parser_init.add_argument(
        'N', help='Number of points to generate',type=int)

    parser_init.add_argument(
        'Output', help='Target filename for the output data',type=str)

    parser_init.add_argument(
        'Bounds', help='Pairs of lower and upper bounds',nargs='*',type=float)

    parser_init.add_argument(
        '-f', '--fmt', help='List of labels used in the output file', required=False,nargs='*',type=str)

    parser_init.add_argument(
        '-s', '--fst', help='List of formatting strings used in the output file', required=False,nargs='*',type=str)

    parser_init.add_argument(
        '-l', '--log', help='List of parameters (starting at 0) whose sampling will be in log-space', required=False,nargs='*',type=int)

    parser_next = subparsers.add_parser('next', help='Get the next set of points')

    parser_next.add_argument(
        'N', help='Number of points to generate',type=int)

    parser_next.add_argument(
        'Input', help='Filename for the input data',type=str)

    parser_next.add_argument(
        'Output', help='Target filename for the output data',type=str)

    parser_next.add_argument(
        '-m', '--method', help='Method for choosing new points',type=str)

    parser_next.add_argument(
        '-p', '--plot', help='Base Filename for plots',type=str)

    parser_next.add_argument(
        '-f', '--fmt', help='List of labels used in the output file', required=False,nargs='*',type=str)

    parser_next.add_argument(
        '-s', '--fst', help='List of formatting strings used in the output file', required=False,nargs='*',type=str)

    parser_next.add_argument(
        '-l', '--log', help='List of parameters (starting at 0) whose sampling will be in log-space', required=False,nargs='*',type=int)

    parser_next.add_argument(
        '-a', '--args', help='Additional arguments for the optimizer', nargs='*')



    parser_analyze = subparsers.add_parser('analyze', help='Take a look at the data')

    parser_analyze.add_argument(
        'Input', help='Filename for the input data')

    parser_analyze.add_argument(
        '-p', '--plot', help='Save a plot of the fit',type=str)

    parser_analyze.add_argument(
        '-m', '--method', help='Method for choosing new points',type=str)

    parser_analyze.add_argument(
        '-e', '--err', action='store_true', help='Plot the error in the fit')

    parser_analyze.add_argument(
        '-f', '--labels', help='List of labels to use in the plots', required=False,nargs='*',type=str)

    parser_analyze.add_argument(
        '-l', '--log', help='List of parameters (starting at 0) whose sampling will be in log-space. NOT YET IMPLEMENTED.', required=False,nargs='*',type=int)

    parser_analyze.add_argument(
        '-b', '--box', help='List of pairs of lower and upper bounds to use for analysis and plotting.', required=False,nargs='*')

    args = parser.parse_args()
    
    kwargs = {k:v for k,v in vars(args).iteritems() if v is not None}

    commands = {'init': runInit, 'next': runNext, 'analyze' : runAnalysis}

    command = commands[kwargs['subparser']]
    del kwargs['subparser']
    command(**kwargs)
