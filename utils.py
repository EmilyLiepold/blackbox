from scipy.optimize import minimize,fmin, minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import erf
import blackbox as bb
import copy
import vegas as v
import pickle as p

VERSION = 180709

def loadFile(f,grabLabels=False):
    F = open(f,'r')
    outList = []
    inFirstLine = True
    lines = []
    nCols = 0
    labels = None
    for i,FF in enumerate(F):
        if FF[0] == "#":
            if i == 0:
                labels=FF.strip('#').split()[:-1]
            continue
        lines.append(FF.split())
        nCols = len(FF.split()) if len(FF.split()) > nCols else nCols

    for l in lines:
        if l[0] == "#":
            continue
        if len(l) == nCols:
            outList.append(map(float,l))
        else:
            continue
    if grabLabels:
        return(np.asarray(outList),labels)
    else:
        return(np.asarray(outList))

def flattenFor(paramRanges):
    ## This function takes in a list of lists and spits out a
    ## list of all of the combinations of those values.

    ## The idea here is to flatten an arbitrary number of for loops
    ## into a single loop over those combinations. 

    ## Determine the number of iterations along each parameter
    n = [len(p) for p in paramRanges]

    ## Build a place to put the output.
    params = np.zeros((np.prod(n),len(n)))

    ## Begin the loop through combinations.
    for X in range(np.prod(n)):

        ## X is the flattened index, and we're going to use that
        ## index to count back to 0 with the correct parameters
        ileft = copy.deepcopy(X)

        ## Loop through the each parameter
        for j in reversed(range(len(n))):

            ## Grab the item from the range for that parameter
            ## and decrease the index accordingly.
            params[X,j] = paramRanges[j][ileft % n[j]]
            ileft = (ileft - ileft % n[j]) / n[j]
    return(params)


def chisqToPDF(chisq,d):
    fRed = np.exp(-np.divide(np.subtract(chisq,np.min(chisq)),d))
    return np.divide(fRed,np.sum(fRed))


def PDFtoChisq(fRed,d):
    chisq = np.multiply(-np.log(fRed),d)
    return np.subtract(chisq,np.min(chisq))

def marginalizeMC(fit,box,resolution=10,err=None,remainingAxes=[],batchSize=1):

    ## Get the dimensionality of the space
    d = len(box)

    ## Find the parameter values along each dimension
    axes = [np.linspace(box[i][0],box[i][1],resolution) for i in range(d)]

    ## Get lists of the remaining and removed parameters
    removedAxes = set(range(d))
    for i in set(remainingAxes):
        removedAxes.remove(i)
    removedAxes = list(removedAxes)
    remainingAxes = list(set(remainingAxes))

    ## Get the dimensionality after the marginalization
    remainingD = len(remainingAxes)

    ## Set up a place for the outputs
    image = np.zeros([resolution for r in remainingAxes])
    if err is not None:
        errImage = np.zeros([resolution for r in remainingAxes])

    ## Find the box which spans the removed parameters
    linebox = box[removedAxes]

    ## Initialize the integrator
    integ = v.Integrator(linebox,nhcube_batch=1e7,rtol=0.01)

    ## Construct a list of the possible combinations of un-marginalized parameters
    params = flattenFor([range(resolution/batchSize) for r in remainingAxes])

    ## Construct a list of the positions lying within the small batch blocks.
    params2 = flattenFor([range(batchSize) for r in remainingAxes])

    ## Start the main loop through the output space
    for IIII,p in enumerate(params):

        ## Construct the integrand
        @v.batchintegrand
        def intmask(x):
            
            ## Repeat the sample points for each batch point
            y = [x for k in range(batchSize**(remainingD))]

            ## Get the number of sample points
            L = x.shape[0]

            ## Make room for the batch points
            ys = np.empty((0, x.shape[1]+(remainingD)))

            ## Loop through the batch grid
            for p2 in params2:
                ## Label the point in the batch grid
                index = int(np.sum([val * batchSize**power for power, val in enumerate(p2[::-1])]))
                
                ## Loop through the remaining parameters
                for IIIII in range(len(remainingAxes)):
                    
                    ## Add the proper remaining parameters to each batch point.
                    y[index] = np.insert(y[index],remainingAxes[IIIII],axes[remainingAxes[IIIII]][batchSize * int(p[IIIII]) + int(p2[IIIII])],axis=1)

                ## Add the full points to the set to sample
                ys = np.r_[ys,y[index]]

            ## Calculate our fit on each of the batch points
            f = fit(ys)

            ## Determine the minimum of the fit over the sample points
            mf = np.min(f)

            ## Calculate the likelihood (up to a constant)
            pEval = np.exp((np.divide(np.subtract(mf,f),d)))

            ## Calculate the integrand of the numerator (up to the same constant)
            numEval = np.multiply(f,pEval)

            ## If we have an error fit, calculate the integrand in its numerator, again up to a constant.
            if err is not None:
                e = err(ys)
                errEval = np.multiply(e,pEval)

            ## Start the dictionary which will be passed to the integrator.
            out = {}

            ## Save the values to the dictionary
            for p2 in params2:
                index = int(np.sum([val * batchSize**power for power, val in enumerate(p2[::-1])]))
                out['p' + str(index)] = pEval[L * (index):L * (index + 1)]
                out['n' + str(index)] = numEval[L * (index):L * (index + 1)]
                if err is not None:
                    out['e' + str(index)] = errEval[L * (index):L * (index + 1)]

            return(out)

        ## Perform the integrations!
        C = integ(intmask, nitn=1e1, neval=1e3)

        ## Construct the output data, point by point
        for p2 in params2:
            index = int(np.sum([val * batchSize**power for power, val in enumerate(p2[::-1])]))
            imageIndex = tuple([batchSize * int(p[IIIII]) + int(p2[IIIII]) for IIIII in range(len(remainingAxes))])
            image[imageIndex] = C['n'  + str(index)].val / C['p' + str(index)].val
            if err is not None:
                errImage[imageIndex] = C['e'  + str(index)].val / C['p' + str(index)].val

    ## If a point is poorly resolved, remove it.
    image[image < 0.] = np.max(image)
    if err is not None:
        errImage[errImage < 0.] = np.max(errImage)
        return image, errImage
    else:
        return image

def getFuncMinimum(fit,box,niter=1000):
    fs = []
    for i in range(niter):
        pt = np.random.rand(1,len(box))
        xx = np.asarray(bb.unScalePoints(box,pt))
        func_min = minimize(fit,xx,bounds = box)
        fs.append(func_min.fun[0])

    return(np.min(fs))



def analyzeFit(fit,box,plot=True,showPlot=False,plotfn='fit',datafn='data',labels=None,searchRange = None,searchFraction=0.2,PDF_func=chisqToPDF,PDF_inv_func=PDFtoChisq,errFit=None,resolution=0):
    ## This function will take a fit function which is defined over a space defined by box and return the best fit parameters over that space
    ## That fit function must take in a list of lists of parameters with a shape (n,d) and return a list with shape (n).
    
    ## box should be a list of [min,max] for each dimension, for a shape (d,2).
    ## There are several optional parameters.
    ##     plot is a Boolean, which indicates whether or not we should plot the pairwise marginal PDFs.
    ##     showPlot will use plt.show() to bring those plots to the screen while running.
    ##     plotfn is a string which gives the filename where we should save that plot (if it exists.)
    ##     labels should be a list of strings with length (d). These labels will be used on the plot.
    ##     searchRange is a list with the same shape as 'box' which describes the mins and maxs where we
    ##          will actually compute the PDF.
    ##     searchFraction describes the fraction of the box extent over which we will compute the PDF.
    ##     extent has the same shape as 'box' and overrides 'box' when returning values. It is assumed that
    ##          if 'box' describes a unit cube, then 'extent' includes the physical parameters which
    ##          correspond to that box. This param is a bit hacky so I'll try to make it nicer.
    ##     PDF_func is a function which turns the output of 'fit' into a member of a probability distribution.

    if resolution != 0:
        useMC = True
    else:
        useMC = False

    if errFit is not None:
        plotErr = True
    else:
        plotErr = False

    searchRange = np.asarray(box)

    d = len(box)

    datalower1d = [[] for i in range(d)]
    dataupper1d = [[] for i in range(d)]


    if useMC:
        axisLists = [np.linspace(b[0],b[1],resolution) for b in box]
        for ii in range(d):

            if plotErr:
                image, errImage = marginalizeMC(fit,box,resolution=resolution,err=errFit,remainingAxes=[ii],batchSize=1)
                datalower1d[ii].append(image)
                datalower1d[ii].append(errImage)

                dataupper1d[ii].append(image)
                dataupper1d[ii].append(errImage)
            else:
                image = marginalizeMC(fit,box,resolution=resolution,err=None,remainingAxes=[ii],batchSize=1)
                datalower1d[ii].append(image)
                
                dataupper1d[ii].append(image)

    
    else:
        # Get the number of dimensions and find the smallest grid with more than a given number of points in that dimension.
        N,n = getGridDimensions(100000,d)

        axisLists = [np.linspace(b[0],b[1],n) for b in box]
        
        sPoints = flattenFor(axisLists)

        # Determine the desired output shape and perform the fit function over the grid.
        s = [n for K in range(d)]
        f = fit(sPoints).reshape(s)
        if plotErr:
            ferr = errFit(sPoints).reshape(s)

        # Turn the fit function grid into a PDF grid.
        pF = PDF_func(f,d)

        minIndices = np.unravel_index(np.argmin(f),np.shape(f))
        for ii in range(d):
            datalower1d[ii].append(marginalizeOverPDF(pF,f,[ii]))
            dataupper1d[ii].append(sliceArray(f,[ii],minIndices))
            if plotErr:
                datalower1d[ii].append(marginalizeOverPDF(pF,ferr,[ii]))
                dataupper1d[ii].append(sliceArray(ferr,[ii],minIndices))

    # Find the mean and stdev from the 1D margins
    bestFits = np.asarray([bestValFromPDF(chisqToPDF(datalower1d[i][0],d),axisLists[i]) for i in range(d)])

    # If all of the parameters are good and we want to plot, then plot!
    if plot:
        if labels == None:
            print("No labels provided! Reverting to default labelling.")
            labels = ["Parameter " + str(i+1) for i in range(d)]
        if len(labels) != d:
            print("The number of labels doesn't match the number of parameters! Reverting to default labelling.")
            labels = ["Parameter " + str(i+1) for i in range(d)]
        plt.close()

        chisqMin = getFuncMinimum(fit,box,niter=10)

        chisqCutoff = - d * np.log(1 - erf(5 / 2**0.5))
        levels = [chisqMin - d * np.log(1 - erf(II / 2**0.5)) for II in [1,2,3,4]]

        ## START PRODUCING THE DATA TO PLOT
        datalower = [[[] for i in range(d)] for j in range(d)]
        dataupper = [[[] for i in range(d)] for j in range(d)]

        ## We've already calculated the 1D stuff, so just do the 2D stuff
        for ii in range(d-1):
            for jj in range(ii+1,d):
                if useMC:
                    if plotErr:
                        image, errImage = marginalizeMC(fit,box,resolution=resolution,err=errFit,remainingAxes=[ii,jj],batchSize=1)
                        datalower[ii][jj].append(np.transpose(image))
                        datalower[ii][jj].append(np.transpose(errImage))

                        dataupper[ii][jj].append(np.transpose(image))
                        dataupper[ii][jj].append(np.transpose(errImage))
                    else:
                        image = marginalizeMC(fit,box,resolution=resolution,err=None,remainingAxes=[ii,jj],batchSize=1)
                        datalower[ii][jj].append(np.transpose(image))
                        
                        dataupper[ii][jj].append(np.transpose(image))

                else:
                    datalower[ii][jj].append(np.transpose(marginalizeOverPDF(pF,f,[ii,jj])))
                    dataupper[ii][jj].append(sliceArray(f,[ii,jj],minIndices))
                    if plotErr:
                        datalower[ii][jj].append(np.transpose(marginalizeOverPDF(pF,ferr,[ii,jj])))
                        dataupper[ii][jj].append(sliceArray(ferr,[ii,jj],minIndices))


        Ncornerplots = 2 if plotErr else 1

        figsize=(10,10)


        plt.close()
        with PdfPages(plotfn + '.pdf') as pdf:

            for III in range(Ncornerplots):

                fig, axes = plt.subplots(d+2,d+2,figsize=figsize)

                axes[0,0].set_visible(False)
                axes[-1,0].set_visible(False)
                axes[0,-1].set_visible(False)
                axes[-1,-1].set_visible(False)

                for ii in range(d):
                    dataIm = datalower1d[ii][III]
                    dataMin = dataupper1d[ii][III]

                    axes[-1,ii+1].plot(axisLists[ii], dataIm)
                    axes[0,ii+1].plot(axisLists[ii], dataMin)
                    axes[-1,ii+1].set_xlabel(labels[ii])

                    axes[0,ii+1].xaxis.set_label_position("top")
                    axes[0,ii+1].set_xlabel(labels[ii])
                    axes[0,ii+1].tick_params(axis='x',direction='in', pad = -15)

                    axes[ii+1,0].plot(dataIm, axisLists[ii])
                    axes[ii+1,-1].plot(dataMin, axisLists[ii])

                    axes[ii+1,0].set_ylabel(labels[ii])
                    axes[ii+1,0].set_xticklabels([])

                    axes[ii+1,-1].yaxis.set_label_position("right")
                    axes[ii+1,-1].set_ylabel(labels[ii])
                    axes[ii+1,-1].set_xticklabels([])
                    axes[ii+1,-1].tick_params(axis='y',direction='in', pad = -25)

                    axes[ii+1,ii+1].set_visible(False)

                for ii in range(d-1):
                    for jj in range(ii+1,d):

                        dataIm = datalower[ii][jj][III]
                        dataMin = dataupper[ii][jj][III]

                        chisqIm = datalower[ii][jj][0]
                        chisqMin = dataupper[ii][jj][0]


                        axes[jj+1,ii+1].set_xticklabels([])
                        axes[jj+1,ii+1].set_yticklabels([])

                        axes[ii+1,jj+1].set_xticklabels([])
                        axes[ii+1,jj+1].set_yticklabels([])

                        axes[jj+1,ii+1].imshow(np.flipud(dataIm),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet',vmin=np.min(dataIm),vmax=np.min(dataIm) + chisqCutoff)
                        axes[jj+1,ii+1].contour(chisqIm,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),levels = levels,colors='r')

                        axes[ii+1,jj+1].imshow(np.flipud(dataMin),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet',vmin=np.min(dataMin),vmax=np.min(dataMin) + chisqCutoff)
                        axes[ii+1,jj+1].contour(chisqMin,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),levels = levels,colors='r')

                fig.subplots_adjust(hspace=0.,wspace=0.)

                pdf.savefig()
                plt.close()

            for i in range(d):
                fig, axes = plt.subplots(1,figsize=figsize)
                axes.plot(axisLists[i], datalower1d[i][0])
                if plotErr:
                    axes.errorbar(axisLists[i], datalower1d[i][0],yerr=datalower1d[i][1])
                axes.set_xlabel(labels[i])
                axes.set_ylabel("Marginalized $\chi^2$")
                axes.set_ylim(np.min(datalower1d[i][0]), np.min(datalower1d[i][0]) + chisqCutoff)
                axes.set_title(labels[i] + " vs. $\chi^2$ after marginalizing")
                pdf.savefig()

                plt.close()
            for i in range(d):
                fig, axes = plt.subplots(1,figsize=figsize)

                axes.plot(axisLists[i], dataupper1d[i][0])
                if plotErr:
                    axes.errorbar(axisLists[i], dataupper1d[i][0],yerr=dataupper1d[i][1])
                axes.set_xlabel(labels[i])
                axes.set_ylabel("$\chi^2$")
                axes.set_ylim(np.min(dataupper1d[i][0]), np.min(dataupper1d[i][0]) + chisqCutoff)
                axes.set_title(labels[i] + " vs. $\chi^2$ through minimum (IN PROGRESS)")
                pdf.savefig()
                plt.close()

            for ii in range(d-1):
                for jj in range(ii+1,d):
                    fig, axes = plt.subplots(1,figsize=figsize)

                    chisqIm = datalower[ii][jj][0]
                    dataIm = datalower[ii][jj][0]

                    axes.imshow(np.flipud(dataIm),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet',vmin=np.min(dataIm),vmax=np.min(dataIm) + chisqCutoff)
                    axes.contour(chisqIm,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),levels = levels,colors='r')

                    axes.set_xlabel(labels[ii])
                    axes.set_ylabel(labels[jj])
                    axes.set_title(labels[ii] + " vs. "+labels[jj]+" after marginalization")
                    pdf.savefig()
                    plt.close()

            for ii in range(d-1):
                for jj in range(ii+1,d):
                    fig, axes = plt.subplots(1,figsize=figsize)

                    chisqIm = dataupper[ii][jj][0]
                    dataIm = dataupper[ii][jj][0]

                    axes.imshow(np.flipud(dataIm),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet',vmin=np.min(dataIm),vmax=np.min(dataIm) + chisqCutoff)
                    axes.contour(chisqIm,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),levels = levels,colors='r')

                    axes.set_xlabel(labels[ii])
                    axes.set_ylabel(labels[jj])
                    axes.set_title(labels[ii] + " vs. "+labels[jj]+" through minimum (IN PROGRESS)")
                    pdf.savefig()
                    plt.close()



    ## SAVE THE DATA
    if useMC:
        fname = datafn + '_res_' + str(resolution)
    else:
        fname = datafn

    p.dump( datalower1d, open( fname + '_lower_1d.p', "wb" ) )
    p.dump( dataupper1d, open( fname + '_upper_1d.p', "wb" ) )
    if plot:
        p.dump( datalower, open( fname + '_lower_2d.p', "wb" ) )
        p.dump( dataupper, open( fname + '_upper_2d.p', "wb" ) )

    return bestFits

def plotSlices(fit,box,plotRange,plotName,unslicedIndices=[0,1], nSlices = [10,10]):

    ## This scheme will only work for 4D systems

    fig, axes = plt.subplots(nSlices[0],nSlices[1])

    slicedAxes = set(range(len(box)))
    for i in set(unslicedIndices):
        slicedAxes.remove(i)
    slicedAxes = list(slicedAxes)

    slicedLists = []
    for i in range(2):
        slicedLists.append(np.linspace(plotRange[slicedAxes[i],0],plotRange[slicedAxes[i],1],num=nSlices[i]))


    imageSize = 50

    unslicedLists = []
    for i in range(2):
        unslicedLists.append(np.linspace(plotRange[unslicedIndices[i],0],plotRange[unslicedIndices[i],1],num=imageSize))


    print slicedLists


    chisqrange = - len(box) * np.log(1 - erf(5 / 2**0.5))



    fs = []
    for i in range(1000):
        print i
        pt = np.random.rand(1,len(box))
        xx = np.asarray(bb.unScalePoints(box,pt))
        func_min = minimize(fit,xx,bounds = box)
        fs.append(func_min.fun[0])





    # x0 = [0.5 * (b[1] + b[0]) for b in box]
    # func_min = minimize(fit,x0,bounds = box)
    # print func_min
    # print dir(func_min)

    # chisqmin = func_min.fun[0]
    chisqmin = np.min(fs)
    print chisqmin, chisqrange
    # chisqmin = 0
    # chisqrange = 1

    for i in range(nSlices[0]):
        for j in range(nSlices[1]):
            print i,j
            image = np.zeros((imageSize,imageSize))
            for k in range(imageSize):
                for l in range(imageSize):
                    params = np.zeros(len(box))
                    params[slicedAxes[0]] = slicedLists[0][i]
                    params[slicedAxes[1]] = slicedLists[1][j]
                    params[unslicedIndices[0]] = unslicedLists[0][k]
                    params[unslicedIndices[1]] = unslicedLists[1][l]
                    # print params
                    image[k,l] = fit([params])
            axes[i,j].imshow(np.flipud(np.transpose(image)),vmin=chisqmin,vmax=chisqmin + chisqrange, cmap = 'jet')
            axes[i,j].set_yticklabels([])
            axes[i,j].set_xticklabels([])

    fig.subplots_adjust(hspace=0.,wspace=0.)


    plt.savefig(plotName)

    plt.show()

def getGridDimensions(N,d):
    n = N**(1./d)
    n = int(n - (n % 1) + 1)
    N = int(n**d)
    return(N,n)


def marginalizePDF(PDF,remainingAxes):
    axes = set(range(len(PDF.shape)))
    for i in set(remainingAxes):
        axes.remove(i)
    mPDF = np.sum(PDF,axis=tuple(axes))
    mPDF = np.divide(mPDF,np.sum(mPDF))
    return(mPDF)


def sliceArray(data,remainingAxes,slicePoint):
    slicePoint = list(slicePoint)
    for i in remainingAxes:
        slicePoint[i] = slice(None)
    return(data[slicePoint])

def marginalizeOverPDF(PDF,fx,remainingAxes):
    axes = set(range(len(PDF.shape)))
    for i in set(remainingAxes):
        axes.remove(i)

    minPDF = np.min(PDF[np.nonzero(PDF)])
    PDF = np.add(PDF,minPDF)

    mFxPDF = np.sum(np.multiply(PDF,fx),axis=tuple(axes))
    mPDF = marginalizePDF(PDF,remainingAxes)

    return(np.divide(mFxPDF,mPDF))


def bestValFromPDF(PDF,axis):
    
    PDF = np.divide(PDF,np.sum(PDF))
    
    mean = np.sum(np.multiply(PDF,axis))
    
    stdev = np.sqrt(np.sum(np.multiply(PDF,np.power(np.subtract(axis,mean),2))))
    
    return mean, stdev

def plotNewPointsRBF(prevPoints,newPoints,plotfn,PDF_func=chisqToPDF,labels=None):
    ## This function will take a fit function which is defined over a space defined by box and return the best fit parameters over that space
    ## That fit function must take in a list of lists of parameters with a shape (n,d) and return a list with shape (n).
    
    ## box should be a list of [min,max] for each dimension, for a shape (d,2).
    ## There are several optional parameters.
    ##     plot is a Boolean, which indicates whether or not we should plot the pairwise marginal PDFs.
    ##     showPlot will use plt.show() to bring those plots to the screen while running.
    ##     plotfn is a string which gives the filename where we should save that plot (if it exists.)
    ##     labels should be a list of strings with length (d). These labels will be used on the plot.
    ##     searchRange is a list with the same shape as 'box' which describes the mins and maxs where we
    ##          will actually compute the PDF.
    ##     searchFraction describes the fraction of the box extent over which we will compute the PDF.
    ##     extent has the same shape as 'box' and overrides 'box' when returning values. It is assumed that
    ##          if 'box' describes a unit cube, then 'extent' includes the physical parameters which
    ##          correspond to that box. This param is a bit hacky so I'll try to make it nicer.
    ##     PDF_func is a function which turns the output of 'fit' into a member of a probability distribution.


    ## Get the Bayes fit

    fit = bb.getFit(prevPoints,fitkwargs={},method='rbf')

    ## Get the box shape
    box = bb.getBox(prevPoints[:,:-1])

    # Get the number of dimensions and find the smallest grid with more than a given number of points in that dimension.
    d = len(box)
    N,n = getGridDimensions(100000,d)

    axisLists = [np.linspace(b[0],b[1],n+2)[1:-1] for b in box]
    

    # Generate a function which will return a point in our space parameterized by a single index up to N.
    def gridder(X):
        params = np.zeros(d)
        ileft = X * 1
        for j in reversed(range(d)):
            params[j] = axisLists[j][ileft % n]
            ileft = (ileft - ileft % n) / n
        return params

    # Construct a list of points which comprise the grid.
    sPoints = np.asarray([gridder(j) for j in range(N)])

    # Determine the desired output shape and perform the fit function over the grid.
    s = [n for K in range(d)]
    f = fit(sPoints).reshape(s)

    # Turn the fit function grid into a PDF grid.
    pF = PDF_func(f,d)

    if labels == None:
        print("No labels provided! Reverting to default labelling.")
        labels = ["Parameter " + str(i+1) for i in range(d)]
    if len(labels) != d:
        print("The number of labels doesn't match the number of parameters! Reverting to default labelling.")
        labels = ["Parameter " + str(i+1) for i in range(d)]
    plt.close()

    chisqCutoff = - d * np.log(1 - erf(5 / 2**0.5))

    levels = [- d * np.log(1 - erf(II / 2**0.5)) for II in [1,2,3,4]]

    fig, axes = plt.subplots(d+1,d+1)
    
    for ii in range(d):
        for jj in range(d):
            axes[ii,jj].set_xticklabels([])
            
            if ii != d-1:
                axes[ii+1,jj+1].set_yticklabels([])


            # axes[ii+1,jj+1].set_yticklabels([])
            

            dist = marginalizePDF(pF,[ii,jj])
            chisqMarg = marginalizeOverPDF(pF,f,[ii,jj])
            


            if ii == jj:
                axes[d-ii-1,d-jj].set_visible(False)
                

                axes[-1,d-jj].plot(axisLists[ii], chisqMarg)
                

                axes[-1,d-jj].set_xlabel(labels[ii])
                

                axes[d-ii-1,0].plot(chisqMarg, axisLists[ii])
                

                axes[d-ii-1,0].set_ylabel(labels[ii])
                

            elif ii < jj:

                L = [np.min(chisqMarg) + level for level in levels]
                
                axes[d-ii-1,d-jj].imshow(np.flipud(chisqMarg),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet',vmin=np.min(chisqMarg),vmax=np.min(chisqMarg) + chisqCutoff)
                axes[d-ii-1,d-jj].contour(chisqMarg,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),colors = 'r', levels = L )
                # chisqIm = PDFtoChisq(dist,d)
                
                # print((min(axisLists[ii]),max(axisLists[ii])))
                axes[d-ii-1,d-jj].scatter(prevPoints[:,jj],prevPoints[:,ii],c='y',edgecolor='k')
                axes[d-ii-1,d-jj].scatter(newPoints[:,jj],newPoints[:,ii],c='r',edgecolor='k')
                axes[d-ii-1,d-jj].set_xlim(min(axisLists[jj]),max(axisLists[jj]))
                axes[d-ii-1,d-jj].set_ylim(min(axisLists[ii]),max(axisLists[ii]))

                
            else:
                axes[d-ii-1,d-jj].set_visible(False)
                

    fig.subplots_adjust(hspace=0.,wspace=0.)
    

    axes[-1,0].set_visible(False)
    

    fig.savefig(plotfn + "_fit.png")
    
    plt.close()

    pass

def plotNewPointsBayes(prevPoints,newPoints,plotfn,PDF_func=chisqToPDF,labels=None):
    ## This function will take a fit function which is defined over a space defined by box and return the best fit parameters over that space
    ## That fit function must take in a list of lists of parameters with a shape (n,d) and return a list with shape (n).
    
    ## box should be a list of [min,max] for each dimension, for a shape (d,2).
    ## There are several optional parameters.
    ##     plot is a Boolean, which indicates whether or not we should plot the pairwise marginal PDFs.
    ##     showPlot will use plt.show() to bring those plots to the screen while running.
    ##     plotfn is a string which gives the filename where we should save that plot (if it exists.)
    ##     labels should be a list of strings with length (d). These labels will be used on the plot.
    ##     searchRange is a list with the same shape as 'box' which describes the mins and maxs where we
    ##          will actually compute the PDF.
    ##     searchFraction describes the fraction of the box extent over which we will compute the PDF.
    ##     extent has the same shape as 'box' and overrides 'box' when returning values. It is assumed that
    ##          if 'box' describes a unit cube, then 'extent' includes the physical parameters which
    ##          correspond to that box. This param is a bit hacky so I'll try to make it nicer.
    ##     PDF_func is a function which turns the output of 'fit' into a member of a probability distribution.


    ## Get the Bayes fit

    fit, stdFit = bb.getFit(prevPoints,fitkwargs={'returnStd':True},method='bayes')

    # ptsForBox = np.append(prevPoints[:,:-1],newPoints,axis=0)

    ## Get the box shape
    box = bb.getBox(prevPoints[:,:-1])
    # box = bb.expandBox(box,0.1)
    # print 'box'
    # print box

    # d = len(box)
    # print box

    # Get the number of dimensions and find the smallest grid with more than a given number of points in that dimension.
    d = len(box)
    N,n = getGridDimensions(100000,d)

    axisLists = [np.linspace(b[0],b[1],n+2)[1:-1] for b in box]

    # Generate a function which will return a point in our space parameterized by a single index up to N.
    def gridder(X):
        params = np.zeros(d)
        ileft = X * 1
        for j in reversed(range(d)):
            params[j] = axisLists[j][ileft % n]
            ileft = (ileft - ileft % n) / n
        return params

    # Construct a list of points which comprise the grid.
    sPoints = np.asarray([gridder(j) for j in range(N)])

    # print sPoints
    # print box
    # Determine the desired output shape and perform the fit function over the grid.
    s = [n for K in range(d)]
    f = fit(sPoints).reshape(s)
    Sig = stdFit(sPoints).reshape(s)

    # print np.min(f),np.mean(f),np.max(f),np.std(f)

    # Turn the fit function grid into a PDF grid.
    pF = PDF_func(f,d)

    if labels == None:
        print("No labels provided! Reverting to default labelling.")
        labels = ["Parameter " + str(i+1) for i in range(d)]
    if len(labels) != d:
        print("The number of labels doesn't match the number of parameters! Reverting to default labelling.")
        labels = ["Parameter " + str(i+1) for i in range(d)]
    plt.close()

    chisqCutoff = - d * np.log(1 - erf(5 / 2**0.5))

    levels = [- d * np.log(1 - erf(II / 2**0.5)) for II in [1,2,3,4]]



    fig, axes = plt.subplots(d+1,d+1)
    figchisq, axeschisq = plt.subplots(d+1,d+1)
    for ii in range(d):
        for jj in range(d):
            axes[ii,jj].set_xticklabels([])
            axeschisq[ii,jj].set_xticklabels([])

            if ii != d-1:
                axes[ii+1,jj+1].set_yticklabels([])
                axeschisq[ii+1,jj+1].set_yticklabels([])

            dist = marginalizePDF(pF,[ii,jj])
            chisqMarg = marginalizeOverPDF(pF,f,[ii,jj])
            sigMarg = marginalizeOverPDF(pF,Sig,[ii,jj])


            if ii == jj:
                axes[d-ii-1,d-jj].set_visible(False)
                axeschisq[d-ii-1,d-jj].set_visible(False)

                axes[-1,d-jj].plot(axisLists[ii], chisqMarg)
                axeschisq[-1,d-jj].plot(axisLists[ii], sigMarg)

                axes[-1,d-jj].set_xlabel(labels[ii])
                axeschisq[-1,d-jj].set_xlabel(labels[ii])

                axes[d-ii-1,0].plot(chisqMarg, axisLists[ii])
                axeschisq[d-ii-1,0].plot(sigMarg, axisLists[ii])

                axes[d-ii-1,0].set_ylabel(labels[ii])
                axeschisq[d-ii-1,0].set_ylabel(labels[ii])

            elif ii < jj:
                
                L = [np.min(chisqMarg) + level for level in levels]
                
                axes[d-ii-1,d-jj].imshow(np.flipud(chisqMarg),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet',vmin=np.min(chisqMarg),vmax=np.min(chisqMarg) + chisqCutoff)
                axes[d-ii-1,d-jj].contour(chisqMarg,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),colors = 'r', levels = L )

                # axeschisq[d-ii-1,d-jj].imshow(np.flipud(dist),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet')#,vmin=np.min(sigMarg),vmax=np.min(sigMarg) + chisqCutoff)
                axeschisq[d-ii-1,d-jj].imshow(np.flipud(sigMarg),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet')#,vmin=np.min(sigMarg),vmax=np.min(sigMarg) + chisqCutoff)
                axeschisq[d-ii-1,d-jj].contour(chisqMarg,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),colors = 'r', levels = L )

                axes[d-ii-1,d-jj].scatter(prevPoints[:,jj],prevPoints[:,ii],c='y',edgecolor='k')
                axes[d-ii-1,d-jj].scatter(newPoints[:,jj],newPoints[:,ii],c='r',edgecolor='k')
                axes[d-ii-1,d-jj].set_xlim(min(axisLists[jj]),max(axisLists[jj]))
                axes[d-ii-1,d-jj].set_ylim(min(axisLists[ii]),max(axisLists[ii]))

                axeschisq[d-ii-1,d-jj].scatter(prevPoints[:,jj],prevPoints[:,ii],c='y',edgecolor='k')
                axeschisq[d-ii-1,d-jj].scatter(newPoints[:,jj],newPoints[:,ii],c='r',edgecolor='k')
                axeschisq[d-ii-1,d-jj].set_xlim(min(axisLists[jj]),max(axisLists[jj]))
                axeschisq[d-ii-1,d-jj].set_ylim(min(axisLists[ii]),max(axisLists[ii]))
            else:
                axes[d-ii-1,d-jj].set_visible(False)
                axeschisq[d-ii-1,d-jj].set_visible(False)

    fig.subplots_adjust(hspace=0.,wspace=0.)
    figchisq.subplots_adjust(hspace=0.,wspace=0.)

    axes[-1,0].set_visible(False)
    axeschisq[-1,0].set_visible(False)

    fig.savefig(plotfn + "_fit.png")
    figchisq.savefig(plotfn + "_fitErr.png")
    plt.show()
    plt.close()

    pass
