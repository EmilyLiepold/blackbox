import numpy
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import time
import blackbox as bb

def areaFromPath(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a


def checkForConvergence(fit,prevFit,fmax,prevFmax,d):

    d = 2

    fitImage = numpy.zeros((200,200))
    prevFitImage = numpy.zeros((200,200))
    for XX in range(200):
        for YY in range(200):
            fitImage[XX,YY] = fit(numpy.asarray([XX / 200., YY / 200.]))
            prevFitImage[XX,YY] = prevFit(numpy.asarray([XX / 200., YY / 200.]))

    fitMin = numpy.min(fitImage)
    prevFitMin = numpy.min(prevFitImage)


    confLevels = [0.,2.3,6.2,11.8]
    levels = [(fitMin * fmax + l) / fmax for l in confLevels]
    prevlevels = [(fitMin * prevFmax + l) / prevFmax for l in confLevels]

    colors = ['r','g','y','m']
    

    fig, axes = plt.subplots(2)

    axes[0].imshow(numpy.flipud(numpy.transpose(fitImage)),aspect='auto',vmin=fitMin,vmax=(fitMin + 40 / fmax))
    newcontour = axes[0].contour(numpy.flipud(numpy.transpose(fitImage)),levels=levels[1:],colors=colors)#,colors=[0.1 + colorIndex / 10.])#,levels=[fitMin,fitMin+2.3, fitMin + 6.2, fitMin + 11.8],colors='r')
    axes[1].imshow(numpy.flipud(numpy.transpose(prevFitImage)),aspect='auto',vmin=prevFitMin,vmax=(prevFitMin + 40 / prevFmax))
    oldcontour = axes[1].contour(numpy.flipud(numpy.transpose(prevFitImage)),levels=prevlevels[1:],colors=colors)
    plt.close()

    newarea = areaFromPath(newcontour.collections[1].get_paths()[0].vertices)
    oldarea = areaFromPath(oldcontour.collections[1].get_paths()[0].vertices)
    print oldarea, newarea, abs(2. * (newarea - oldarea) / (newarea + oldarea))
    if abs(2. * (newarea - oldarea) / (newarea + oldarea)) < 0.01:
        return True
    return False



def checkForConvergenceIntegration(fit,prevFit,fmax,prevFmax,d):


    N = 100000

    n = (N)**(1./d)

    n = int(n + 1 - (n % 1))

    N = n**d

    oldMinimum, newMin, iter, funcalls, warnflag = fmin(fit,[0.5 for k in range(d)], full_output=True, disp=False)
    newMinimum, oldMin, iter, funcalls, warnflag = fmin(prevFit,[0.5 for k in range(d)], full_output=True, disp=False)

    boxBounds = [[max(min(oldMinimum[i],newMinimum[i])-0.1,0.),min(max(oldMinimum[i],newMinimum[i])+0.1, 1.0)] for i in range(d)]


    def gridder(i):
        params = numpy.zeros(d)
        ileft = i * 1
        for j in range(d):
            params[j] = ((ileft % n) / (n - 1.0)) * (boxBounds[j][1] - boxBounds[j][0]) + boxBounds[j][0]
            ileft = (ileft - ileft % n) / n
        return params


    # fitImage = numpy.zeros((200,200))
    # prevFitImage = numpy.zeros((200,200))
    # for XX in range(200):
    #     for YY in range(200):
    #         fitImage[XX,YY] = fit(numpy.asarray([XX / 200., YY / 200.]))
    #         prevFitImage[XX,YY] = prevFit(numpy.asarray([XX / 200., YY / 200.]))

    # fitMin = numpy.min(fitImage)
    # prevFitMin = numpy.min(prevFitImage)


    confLevels = [0.,2.3,6.2,11.8]
    levels = [(newMin + l / fmax) for l in confLevels]
    prevlevels = [(oldMin + l/ prevFmax) for l in confLevels]

    totalPoints = 0
    oldPoints = 0
    newPoints = 0
    # indices = numpy.arange(N)
    # numpy.random.shuffle(indices)
    # batches = 100
    points = numpy.asarray([gridder(i) for i in range(N)])
    # print points
    # t = time.time()
    # print points.shape
    f = fit(points)
    pf = prevFit(points)
    nf = len(numpy.where(f[:] <= levels[1])[0])
    npf = len(numpy.where(pf[:] <= prevlevels[1])[0])
    # print time.time() - t
    # print len(nf),len(npf),len(points)

    #####################
    # Testing Fit speed #
    #####################
    # t = time.time()
    # print 'hey'
    # t = time.time()
    # points = numpy.asarray([gridder(i) for i in range(N)])
    # print time.time() - t
    # print points.shape
    # t = time.time()
    # print 'heyhey'
    # f = fit(points)
    # print f.shape
    # print time.time() - t
    # t = time.time()
    # for i in range(N):
    #     point = gridder(i)
    #     points = numpy.asarray([point for j in range(50)])
    #     # print points.shape
    #     f = fit(points)
    #     # f = oldfit(point)
    #     if i % 10000 == 0:
    #         print time.time() - t
    #         t = time.time()




    newarea = nf * 1.0 / len(points)
    oldarea = npf * 1.0 / len(points)


    # colors = ['r','g','y','m']
    

    # fig, axes = plt.subplots(2)

    # axes[0].imshow(numpy.flipud(numpy.transpose(fitImage)),aspect='auto',vmin=fitMin,vmax=(fitMin + 40 / fmax))
    # newcontour = axes[0].contour(numpy.flipud(numpy.transpose(fitImage)),levels=levels[1:],colors=colors)#,colors=[0.1 + colorIndex / 10.])#,levels=[fitMin,fitMin+2.3, fitMin + 6.2, fitMin + 11.8],colors='r')
    # axes[1].imshow(numpy.flipud(numpy.transpose(prevFitImage)),aspect='auto',vmin=prevFitMin,vmax=(prevFitMin + 40 / prevFmax))
    # oldcontour = axes[1].contour(numpy.flipud(numpy.transpose(prevFitImage)),levels=prevlevels[1:],colors=colors)
    # plt.close()

    # newarea = areaFromPath(newcontour.collections[1].get_paths()[0].vertices)
    # oldarea = areaFromPath(oldcontour.collections[1].get_paths()[0].vertices)
    print oldarea, newarea, abs(2. * (newarea - oldarea) / (newarea + oldarea))
    if abs(2. * (newarea - oldarea) / (newarea + oldarea)) < 0.01:
        print 'Converged!'
        return True
    return False


def plotFit(fit,points,fmax,fname):
    if len(points[0]) != 3:
        print 'Sorry, I only know how to do 2D plots right now'
        return -1

    bounds = bb.getBox(points[:,:-1])
    extent = [y for x in bounds for y in x]

    d = len(points[0]) - 1

    fitImage = numpy.zeros((200,200))
    n = 200

    def gridder(i):
        params = numpy.zeros(d)
        ileft = i * 1
        for j in range(d):
            params[j] = ((ileft % n) / (n - 1.0)) * (bounds[j][1] - bounds[j][0]) + bounds[j][0]
            ileft = (ileft - ileft % n) / n
        return params

    grid = numpy.asarray([gridder(i) for i in range(n**2)])

    f = fit(grid)

    for i in range(200):
        for j in range(200):
            fitImage[i,j] = f[200 * i + j]

    confLevels = [0.,2.3,6.2,11.8]
    levels = [(numpy.min(fitImage) + l / fmax) for l in confLevels]

    plt.close()
    plt.imshow(numpy.flipud(fitImage),extent=[0.,1.,0.,1.,],vmin=numpy.min(fitImage), vmax=numpy.min(fitImage) + 40./fmax)
    plt.contour(numpy.linspace(0.,1.,num=200),numpy.linspace(0.,1.,num=200), fitImage,levels=levels,colors='r')
    plt.scatter(points[:,0],points[:,1])

    plt.savefig(fname)
