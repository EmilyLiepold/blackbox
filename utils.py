from scipy.optimize import minimize,fmin
import numpy as np
import matplotlib.pyplot as plt



def analyzeFit(fit,box,plot=True,plotfn='fit.png',labels=None,searchRange = None,searchFraction=0.2):

	if searchRange == None:
		try:
			if len(searchFraction) != len(box):
				print "The dimensions in searchFraction and box don't match!"
		except:
			searchFraction = searchFraction * np.ones(len(box))

		searchFraction = [min(s,1.) for s in searchFraction]
		boxCenter = [0.5 * (b[1] + b[0]) for b in box]
		paramRanges = [searchFraction[i] * (b[1] - b[0]) for i,b in enumerate(box)]
		func_min = minimize(fit,boxCenter,bounds=box).x


		searchRange = []
		for i in range(len(box)):
			paramMin = max(func_min[i] -paramRanges[i],box[i][0])
			paramMax = min(func_min[i] + paramRanges[i],box[i][1])
			searchRange.append([paramMin,paramMax])

		searchRange = np.asarray(searchRange)


	d = len(box)
	N = 100000.
	n = N**(1./d)
	n = int(n - (n % 1) + 1)
	N = int(n**d)


	axisLists = [np.linspace(s[0],s[1],n) for s in searchRange]

	def gridder(X):
	    params = np.zeros(d)
	    ileft = X * 1
	    for j in range(d):
	        params[j] = ((ileft % n) / (n - 1.0)) * (searchRange[j][1] - searchRange[j][0]) + searchRange[j][0]
	        ileft = (ileft - ileft % n) / n
	    return params

	sPoints = np.asarray([gridder(j) for j in range(N)])

	s = [n for K in range(d)]
	f = fit(sPoints).reshape(s)

	fRed = np.exp(-np.divide(np.subtract(f,np.min(f)),d))
	fRed = np.divide(fRed,np.sum(fRed))
	pF = fRed


	if plot:
		if labels == None:
			print("No labels provided! Reverting to default labelling.")
			labels = ["Parameter " + str(i+1) for i in range(d)]
		if len(labels) != d:
			print("The number of labels doesn't match the number of parameters! Reverting to default labelling.")
			labels = ["Parameter " + str(i+1) for i in range(d)]
		plt.close()
		fig, axes = plt.subplots(d+1,d+1)
		for ii in range(d):
			for jj in range(d):
				axes[ii,jj].set_xticklabels([])
				axes[ii+1,jj+1].set_yticklabels([])
				dist = marginalizePDF(pF,[ii,jj])
				if ii == jj:
					axes[d-ii-1,d-jj].set_visible(False)
					axes[-1,d-jj].plot(axisLists[ii], dist)
					axes[-1,d-jj].set_xlabel(labels[ii])
					axes[d-ii-1,0].plot(dist, axisLists[ii])
					axes[d-ii-1,0].set_ylabel(labels[ii])
				elif ii < jj:
					axes[d-ii-1,d-jj].imshow(np.flipud(dist),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto')
				else:
					axes[d-ii-1,d-jj].set_visible(False)

		fig.subplots_adjust(hspace=0.,wspace=0.)
		axes[-1,0].set_visible(False)
		plt.savefig(plotfn)
		plt.close()


	bestFits = [bestValFromPDF(marginalizePDF(pF,[i,i]),axisLists[i]) for i in range(d)]


	rerunParams = []
	for i in range(d):
		if (searchRange[i,0] > bestFits[i][0] - 3 * bestFits[i][1]) or (searchRange[i,1] < bestFits[i][0] + 3 * bestFits[i][1]):
			if not np.array_equal(np.asarray(searchRange)[i],np.asarray(box)[i]):
				rerunParams.append(i)

	if len(rerunParams) > 0:
		newsearchFraction = searchFraction[:]

		for i in rerunParams:
			newsearchFraction[i] *= 2
		return analyzeFit(fit,box,plot=plot,plotfn=plotfn,labels=labels,searchRange = None,searchFraction = newsearchFraction)

	return bestFits


def marginalizePDF(PDF,remainingAxes):
	axes = set(range(len(PDF.shape)))
	for i in set(remainingAxes):
		axes.remove(i)
	mPDF = np.sum(PDF,axis=tuple(axes))
	mPDF = np.divide(mPDF,np.sum(mPDF))
	return(mPDF)

def bestValFromPDF(PDF,axis):
	
	PDF = np.divide(PDF,np.sum(PDF))
	
	mean = np.sum(np.multiply(PDF,axis))
	
	stdev = np.sqrt(np.sum(np.multiply(PDF,np.power(np.subtract(axis,mean),2))))
	
	return mean, stdev