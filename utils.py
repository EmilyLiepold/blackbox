from scipy.optimize import minimize,fmin
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import blackbox as bb

def chisqToPDF(chisq,d):
	fRed = np.exp(-np.divide(np.subtract(chisq,np.min(chisq)),d))
	return np.divide(fRed,np.sum(fRed))

def PDFtoChisq(fRed,d):
	chisq = np.multiply(-np.log(fRed),d)
	return np.subtract(chisq,np.min(chisq))


def analyzeFit(fit,box,plot=True,showPlot=False,plotfn='fit',labels=None,searchRange = None,searchFraction=0.2,PDF_func=chisqToPDF,PDF_inv_func=PDFtoChisq):
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


	## if a search range hasn't been provided, construct one.
	if searchRange == None:
		# If the searchFraction is not just a number (it has a length), but doesn't have a value for each dimension, throw an error
		try:
			if len(searchFraction) != len(box):
				print "The dimensions in searchFraction and box don't match!"
		# If the searchFraction is just a number, then make it a list with one element per dimension.
		except:
			searchFraction = searchFraction * np.ones(len(box))

		# If the searchFraction is greater than 1, just make it 1.
		searchFraction = [min(s,1.) for s in searchFraction]

		# Find the center of the box and determine what values correspond to the search fraction.
		boxCenter = [0.5 * (b[1] + b[0]) for b in box]
		paramRanges = [searchFraction[i] * (b[1] - b[0]) for i,b in enumerate(box)]

		# Find the minimum of the the function over the space,
		func_min = minimize(fit,boxCenter,bounds=box).x

		# Contruct the searchRange by moving paramRanges to the left and right of the minimum
		# This keeps us from scanning over the entire box unless that's necessary.

		searchRange = []
		for i in range(len(box)):
			paramMin = max(func_min[i] -paramRanges[i],box[i][0])
			paramMax = min(func_min[i] + paramRanges[i],box[i][1])
			searchRange.append([paramMin,paramMax])

	# Just to be safe, make everything a ndarray.
	searchRange = np.asarray(searchRange)
	# print searchRange

	# Get the number of dimensions and find the smallest grid with more than a given number of points in that dimension.
	d = len(box)
	N,n = getGridDimensions(100000,d)

	# Determine the values which correspond to the points in the grid
	# try:
	# 	len(extent)
	# 	realBounds = extent[:]
	# 	for i in range(len(extent)):
	# 		for ii in range(2):
	# 			realBounds[i][ii] = extent[i][0] + searchRange[i][ii] * (extent[i][1] - extent[i][0])
	# except:
	# 	realBounds = searchRange[:]

	axisLists = [np.linspace(s[0],s[1],n) for s in searchRange]
	

	# Generate a function which will return a point in our space parameterized by a single index up to N.
	def gridder(X):
	    params = np.zeros(d)
	    ileft = X * 1
	    for j in reversed(range(d)):
	        params[j] = ((ileft % n) / (n - 1.0)) * (searchRange[j][1] - searchRange[j][0]) + searchRange[j][0]
	        ileft = (ileft - ileft % n) / n
	    return params

	# Construct a list of points which comprise the grid.
	sPoints = np.asarray([gridder(j) for j in range(N)])

	# Determine the desired output shape and perform the fit function over the grid.
	s = [n for K in range(d)]
	f = fit(sPoints).reshape(s)

	# Turn the fit function grid into a PDF grid.
	pF = PDF_func(f,d)

	# Marginalize over each dimension to find the mean and stdev over those dimensions.
	bestFits = np.asarray([bestValFromPDF(marginalizePDF(pF,[i,i]),axisLists[i]) for i in range(d)])
	
	# Since we are only looking over part of the box, we may need to zoom out to get a better picture of the PDF.
	# Scan over the parameters and find any where the 3 sigma bounds are not within the search region.
	rerunParams = []
	for i in range(d):
		if (searchRange[i,0] > bestFits[i,0] - 3 * bestFits[i,1]) or (searchRange[i,1] < bestFits[i,0] + 3 * bestFits[i,1]):

			# If the 3 sigma bound is outside the search range and the search range does not span the box, then 
			# plan to rerun with a broader search.
			if not np.array_equal(np.asarray(searchRange)[i],np.asarray(box)[i]):
				rerunParams.append(i)

	# If some parameters should be expanded, increase the search fraction and rerun.
	if len(rerunParams) > 0:
		newsearchFraction = searchFraction[:]

		for i in rerunParams:
			newsearchFraction[i] *= 2
		return analyzeFit(fit,box,plot=plot,plotfn=plotfn,labels=labels,searchRange = None,searchFraction = newsearchFraction,showPlot=showPlot,PDF_func=PDF_func)

	# If all of the parameters are good and we want to plot, then plot!
	elif plot:
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

				axes[ii+1,jj+1].set_yticklabels([])
				axeschisq[ii+1,jj+1].set_yticklabels([])

				dist = marginalizePDF(pF,[ii,jj])

				if ii == jj:
					axes[d-ii-1,d-jj].set_visible(False)
					axeschisq[d-ii-1,d-jj].set_visible(False)

					axes[-1,d-jj].plot(axisLists[ii], dist)
					axeschisq[-1,d-jj].plot(axisLists[ii], PDFtoChisq(dist,d))

					axes[-1,d-jj].set_xlabel(labels[ii])
					axeschisq[-1,d-jj].set_xlabel(labels[ii])

					axes[d-ii-1,0].plot(dist, axisLists[ii])
					axeschisq[d-ii-1,0].plot(PDFtoChisq(dist,d), axisLists[ii])

					axes[d-ii-1,0].set_ylabel(labels[ii])
					axeschisq[d-ii-1,0].set_ylabel(labels[ii])

				elif ii < jj:
					axes[d-ii-1,d-jj].imshow(np.flipud(dist),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet')
					chisqIm = PDFtoChisq(dist,d)
					axeschisq[d-ii-1,d-jj].imshow(np.flipud(chisqIm),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',vmin=np.min(chisqIm),vmax=np.min(chisqIm) + chisqCutoff,cmap='jet')
					axeschisq[d-ii-1,d-jj].contour(chisqIm,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),levels = levels,colors='r')
				
				else:
					axes[d-ii-1,d-jj].set_visible(False)
					axeschisq[d-ii-1,d-jj].set_visible(False)

		fig.subplots_adjust(hspace=0.,wspace=0.)
		figchisq.subplots_adjust(hspace=0.,wspace=0.)

		axes[-1,0].set_visible(False)
		axeschisq[-1,0].set_visible(False)

		fig.savefig(plotfn + ".png")
		figchisq.savefig(plotfn + "_chisq.png")
		# print showPlot
		if showPlot:
			plt.show()
		plt.close()

		# try:
		# 	bestFits[:,0] = np.add(np.multiply(bestFits[:,1],np.subtract(extent[:,1],extent[:,0])),extent[:,0])
		# 	bestFits[:,1] = np.multiply(bestFits[:,1],np.subtract(extent[:,1],extent[:,0]))
		# except:
		# 	extent = None
			# bestFits[:,0] = np.add(np.multiply(bestFits[:,1],np.subtract(realBounds[:,1],realBounds[:,0])),realBounds[:,0])
			# bestFits[:,1] = np.multiply(bestFits[:,1],np.subtract(realBounds[:,1],realBounds[:,0]))


		
	return bestFits

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

def marginalizeOverPDF(PDF,fx,remainingAxes):
	axes = set(range(len(PDF.shape)))
	for i in set(remainingAxes):
		axes.remove(i)

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

	fit = bb.getFit(prevPoints)

	## Get the box shape
	box = bb.getBox(prevPoints[:,:-1])

	# d = len(box)
	# print box

	# Get the number of dimensions and find the smallest grid with more than a given number of points in that dimension.
	d = len(box)
	N,n = getGridDimensions(100000,d)

	# Determine the values which correspond to the points in the grid
	# try:
	# 	len(extent)
	# 	realBounds = extent[:]
	# 	for i in range(len(extent)):
	# 		for ii in range(2):
	# 			realBounds[i][ii] = extent[i][0] + searchRange[i][ii] * (extent[i][1] - extent[i][0])
	# except:inpoints[:,:-1] = unScalePoints(box, inpoints[:,:-1])
	# 	realBounds = searchRange[:]

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
	# Determine the desired output shape and perform the fit function over the grid.
	s = [n for K in range(d)]
	f = fit(sPoints).reshape(s)

	# Turn the fit function grid into a PDF grid.
	pF = PDF_func(f,d)

	# # Marginalize over each dimension to find the mean and stdev over those dimensions.
	# bestFits = np.asarray([bestValFromPDF(marginalizePDF(pF,[i,i]),axisLists[i]) for i in range(d)])
	
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
			

			axes[ii+1,jj+1].set_yticklabels([])
			

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
	
	# print showPlot
	# if showPlot:
	# plt.show()
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

	fit, stdFit = bb.getFit(prevPoints,method='bayes',returnStd=True)

	## Get the box shape
	box = bb.getBox(prevPoints[:,:-1])

	# d = len(box)
	# print box

	# Get the number of dimensions and find the smallest grid with more than a given number of points in that dimension.
	d = len(box)
	N,n = getGridDimensions(100000,d)

	# Determine the values which correspond to the points in the grid
	# try:
	# 	len(extent)
	# 	realBounds = extent[:]
	# 	for i in range(len(extent)):
	# 		for ii in range(2):
	# 			realBounds[i][ii] = extent[i][0] + searchRange[i][ii] * (extent[i][1] - extent[i][0])
	# except:inpoints[:,:-1] = unScalePoints(box, inpoints[:,:-1])
	# 	realBounds = searchRange[:]

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
	# Determine the desired output shape and perform the fit function over the grid.
	s = [n for K in range(d)]
	f = fit(sPoints).reshape(s)
	Sig = stdFit(sPoints).reshape(s)

	# Turn the fit function grid into a PDF grid.
	pF = PDF_func(f,d)

	# # Marginalize over each dimension to find the mean and stdev over those dimensions.
	# bestFits = np.asarray([bestValFromPDF(marginalizePDF(pF,[i,i]),axisLists[i]) for i in range(d)])
	
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

				# chisqIm = PDFtoChisq(dist,d)
				axeschisq[d-ii-1,d-jj].imshow(np.flipud(sigMarg),extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])), aspect='auto',cmap='jet')#,vmin=np.min(sigMarg),vmax=np.min(sigMarg) + chisqCutoff)
				axeschisq[d-ii-1,d-jj].contour(chisqMarg,extent=(min(axisLists[jj]),max(axisLists[jj]),min(axisLists[ii]),max(axisLists[ii])),colors = 'r', levels = L )
				# print((min(axisLists[ii]),max(axisLists[ii])))
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
	# print showPlot
	# if showPlot:
	# plt.show()
	plt.close()

	pass

