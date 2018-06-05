import numpy
from scipy.interpolate import interp2d, LinearNDInterpolator

logMBH, logC, logFDM, ML, chi2 = numpy.loadtxt('/home/cmliepold/Documents/CPMa/Sampling_project/Blackbox/utils/1277.dat',usecols=(0,5,6,7,8),skiprows=1,unpack=True)

# Pick out the different values of the dark matter fraction

w3 = numpy.where(logFDM == 3)[0]
w2 = numpy.where(logFDM == 2)[0]
w1 = numpy.where(logFDM == 1)[0]
w0 = numpy.where(logFDM == 0)[0]

# Convert the black hole masses into the correct quantities

reallogMBH = numpy.log10(numpy.multiply(numpy.divide(ML,6.),numpy.power(10.,logMBH)))

## Construct a function which interpolates the scatter data into a continuous function.

## First, do this in the case that we only have a pair of parameters and fix f_DM = 3

rlogMBH = reallogMBH[w3]
rML = ML[w3]
rchi2 = chi2[w3]

## Construct the interpolator
chi2fn = interp2d(rlogMBH,rML,rchi2)

def chi2_2d(params):
	## This function takes in a list with length 2 and sends that list to the interpolator.
	## It returns a float.
	return chi2fn(*params)

def chi2_2d_vec(paramlist):
	## This function takes in a list of lists. Each of the elements of the list should have length 2
	## Those elements will be sent to the interpolator function.
	## If the input is one-dimensional, that single list will be sent along to the interpolator.

	## If the input is one-dimensional, then a list containing a single float is returned.
	## If the input is two-dimensional, then a list of floats is returned.

	if len(numpy.shape(numpy.asarray(paramlist))) == 1:
		return numpy.asarray(chi2_2d(paramlist)).reshape(-1,1)
	return numpy.asarray([chi2_2d(param) for param in paramlist])


## Now let's repeat for the three-dimensional case.

# Shape the input lists for the interpolator
datacube = numpy.asarray([reallogMBH,ML,logFDM]).T

# Construct the interpolation
chi2_3d_fn = LinearNDInterpolator(datacube,chi2)

def chi2_3d(params):
	## This function takes in a list with length 3 and sends that list to the interpolator.
	## It returns a float.
    return chi2_3d_fn(*params)

def chi2_3d_vec(paramlist):
		## This function takes in a list of lists. Each of the elements of the list should have length 3
	## Those elements will be sent to the interpolator function.
	## If the input is one-dimensional, that single list will be sent along to the interpolator.

	## If the input is one-dimensional, then a list containing a single float is returned.
	## If the input is two-dimensional, then a list of floats is returned.

	if len(numpy.shape(numpy.asarray(paramlist))) == 1:
		return numpy.asarray(chi2_3d(paramlist)).reshape(-1,1)
	return numpy.asarray([chi2_3d(param) for param in paramlist])