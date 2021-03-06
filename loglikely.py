from scipy import stats
import numpy as np
from scipy.optimize import minimize
import pylab as plt
import scipy

latencies = np.loadtxt('latencies.txt')

sortedLate = np.sort(latencies)
cumul  = np.cumsum(np.ones(len(sortedLate)))/len(sortedLate)

polycoeffs = scipy.polyfit(sortedLate[sortedLate>30], cumul[sortedLate>30], 1)
print polycoeffs
yfit = scipy.polyval(polycoeffs, sortedLate)

#ydata = np.array([0.1,0.15,0.2,0.3,0.7,0.8,0.9, 0.9, 0.95])
#xdata = np.array(range(0,len(ydata),1))

def singleExponential(xdata,k):
    return (1. - np.exp(-xdata/k))

def doubleExponential(xdata,k1,k2,theta):
    return theta*(1. - np.exp(-xdata/k1)) + (1.-theta)*(1. - np.exp(-xdata/k2))

def singleExponentialLLE(params):
    k = params[0]
    #x0 = params[1]
    sd = params[1]

    yPred = singleExponential(sortedLate[sortedLate<30],k)
    #yPred = 1 / (1+ np.exp(-k*(xdata-x0)))

    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf((cumul-yfit)[sortedLate<30]+1., loc=yPred, scale=sd ) )

    return(LL)

def doubleExponentialLLE(params):
    k1 = params[0]
    k2 = params[1]
    theta = params[2]
    sd = params[3]

    yPred = doubleExponential(sortedLate[sortedLate<30],k1,k2,theta)
    #yPred = 1 / (1+ np.exp(-k*(xdata-x0)))

    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf((cumul-yfit)[sortedLate<30]+1., loc=yPred, scale=sd ) )

    return(LL)


initParamsSE = [1, 1]

resultsSE = minimize(singleExponentialLLE, initParamsSE, method='Nelder-Mead')
print resultsSE.x
print 'single exponential :', singleExponentialLLE(resultsSE.x)

initParamsDE = [2., 30., 0.1, 1]

resultsDE = minimize(doubleExponentialLLE, initParamsDE, method='Nelder-Mead')
print resultsDE.x
print 'double exponential :', doubleExponentialLLE(resultsDE.x)


estParmsSE = resultsSE.x
yOutSE =  singleExponential(sortedLate,estParmsSE[0]) # 1 / (1+ np.exp(-estParms[0]*(xdata-estParms[1])))

estParmsDE = resultsDE.x
yOutDE =  doubleExponential(sortedLate,estParmsDE[0],estParmsDE[1],estParmsDE[2]) # 1 / (1+ np.exp(-estParms[0]*(xdata-estParms[1])))

plt.clf()
plt.plot(sortedLate,(cumul-yfit)+1., 'go')
plt.plot(sortedLate, yOutSE)
plt.plot(sortedLate, yOutDE)
plt.plot(sortedLate,yfit)
plt.show()

