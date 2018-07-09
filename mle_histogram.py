from scipy import stats
import numpy as np
from scipy.optimize import minimize
import pylab as plt

latencies = np.loadtxt('latencies.txt')

cc, bbRaw = np.histogram(latencies,bins=50,density=True)
bb = (bbRaw[:1]+bbRaw[:-1])/2.

#ydata = np.array([0.1,0.15,0.2,0.3,0.7,0.8,0.9, 0.9, 0.95])
#xdata = np.array(range(0,len(ydata),1))

def singleExponential(xdata,k):
    return np.exp(-k*xdata)

def doubleExponential(xdata,k1,k2,theta):
    return theta*np.exp(-k1*xdata) + (1.-theta)*np.exp(-k2*xdata)

def singleExponentialLLE(params):
    k = params[0]
    #x0 = params[1]
    sd = params[1]

    yPred = singleExponential(bb,k)
    #yPred = 1 / (1+ np.exp(-k*(xdata-x0)))

    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf(cc, loc=yPred, scale=sd ) )

    return(LL)

def doubleExponentialLLE(params):
    k1 = params[0]
    k2 = params[1]
    theta = params[2]
    sd = params[3]

    yPred = doubleExponential(bb,k1,k2,theta)
    #yPred = 1 / (1+ np.exp(-k*(xdata-x0)))

    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf(cc, loc=yPred, scale=sd ) )

    return(LL)


initParamsSE = [1., 0.2]

resultsSE = minimize(singleExponentialLLE, initParamsSE, method='Nelder-Mead')
print resultsSE.x
print 'single exponential :', singleExponentialLLE(resultsSE.x)

initParamsDE = [0.1, 3., 0.5, 0.1]

resultsDE = minimize(doubleExponentialLLE, initParamsDE, method='Nelder-Mead')
print resultsDE.x
print 'double exponential :', doubleExponentialLLE(resultsDE.x)


estParmsSE = resultsSE.x
yOutSE =  singleExponential(bb,estParmsSE[0]) # 1 / (1+ np.exp(-estParms[0]*(xdata-estParms[1])))

estParmsDE = resultsDE.x
yOutDE =  doubleExponential(bb,estParmsDE[0],estParmsDE[1],estParmsDE[2]) # 1 / (1+ np.exp(-estParms[0]*(xdata-estParms[1])))

plt.clf()
plt.plot(bb,cc, 'go')
plt.plot(bb, yOutSE)
plt.plot(bb, yOutDE)
plt.show()

