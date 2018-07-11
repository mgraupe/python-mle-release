from scipy import stats
import numpy as np
from scipy.optimize import minimize
import pylab as plt


lat1 = np.random.exponential(scale=0.5,size=1000)
lat2 = np.random.exponential(scale=5.,size=1000)

latencies = np.concatenate((lat1,lat2))

cc, bbRaw = np.histogram(latencies,bins=1000,density=True)
bb = (bbRaw[:-1]+bbRaw[1:])/2.

sortedLate = np.sort(latencies)
cumul  = np.cumsum(np.ones(len(sortedLate)))/len(sortedLate)

#ydata = np.array([0.1,0.15,0.2,0.3,0.7,0.8,0.9, 0.9, 0.95])
#xdata = np.array(range(0,len(ydata),1))

def doubleExponentialCumul(xdata,t1,t2,beta):
    return beta*(1. - np.exp(-xdata/t1)) + (1.-beta)*(1. - np.exp(-xdata/t2))

def doubleExponential(xdata,k1,k2,theta):
    return theta*np.exp(-xdata/k1)/k1 + (1.-theta)*np.exp(-xdata/k2)/k2

def doubleExponentialCumulLLE(params):
    t1 = params[0]
    t2 = params[1]
    beta = params[2]
    sd = params[3]

    yPredC = doubleExponentialCumul(sortedLate,t1,t2,beta)
    #yPred = 1 / (1+ np.exp(-k*(xdata-x0)))

    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf(cumul, loc=yPredC, scale=sd ) )
    return(LL)

def doubleExponentialLLE(params):
    k1 = params[0]
    k2 = params[1]
    theta = params[2]
    #amp = params[3]
    sd = params[3]

    yPred = doubleExponential(bb,k1,k2,theta)
    #yPred = 1 / (1+ np.exp(-k*(xdata-x0)))

    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf(cc, loc=yPred, scale=sd ) )

    return(LL)


initParamsDECumul = [0.1, 3., 0.6, 0.1]

resultsDEC = minimize(doubleExponentialCumulLLE, initParamsDECumul, method='Nelder-Mead')
print 'cumulative double exponential fit results', resultsDEC.x
print 'cumulative double exponential MLE :', doubleExponentialCumulLLE(resultsDEC.x)

initParamsDE = [0.1, 3., 0.6, 0.1]
resultsDE = minimize(doubleExponentialLLE, initParamsDE, method='Nelder-Mead')
print 'histogram fit results :', resultsDE.x
print 'histogram double exponential MLE :', doubleExponentialLLE(resultsDE.x)


estParmsDEC = resultsDEC.x
yOutDEC =  doubleExponentialCumul(sortedLate,estParmsDEC[0],estParmsDEC[1],estParmsDEC[2]) # 1 / (1+ np.exp(-estParms[0]*(xdata-estParms[1])))

estParmsDE = resultsDE.x
yOutDE =  doubleExponential(bb,estParmsDE[0],estParmsDE[1],estParmsDE[2]) # 1 / (1+ np.exp(-estParms[0]*(xdata-estParms[1])))

plt.clf()
plt.plot(bb,cc, 'o')
plt.plot(sortedLate,cumul, 'o')
plt.plot(sortedLate, yOutDEC)
plt.plot(bb, yOutDE)
#plt.hist(cc-yOutDE)
plt.show()

