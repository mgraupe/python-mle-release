import numpy as np
from mle import var, Normal
import matplotlib.pyplot as plt
import pdb

latencies = np.loadtxt('latencies.txt')

sortedLate = np.sort(latencies)
cumul  = np.cumsum(np.ones(len(sortedLate)))/len(sortedLate)



#pdb.set_trace()

# Define model
x = var('x', observed=True, vector=True)
y = var('y', observed=True, vector=True)

k = var('k')
sigma1 = var('sigma1')

model1 = Normal(y, (1.-np.exp(-k*x)), sigma1)

k1 = var('k1')
k2 = var('k2')
theta = var('theta')
sigma2 = var('sigma2')

model2 = Normal(y,( theta*(1.-np.exp(-k1*x)) + (1.-theta)*(1.-np.exp(-k2*x)) ), sigma2)
# Generate data
#xs = np.linspace(0, 2, 20)
#ys = 0.5 * xs + 0.3 + np.random.normal(0, 0.1, 20)

# Fit model to data
result1 = model1.fit({'x': sortedLate, 'y': cumul}, {'k': 1, 'sigma1': 1})
result2 = model2.fit({'x': sortedLate, 'y': cumul}, {'k1': 0.1, 'k2' : 1., 'theta' : 0.5, 'sigma2': 1})

print(result1)
print(result2)


#plt.plot(sortedLate,cumul/len(cumul))
#plt.show()