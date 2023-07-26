#Libraries
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from functions import clt_verifier
import matplotlib.gridspec as gspec
from scipy import stats

# constants
sample_size = 10000
num_bins = 1000
Avg_error = np.empty(5)

# 1. Testing with uniform distribution

#constants
mean = 0.5
std = 1/np.sqrt(12)

## Defining the sample generator matrix
def sum_uniform(num_samples):
    return np.sum(np.random.uniform(size=(num_samples,sample_size)),axis=0)

# plots
fig_uniform,Avg_error[0] = clt_verifier(sum_uniform,num_bins,mean,std)
fig_uniform.suptitle("Summation on Uniform rvs",va='bottom')


#3. normal distribution
# Constants
mean = 0
std = 1

# distribution samples
def sum_gauss(num_samples):
    return np.sum(np.random.normal(mean,std,size=(num_samples,sample_size)),axis=0)

# plots
fig_gauss,Avg_error[1]=clt_verifier(sum_gauss,num_bins,mean,std)
fig_gauss.suptitle("Summation on Gaussian rvs",va='bottom')


# 3. Testing with normal distribution

# Constants
mean = 2
std = 3

# distribution samples
def sum_gauss(num_samples):
    return np.sum(np.random.normal(mean,std,size=(num_samples,sample_size)),axis=0)

# plots
fig_gauss,Avg_error[2]=clt_verifier(sum_gauss,num_bins,mean,std)
fig_gauss.suptitle("Summation on Gaussian rvs",va='bottom')

# 4. Testing with exponential distribution
#constants
scale = 2

# defining the sum
def sum_exponential(num_samples):
    return np.sum(np.random.exponential(scale,size=(num_samples,sample_size)),axis=0)
    
# plots
fig_gauss,Avg_error[3]=clt_verifier(sum_exponential,num_bins,scale,scale)
fig_gauss.suptitle("Summation on Exponential rvs",va='bottom')


# 5. Testing with pareto distribution, a/x^(1+a), x>=1 and a>=0

# Constants
a = 3
mean = a/(a-1)
std = np.sqrt(a/((a - 2)*((a-1)**2)))

# ditribution samples
def sum_pareto(num_samples):
    return np.sum(stats.pareto.rvs(a,size=(num_samples,sample_size)),axis=0)

# plots
fig_pareto,Avg_error[4]=clt_verifier(sum_pareto,num_bins,mean,std)
fig_pareto.suptitle("Summation on pareto rvs, a=3",va='bottom')

# Comparison plot

print(Avg_error)

plt.scatter(np.arange(5),Avg_error,marker='o',color='b')
plt.xlabel('Distributions')
plt.ylabel('Average Errors')
plt.grid()
plt.show()