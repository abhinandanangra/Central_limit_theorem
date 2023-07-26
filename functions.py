# This file contains all the functions and you can import the requried functions from here using the import module.
# Do not forget to initialize the necessary libraries.
  
# Libraries
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from scipy import stats
from scipy.integrate import quad

# fxn1, return first positive index of an array or -1 if none found.
def indx_first_positive(array):
    for i in range(len(array)):
        if array[i] > 0:
            return i
    return -1  # Return -1 if no positive number is found

# fxn2 , returns a matrix containing distribution vectors as rows for lsing previous neighbour case
def lsing_PNI(sample_size, num_samples, beta, J):

  X = np.empty([num_samples,sample_size])
  x0 = np.random.choice([-1,1], sample_size ,p =[0.5, 0.5])
  X[0] = x0

  for j in range(1,num_samples):

    x = np.zeros(sample_size)
    pa = np.exp(beta*J*X[j-1])/(np.exp(beta*J*X[j-1])+np.exp(-beta*J*X[j-1]))
    pb = np.exp(-beta*J*X[j-1])/(np.exp(beta*J*X[j-1])+np.exp(-beta*J*X[j-1]))
  
    for i in range(sample_size):
       x[i] = np.random.choice([1,-1], p=[pa[i],pb[i]] )

    X[j] = x

  return X  

#fxn3, returns a matrix containing distribution vectors as rows for lsing long range coorelation case
def lsing_LRC(num_samples,sample_size,beta,alpha):
  X = np.empty([num_samples,sample_size])
  x0 = np.random.choice([-1,1], sample_size ,p =[0.5, 0.5])
  X[0] = x0
  x = np.zeros(sample_size)
  for j in range(1,num_samples):
      q = np.zeros(sample_size)

      for i in range(j):
        J = 1/(abs(i-j)**alpha)
        q += J*beta*X[i]

      for k in range(sample_size):
        p1 = math.exp(q[k]) / (math.exp(q[k]) + math.exp(-q[k]))
        p2 = math.exp(-q[k]) / (math.exp(q[k]) + math.exp(-q[k]))
        x[k] = np.random.choice([1,-1], p =[p1,p2])

      X[j] = x

  return X

#fxn4, Expectation and variances considering histograms as a pmf and bin-centeres and random variables
def exp_histmethod(data, num_bins):
    counts, edges = np.histogram(data, bins=num_bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    expectation = np.sum((bin_centers * counts) / np.sum(counts))
    return expectation


def stdev_histmethod(data, num_bins):
    counts, edges = np.histogram(data, bins=num_bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    expectation = exp_histmethod(data, num_bins=num_bins)
    variance = np.sum((((bin_centers - expectation) ** 2) * counts) / np.sum(counts))
    standard_deviation = np.sqrt(variance)
    return standard_deviation

#fxn5, Random Sampler for a given distribution which is normalized;
# based on Acceptance - Rejection Sampling
def ar_random_sampler(a,b, normalized_form, sample_size, num_samples): # a and b are the bounds on the values of distribution
  sample_matrix = np.empty((num_samples,sample_size))
  for i in range(num_samples):
    sample_i = []
    while len(sample_i)< sample_size:
        x = np.random.uniform(a,b)
        y = np.random.random()
        if y <= normalized_form(x): #Define the normalized_form as an independent funciton first
          sample_i.append(x)
    sample_matrix[i] = sample_i
  return sample_matrix # returns  a matrix with rows as distribution vectors


# fx6, Root mean square calculator
def calculate_rmse(array1, array2):
   return np.sqrt(np.mean((array1-array2)**2))

#fxn7, clt verifier
def clt_verifier(sum,num_bins,mean_i,std_i):
  mean = []
  std = []
  rmse_histogram = []
  n_val = np.arange(0,1000,10)
  for num_samples in n_val:
    sum_samples = sum(num_samples)
    counts,edges = np.histogram(sum_samples, bins=num_bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    rmse_histogram.append(calculate_rmse(counts,stats.norm.pdf(bin_centers,num_samples*mean_i,np.sqrt(num_samples)*std_i))) 
    mean.append(np.mean(sum_samples))
    std.append(np.std(sum_samples))

  error_mean = mean - (n_val*mean_i)
  error_std = std - np.sqrt(n_val)*std_i

  fig = plt.figure(figsize=(12,12))
  gs = gspec.GridSpec(2, 2, figure=fig)
  fig.subplots_adjust(top=0.85) 

  ax_error = fig.add_subplot(gs[0,:])
  ax_error.plot(n_val,error_mean, label = "Error in mean")
  ax_error.plot(n_val,error_std, label = 'Error in std. dev.')
  plt.legend()
  ax_error.set_xlabel('Number of samples')
  ax_error.set_ylabel('Error')
  plt.grid()
  
  ax_hist_error = fig.add_subplot(gs[1,1])
  ax_hist_error.plot(n_val, rmse_histogram, label='RMSE for Sum distribution')
  ax_hist_error.set_xlabel('Number of samples')
  ax_hist_error.set_ylabel('Error for sum of rvs')
  plt.grid()
  
  ax_hist = fig.add_subplot(gs[1,0])
  sns.histplot(sum_samples,kde='True',color='red')
  plt.title('Histogram plot for 1000 samples')
  
  fig.tight_layout(h_pad=7,w_pad=2)
  return fig,np.mean(rmse_histogram[1:])

#fxn8, coarse graining, for two variables at a time , with appropriate scaling

def coarse_graining(initial_matrix,num_iterations):
  cov_offdiag_sum_val = []
  matrix = initial_matrix
  for j in range(num_iterations):
      if matrix.shape[0] % 2 != 0:
          matrix = matrix[:-1, :]  # Discard the last row if the number of rows is odd
      matrix_next = (matrix[::2] + matrix[1::2]) / math.sqrt(2) # scaling with root(2) based on CLT
      cov_next_matrix = np.cov(matrix_next, rowvar=True)
      matrix = matrix_next
      if cov_next_matrix.shape[0] <= 3: # why 3, determine yourself
          break
      off_diagonal_matrix = cov_next_matrix[~np.eye(cov_next_matrix.shape[0], dtype=bool)]
      n_rows = cov_next_matrix.shape[0]
      cov_offdiag_sum_val.append(abs(off_diagonal_matrix.sum())/((n_rows**2)-n_rows)) # this here is called a boolean mask
      
  # Plotting the results
  plt.plot(np.arange(len(cov_offdiag_sum_val)), cov_offdiag_sum_val)
  plt.xlabel('Number of iterations')
  plt.ylabel('Sum of off-diagonal elements of Covariance matrix')
  plt.title('Coarse Graining')

#fxn9, fourier transform
def fourier_transform(func, x): #x is array values and func(t) is characteristic equation
    ft_integrand_real = lambda t: np.real(func(t)*np.exp(-1j*x*t))
    ft_integrand_imag = lambda t: np.imag(func(t)*np.exp(-1j*x*t))
    ft_real = quad(ft_integrand_real, -np.inf, np.inf)[0]
    ft_imag = quad(ft_integrand_imag, -np.inf, np.inf)[0]
    return (ft_real + 1j*ft_imag)/(2*np.pi) # returns a complex array dependent on x.

      