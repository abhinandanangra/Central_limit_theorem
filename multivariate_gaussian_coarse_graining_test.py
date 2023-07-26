import numpy as np
import matplotlib.pyplot as plt
from functions import coarse_graining,calculate_rmse
from scipy import stats

# constants
sample_size = 500
num_samples = 1024
num_iterations = 10
bins = 100
mean = 0
std = 1

# generating the mean and covariance arrays
mean_array = np.zeros(num_samples)
matrix_gauss = np.random.normal(mean,std,size=(num_samples,sample_size))


def coarse_graining(initial_matrix,num_iterations,bins,mean,std):
  error = []
  matrix = initial_matrix
  for j in range(num_iterations):
      if matrix.shape[0] % 2 != 0:
          matrix = matrix[:-1, :]  # Discard the last row if the number of rows is odd
      matrix_next = (matrix[::2] + matrix[1::2]) / np.sqrt(2) # scaling with root(2) based on CLT
      sum_samples = np.sum(matrix_next,axis=0)
      counts,edges = np.histogram(sum_samples, bins,density=True)
      bin_centers = (edges[:-1] + edges[1:]) / 2
      error.append(calculate_rmse(counts,stats.norm.pdf(bin_centers,len(matrix_next)*mean,np.sqrt(len(matrix_next))*std))) 
      matrix = matrix_next
    # Plotting the results
  plt.plot(np.arange(len(error)), error)
  plt.xlabel('Number of iterations')
  plt.ylabel('RMSE')
  plt.title('Coarse Graining') 
  
  
coarse_graining(matrix_gauss,10, bins,mean,std)
plt.show()