# libraries
import numpy as np
import matplotlib.pyplot as plt
from functions import exp_histmethod ,stdev_histmethod

# Constants
num_samples = 1000
mean  = 0
sample_size = 10000

# Main progamme
sum_samples = np.zeros(sample_size)
stdev = np.abs(np.random.standard_cauchy(num_samples))   #disorder in the standard deviation
for i in range(num_samples):
    sample_i = np.random.normal(mean, stdev[i], sample_size)
    sum_samples += sample_i

print(exp_lnn(sum_samples,100))
print(stdev_lnn(sum_samples,100)/(num_samples))
print(np.mean(stdev))

plt.hist(sum_samples, bins = 100 , density=True)
plt.show()

