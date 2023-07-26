# libraries
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions import lsing_LRC
import matplotlib.gridspec as gspec

'''# Defining the axis
alpha_val = np.arange(0,4,0.1)
beta_val = [0,0.05,0.1,0.4,1.0, 2.0,3.0, 6.0]  # Beta Values for which we generate the plots.

#Constants
num_samples = 100
sample_size = 100

# code
stdev_matrix = np.empty((len(beta_val),len(alpha_val)))
for i,beta in enumerate(beta_val):
  for idx,alpha in enumerate(alpha_val):
      stdev_matrix[i,idx] = np.std(np.sum(lsing_LRC(num_samples,sample_size,beta,alpha), axis= 0))

# plotting
num_rows = int((len(beta_val))/2)
width_figure = 6
height_figure = 10*len(beta_val)
fig = plt.figure(figsize=(width_figure,height_figure))
gs = gspec.GridSpec(num_rows, 2, figure=fig)

for i in range(num_rows):
   ax = fig.add_subplot(gs[i,0])
   plt.plot(alpha_val,stdev_matrix[i])
   plt.title(r'$\beta$ = ' + str(beta_val[i]))

for i in range(num_rows,len(beta_val)):
   ax = fig.add_subplot(gs[i-num_rows,1])
   plt.plot(alpha_val,stdev_matrix[i])
   plt.grid()
   plt.ylabel('Standard deviation of the sum')
   plt.xlabel('alpha_values')
   plt.title(r'$\beta$ = ' + str(beta_val[i]))
'''

fig = plt.figure(figsize=(10,10))
gs= gspec.GridSpec(1,2,figure=fig)

ax_low_alpha = fig.add_subplot(gs[0,0])
sns.histplot(np.sum(lsing_LRC(100,1000,1,0.2),axis=0), kde='True', color='red', stat='probability')
ax_low_alpha.set_title('alpha = 0.2')

ax_high_alpha = fig.add_subplot(gs[0,1])
sns.histplot(np.sum(lsing_LRC(100,1000,1,4),axis=0), kde=True, stat='probability')
ax_high_alpha.set_title('alpha = 4')

plt.tight_layout()
plt.show()