# Importing all the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gspec
from functions import lsing_PNI


'''beta_val = np.arange(0,4,0.1)
stdev = []

for beta in beta_val:
    stdev.append(np.std(np.sum(lsing_PNI(100,1000,beta,1), axis=0)))'''

fig = plt.figure(figsize=(10,10))
gs= gspec.GridSpec(1,2,figure=fig)

ax_low_beta = fig.add_subplot(gs[0,0])
sns.histplot(np.sum(lsing_PNI(100,1000,0.1,1),axis=0), kde='True', color='red', stat='probability')
ax_low_beta.set_title('Beta = 0.1')

ax_high_beta = fig.add_subplot(gs[0,1])
sns.histplot(np.sum(lsing_PNI(100,1000,6,1),axis=0), kde=True, stat='probability')
ax_high_beta.set_title('Beta = 6')

'''plt.plot(beta_val,stdev)
plt.xlabel('Beta_values')
plt.ylabel("Standard deviation of the sum")'''
plt.show()

# How does a binomial reach gaussian study that.
# Match with gaussian
# Optmize the program
# The anomaly is that, when we record probabilities related to x1 only, then we get an increasing plot while for purely previous neighbour case we get decreasing plot.


