import numpy as np
import matplotlib.pyplot as plt

# Define the Ising model parameters
L = 32  # Size of the lattice
J = 1.0  # Interaction strength
T = 2.0  # Temperature

# Initialize a structured spin configuration (all up or all down)
initial_spins = np.ones(L)

# Perform the Kadanoff block-spin RG transformation with temperature (T) and interaction strength (J)
def kadanoff_rg(spins, T, J):
    n = len(spins) // 2
    block_spins = spins[::2] + spins[1::2]
    p = np.tanh(2 * J / T)
    return np.where(block_spins > 0, p, -p)

# Perform RG iterations
rg_iterations = 10
magnetization = []

for i in range(rg_iterations):
    magnetization.append(np.mean(initial_spins))
    initial_spins = kadanoff_rg(initial_spins, T, J)

# Plot the magnetization vs RG iterations
plt.plot(range(rg_iterations), magnetization, marker='o')
plt.xlabel("RG Iterations")
plt.ylabel("Magnetization")
plt.title("Magnetization vs RG Iterations (Ising Model)")
plt.show()
