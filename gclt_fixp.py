import numpy as np
import matplotlib.pyplot as plt

def custom_distribution(num_samples, sample_size, p, lam):
    # Generate samples from the custom distribution
    samples = []
    for _ in range(num_samples):
        x = np.random.rand(sample_size)
        gaussian_samples = np.random.normal(0, 1, sample_size)
        exponential_samples = np.random.exponential(1/lam, sample_size)
        sample = np.where(x <= p, gaussian_samples, exponential_samples)
        samples.append(sample)
    
    return samples

def stable_pdf(x, alpha, beta, c, mu):
    # Define the characteristic function of the stable distribution
    t = x - mu
    phi = np.exp(1j * t * mu - np.abs(c * t) ** alpha * (1 - 1j * beta * np.sign(t)))
    return np.real(phi) / (2 * np.pi)

def rmse(actual, expected):
    # Calculate the Root Mean Squared Error (RMSE) between actual and expected distributions
    return np.sqrt(np.mean((actual - expected) ** 2))

# Parameters
num_samples_range = [10, 50, 100, 500, 1000, 5000]
sample_size = 100
p = 0.5  # Fixed value of p
lam_range = np.linspace(0.1, 1.0, 10)

# Calculate RMSE for different values of lam
rmse_values = np.zeros((len(lam_range), len(num_samples_range)))
for i, lam in enumerate(lam_range):
    for j, num_samples in enumerate(num_samples_range):
        # Generate samples from the custom distribution
        samples = custom_distribution(num_samples, sample_size, p, lam)
        
        # Calculate the sum of each sample
        sums = np.sum(samples, axis=1)
        
        # Calculate the RMSE between the empirical distribution and the stable distribution
        empirical_distribution, _ = np.histogram(sums, bins=50, density=True)
        expected_distribution = stable_pdf(np.linspace(-10, 10, 50), alpha=2-p, beta=0, c=1, mu=0)
        rmse_values[i, j] = rmse(empirical_distribution, expected_distribution)

# Plot the RMSE values for different values of lam
plt.figure(figsize=(8, 6))
for i, lam in enumerate(lam_range):
    plt.plot(num_samples_range, rmse_values[i], label=f"$\lambda$={lam}")
plt.xlabel("Number of Samples")
plt.ylabel("RMSE")
plt.title("RMSE vs. Number of Samples for Different Values of $\lambda$ (Fixed p=0.5)")
plt.legend()
plt.show()
