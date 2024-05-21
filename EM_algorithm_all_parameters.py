import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

data_path = 'MixNormalData.txt'  
data = np.loadtxt(data_path, skiprows=1)

plt.hist(data, bins='auto')  
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

n = len(data)

z_values = np.full(n, 0.5)

# initial p = 0.5

p = 0.5

random_1 = np.random.choice(data)
random_2 = np.random.choice(data)
sample_sd = np.std(data, ddof=1)

# define parameters for f(x) and g(x)
mu_1, sigma_1 = random_1, sample_sd # f(x)
mu_2, sigma_2 = random_2, sample_sd # g(x)

# give names to distributions

normal_pdf = lambda x, mu, sigma: norm.pdf(x, mu, sigma)

# define mixture distribution

mixture_pdf_x = lambda x, p, mu_1, mu_2, sigma_1, sigma_2: \
	(p * normal_pdf(x, mu_1, sigma_1)) + \
	( (1 - p) * normal_pdf(x, mu_2, sigma_2))

# expectation step formula

responsibility = lambda x, p, mu_1, mu_2, sigma_1, sigma_2:\
	(p * normal_pdf(x, mu_1, sigma_1)) / \
	( (p * normal_pdf(x, mu_1, sigma_1)) + \
	(1 - p) * normal_pdf(x, mu_2, sigma_2) )

min_x = np.min(data)
max_x = np.max(data)
x_values = np.linspace(min_x, max_x, 1000)

pdf_values_list = [] 

for j in range(30):
	for i in range(n):
		z_values[i] = responsibility( data[i], p, mu_1, mu_2, sigma_1, sigma_2)

	mu_1 = np.sum(z_values * data) / np.sum(z_values)
	mu_2 = np.sum((1 - z_values) * data) / np.sum(1 - z_values)
	sigma_1 = np.sqrt(  np.sum(z_values * ( data - mu_1 )**2 ) / np.sum(z_values) )
	sigma_2 = np.sqrt(  np.sum( (1 - z_values) * ( data - mu_2 )**2 ) / np.sum(1 - z_values) )
	p = z_values.sum()/n

	print(f"{mu_1}, {mu_2}, {sigma_1}, {sigma_2}, {p}")

	if (j + 1) % 10 == 0:
		pdf_values = mixture_pdf_x(x_values, p, mu_1, mu_2, sigma_1, sigma_2)
		pdf_values_list.append((x_values, pdf_values, f"Step {j + 1}")) 

plt.figure(figsize=(10, 6))
plt.hist(data, bins='auto', density=True, alpha=0.5, label='Data Histogram')

for x_vals, pdf_vals, label in pdf_values_list:
	plt.plot(x_vals, pdf_vals, label=label, linewidth=2)

plt.title('Data and Mixture PDF over Iterations')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.show()


