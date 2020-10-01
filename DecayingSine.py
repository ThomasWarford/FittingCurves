import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

data = np.genfromtxt('DecayData.csv', delimiter=',')

time = data[1:, 0]
offset = data[1:, 1]
sigma = data[1:, 2]

def decay(time, offset0, A, gamma, omega, phi):
    out = offset0 + A * np.exp(-(gamma * time) / 2) * np.sin(omega * time - phi)
    return out

def chi2(offset0_A_gamma_omega_phi):
    offset0, A, gamma, omega, phi = offset0_A_gamma_omega_phi
    expectedoffset = decay(time, offset0, A, gamma, omega, phi)
    out = sum(pow((expectedoffset-offset) / sigma, 2))
    return out

result = minimize(chi2, (0.01, 0.06, 0.3, 105, 0))
offset0, A, gamma, omega, phi = result.x

    
    
plt.errorbar(time, offset, yerr=sigma, fmt='rx')
x = np.linspace(0, 0.3, 1000)

plt.plot(x, decay(x, offset0, A, gamma, omega, phi))
plt.xlabel('Time / s')
plt.ylabel('x / m')
plt.savefig('DecayingSineFit.png')
plt.show()