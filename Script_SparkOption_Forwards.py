# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 01:22:44 2024

@author: Marouane
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation



def kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    d1 = (np.log(F1 / (G2 + K)) + 0.5 * sigma_tilde**2 * T) / (sigma_tilde * np.sqrt(T))
    d2 = d1 - sigma_tilde * np.sqrt(T)
    
    price = F1 * norm.cdf(d1) - (G2 + K) * norm.cdf(d2)
    return np.exp(-r*T)*price


def modif_kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    X_t = np.log(F1)
    Y_t = np.log(G2 + K)

    I_tilde = np.sqrt(sigma_tilde**2) + 0.5 * ((sigma_G_tilde  - rho * sigma_F)**2) * (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)
    
    
    d1 = (np.log(F1 / (G2 + K)) + 0.5 * I_tilde**2 * T) / (I_tilde * np.sqrt(T))
    d2 = d1 - I_tilde * np.sqrt(T)
    
    price = F1 * norm.cdf(d1) - (G2 + K) * norm.cdf(d2)
    return np.exp(-r*T)*price


def mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r, num_simulations=50000000):
    dt = T
    Z1 = np.random.standard_normal(num_simulations)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(num_simulations)
    
    F1_T = F1 * np.exp((- 0.5 * sigma_F**2) * T + sigma_F * np.sqrt(dt) * Z1)
    G2_T = G2 * np.exp((- 0.5 * sigma_G**2) * T + sigma_G * np.sqrt(dt) * Z2)
    
    payoffs = np.maximum(F1_T - G2_T - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

# Parameters
F1 = 100  # Forward price of electricity
G = 100   # Forward price of gas
h=1
G2=G*h
T = 0.5    # Time to maturity
sigma_F = 0.3  # Volatility of electricity price
sigma_G = 0.2  # Volatility of gas price
r = 0.02  # Risk-free rate

# Generate data for surface plot
K_range = np.arange(0, 21, 1)
rho_range = [0.80, 0.85, 0.90, 0.95, 0.999]

kirk_prices = np.zeros((len(K_range), len(rho_range)))
mc_prices = np.zeros((len(K_range), len(rho_range)))
modif_kirk_prices = np.zeros((len(K_range), len(rho_range)))
relative_errors = np.zeros((len(K_range), len(rho_range)))
modif_relative_errors = np.zeros((len(K_range), len(rho_range)))


# for i, K in enumerate(K_range):
#     for j, rho in enumerate(rho_range):
#         print(str(i) + '::' + str(j))
#         kirk_prices[i, j] = kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
#         mc_prices[i, j] = mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
#         relative_errors[i, j] = (kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j]

start_time = time.time() 
for i, K in enumerate(K_range):
    for j, rho in enumerate(rho_range):
        print('K:' + str(K) + '::' + 'rho: ' + str(rho))
        kirk_prices[i, j] = kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
        modif_kirk_prices[i, j] = modif_kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
        mc_prices[i, j] = mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
        relative_errors[i, j] = abs(kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j] * 100
        modif_relative_errors[i, j] = abs(modif_kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j] * 100
      
end_time = time.time()      
execution_time = end_time - start_time 
print(f"Execution Time: {execution_time} seconds")    
        
      
############################## PLOT  ####################################################################
#2D PLot      
for j, rho in enumerate(rho_range):
    plt.plot(K_range,relative_errors[:, j], label='Original Kirk formula', color='blue')
    plt.plot(K_range,modif_relative_errors[:, j], label='MC', color='red')
    plt.plot(K_range,[0] * len(K_range), label='y=0', color='green', linestyle='--')
    plt.xlabel('K')
    plt.ylabel(f'Error of Kirk Formula ({rho})')
    plt.legend()
    plt.show()
    
        
    
 

# Create surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

K, RHO = np.meshgrid(K_range, rho_range)
surf = ax.plot_surface(K, RHO, relative_errors.T, cmap='coolwarm',alpha=0.7, edgecolor='k')

modif_surf = ax.plot_surface(K, RHO, modif_relative_errors.T, cmap='viridis', alpha=0.5)


# Add colorbars for both surfaces
cbar1 = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar1.set_label('Error (%) (Original Kirk)')
cbar2 = fig.colorbar(modif_surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar2.set_label('Error (%) (Modified Kirk)')



ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Correlation (ρ)')
ax.set_zlabel('Relative Error')
ax.set_title("Relative Errors: Abs(Kirk - MC) / MC")

# Add a color bar
#fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#ax.view_init(elev=30, azim=60)  # elev = vertical angle, azim = horizontal angle


def update(frame):
    ax.view_init(elev=30, azim=frame)  # Rotate the plot by changing azim
    return ax,

ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

ani.save('C:/Users/Marouane/OneDrive/Echange/Uniper/numerical result/rotating_3d_surface.mp4', writer='ffmpeg', fps=10)  # For .mp4



plt.tight_layout()
plt.show()









    
#######################################
k,rho = 20,0.999
kirk_approx = kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
mc_price = mc_simulation(F1, G2, k, T, sigma_F, sigma_G, rho, r)

relative_error_ = abs(kirk_approx - mc_price) / mc_price * 100
relative_errors[i,j]


###################################
   


kirk_prices[i, j]
modif_kirk_prices[i, j] 
mc_prices[i, j]

K_range[i]











# Create surface plot
fig = plt.figure(figsize=(18, 5))

# Kirk's Approximation
ax1 = fig.add_subplot(131, projection='3d')
K, RHO = np.meshgrid(K_range, rho_range)
surf1 = ax1.plot_surface(K, RHO, kirk_prices.T, cmap='viridis')
ax1.set_xlabel('Strike Price (K)')
ax1.set_ylabel('Correlation (ρ)')
ax1.set_zlabel('Option Price')
ax1.set_title("Kirk's Approximation")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Monte Carlo Simulation
ax2 = fig.add_subplot(132, projection='3d')
surG2 = ax2.plot_surface(K, RHO, mc_prices.T, cmap='viridis')
ax2.set_xlabel('Strike Price (K)')
ax2.set_ylabel('Correlation (ρ)')
ax2.set_zlabel('Option Price')
ax2.set_title("Monte Carlo Simulation")
fig.colorbar(surG2, ax=ax2, shrink=0.5, aspect=5)

# Relative Errors
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(K, RHO, relative_errors.T, cmap='coolwarm')
ax3.set_xlabel('Strike Price (K)')
ax3.set_ylabel('Correlation (ρ)')
ax3.set_zlabel('Relative Error')
ax3.set_title("Relative Errors")
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# Print some sample values for comparison
print("Sample prices and relative error for K=10, rho=0.90:")
kirk_price = kirk_approximation(F1, G2, 10, T, sigma_F, sigma_G, 0.90, r)
mc_price = mc_simulation(F1, G2, 10, T, sigma_F, sigma_G, 0.90, r)
rel_error = (kirk_price - mc_price) / mc_price
print(f"Kirk's Approximation: {kirk_price:.4f}")
print(f"Monte Carlo Simulation: {mc_price:.4f}")
print(f"Relative Error: {rel_error:.4f}")






































# Create surface plot
fig = plt.figure(figsize=(12, 5))

# Kirk's Approximation
ax1 = fig.add_subplot(121, projection='3d')
K, RHO = np.meshgrid(K_range, rho_range)
surf1 = ax1.plot_surface(K, RHO, kirk_prices.T, cmap='viridis')
ax1.set_xlabel('Strike Price (K)')
ax1.set_ylabel('Correlation (ρ)')
ax1.set_zlabel('Option Price')
ax1.set_title("Kirk's Approximation")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Monte Carlo Simulation
ax2 = fig.add_subplot(122, projection='3d')
surG2 = ax2.plot_surface(K, RHO, mc_prices.T, cmap='viridis')
ax2.set_xlabel('Strike Price (K)')
ax2.set_ylabel('Correlation (ρ)')
ax2.set_zlabel('Option Price')
ax2.set_title("Monte Carlo Simulation")
fig.colorbar(surG2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# Print some sample values for comparison
print("Sample prices for K=10, rho=0.90:")
print(f"Kirk's Approximation: {kirk_approximation(F1, G2, 10, T, sigma_F, sigma_G, 0.90, r):.4f}")
print(f"Monte Carlo Simulation: {mc_simulation(F1, G2, 10, T, sigma_F, sigma_G, 0.90, r):.4f}")