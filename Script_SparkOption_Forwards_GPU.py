# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 01:22:44 2024

@author: Marouane
"""

import cupy as cp  # Replace numpy with cupy
import matplotlib.pyplot as plt
from scipy.stats import norm  # Norm from scipy can stay the same
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
from matplotlib.animation import FuncAnimation



def kirk_approximation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = cp.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    d1 = (cp.log(F1 / (G2 + K)) + 0.5 * sigma_tilde**2 * T) / (sigma_tilde * cp.sqrt(T))
    d2 = d1 - sigma_tilde * cp.sqrt(T)
    
    price = F1 * norm.cdf(d1.get()) - (G2 + K) * norm.cdf(d2.get())
    return cp.exp(-r*T)*price


def modif_kirk_approximation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = cp.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    X_t = cp.log(F1)
    Y_t = cp.log(G2 + K)

    I_tilde = cp.sqrt(sigma_tilde**2) + 0.5 * ((sigma_G_tilde - rho * sigma_F)**2) * (1 / ((cp.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)
    
    d1 = (cp.log(F1 / (G2 + K)) + 0.5 * I_tilde**2 * T) / (I_tilde * cp.sqrt(T))
    d2 = d1 - I_tilde * cp.sqrt(T)
    
    price = F1 * norm.cdf(d1.get()) - (G2 + K) * norm.cdf(d2.get())
    return cp.exp(-r*T)*price


def delta_G_modif_kirk_approximation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = cp.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    X_t = cp.log(F1)
    Y_t = cp.log(G2 + K)

    I_tilde = cp.sqrt(sigma_tilde**2) + 0.5 * ((sigma_G_tilde - rho * sigma_F)**2) * (1 / ((cp.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)
    
    d1 = (cp.log(F1 / (G2 + K)) + 0.5 * I_tilde**2 * T) / (I_tilde * cp.sqrt(T))
    d2 = d1 - I_tilde * cp.sqrt(T)
    
    price = - norm.cdf(d2.get())
    return cp.exp(-r*T)*price



def mc_simulation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r, num_simulations=50000000):
    dt = T
    Z1 = cp.random.standard_normal(num_simulations)
    Z2 = rho * Z1 + cp.sqrt(1 - rho**2) * cp.random.standard_normal(num_simulations)
    
    F1_T = F1 * cp.exp((- 0.5 * sigma_F**2) * T + sigma_F * cp.sqrt(dt) * Z1)
    G2_T = G2 * cp.exp((- 0.5 * sigma_G**2) * T + sigma_G * cp.sqrt(dt) * Z2)
    
    payoffs = cp.maximum(F1_T - G2_T - K, 0)
    price = cp.exp(-r * T) * cp.mean(payoffs)
    return price

# Parameters
F1 = 100  # Forward price of electricity
G = 100   # Forward price of gas
h = 1
G2 = G * h
T = 0.5    # Time to maturity
sigma_F = 0.3  # Volatility of electricity price
sigma_G = 0.2  # Volatility of gas price
r = 0.02  # Risk-free rate

# Generate data for surface plot
K_range = cp.arange(0, 21, 1)
rho_range = [0.80, 0.85, 0.90, 0.95, 0.999]

kirk_prices = cp.zeros((len(K_range), len(rho_range)))
mc_prices = cp.zeros((len(K_range), len(rho_range)))
modif_kirk_prices = cp.zeros((len(K_range), len(rho_range)))
relative_errors = cp.zeros((len(K_range), len(rho_range)))
modif_relative_errors = cp.zeros((len(K_range), len(rho_range)))

start_time = time.time()
for i, K in enumerate(K_range):
    for j, rho in enumerate(rho_range):
        print('K:' + str(K) + '::' + 'rho: ' + str(rho))
        kirk_prices[i, j] = kirk_approximation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r)
        modif_kirk_prices[i, j] = modif_kirk_approximation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r)
        mc_prices[i, j] = mc_simulation_cp(F1, G2, K, T, sigma_F, sigma_G, rho, r)
        relative_errors[i, j] = abs(kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j] * 100
        modif_relative_errors[i, j] = abs(modif_kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j] * 100

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")


mc_prices_np = cp.asnumpy(mc_prices)
kirk_prices_np = cp.asnumpy(kirk_prices)
modif_kirk_prices_np = cp.asnumpy(modif_kirk_prices)

############################## PLOT ####################################################################
# Convert back to NumPy for plotting
K_range_np = cp.asnumpy(K_range)
relative_errors_np = cp.asnumpy(relative_errors)
modif_relative_errors_np = cp.asnumpy(modif_relative_errors)

# 2D Plot
for j, rho in enumerate(rho_range):
    plt.plot(K_range_np, relative_errors_np[:, j], label='Original Kirk formula', color='blue')
    plt.plot(K_range_np, modif_relative_errors_np[:, j], label='MC', color='red')
    plt.plot(K_range_np, [0] * len(K_range_np), label='y=0', color='green', linestyle='--')
    plt.xlabel('K')
    plt.ylabel(f'Error of Kirk Formula ({rho})')
    plt.legend()
    plt.show()
    
        
# # 2D Plot
# for j, rho in enumerate(rho_range):
#     plt.plot(mc_prices_np[:, j], mc_prices_np[:, j], label='x = y', color='g', linestyle='--')  # Tracer la ligne x = y en rouge et en pointillé
#     plt.scatter(mc_prices_np[:, j], kirk_prices_np[:, j], color='b', label='kirk approximation', s=10)  # Les points sont tracés en bleu
#     plt.scatter(mc_prices_np[:, j], modif_kirk_prices_np[:, j], color='r', label='modified kirk approximation', s=10)  # Les points sont tracés en bleu
#     plt.legend()
#     plt.show()





# Create surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

K, RHO = np.meshgrid(K_range_np, rho_range)
surf = ax.plot_surface(K, RHO, relative_errors_np.T, cmap='coolwarm',alpha=0.5, edgecolor='b')

modif_surf = ax.plot_surface(K, RHO, modif_relative_errors_np.T, cmap='viridis', edgecolor='k')

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

## to save the animated surfaces
# def update(frame):
#     ax.view_init(elev=30, azim=frame)  # Rotate the plot by changing azim
#     return ax,

# ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

# ani.save('C:/Users/Marouane/OneDrive/Echange/Uniper/numerical result/rotating_3d_surface.mp4', writer='ffmpeg', fps=10)  # For .mp4

plt.tight_layout()
plt.show()
