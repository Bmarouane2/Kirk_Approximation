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
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from scipy import stats



n_f = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)

def kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    d1 = (np.log(F1 / (G2 + K)) + 0.5 * sigma_tilde**2 * T) / (sigma_tilde * np.sqrt(T))
    d2 = d1 - sigma_tilde * np.sqrt(T)
    
    price = F1 * norm.cdf(d1) - (G2 + K) * norm.cdf(d2)
    return np.exp(-r*T)*price  


def delta_F_kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    d1 = (np.log(F1 / (G2 + K)) + 0.5 * sigma_tilde**2 * T) / (sigma_tilde * np.sqrt(T))
    
    price = norm.cdf(d1)
    return np.exp(-r*T)*price

def delta_G_kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    d1 = (np.log(F1 / (G2 + K)) + 0.5 * sigma_tilde**2 * T) / (sigma_tilde * np.sqrt(T))
    d2 = d1 - sigma_tilde * np.sqrt(T)
    
    dg_dG= sigma_G*K/sigma_tilde*(sigma_G_tilde - rho*sigma_F)/(G2+K)**2
    
    return np.exp(-r*T)*(-norm.cdf(d2) + (G2 + K) *n_f(d2)*np.sqrt(T)*dg_dG)
    # return np.exp(-r*T)*(-norm.cdf(d2))



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



def delta_F_modif_kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    X_t = np.log(F1)
    Y_t = np.log(G2 + K)

    I_tilde = np.sqrt(sigma_tilde**2) + 0.5 * ((sigma_G_tilde  - rho * sigma_F)**2) * (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)
    dI_dF = 0.5 * ((sigma_G_tilde  - rho * sigma_F)**2) * (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * 1/F1
    
    d1 = (np.log(F1 / (G2 + K)) + 0.5 * I_tilde**2 * T) / (I_tilde * np.sqrt(T))
    
    price = norm.cdf(d1)
    return np.exp(-r*T)*(norm.cdf(d1) + F1*n_f(d1)*np.sqrt(T)*dI_dF)


def delta_G_modif_kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r):
    sigma_G_tilde = (G2 / (G2 + K)) * sigma_G
    sigma_tilde = np.sqrt(sigma_F**2 + sigma_G_tilde**2 - 2 * rho * sigma_F * sigma_G_tilde)
    
    X_t = np.log(F1)
    Y_t = np.log(G2 + K)

    I_tilde = np.sqrt(sigma_tilde**2) + 0.5 * ((sigma_G_tilde  - rho * sigma_F)**2) * (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)

    d1 = (np.log(F1 / (G2 + K)) + 0.5 * I_tilde**2 * T) / (I_tilde * np.sqrt(T))
    d2 = d1 - I_tilde * np.sqrt(T)
    
    dg_dG= sigma_G*K/sigma_tilde*(sigma_G_tilde - rho*sigma_F)/(G2+K)**2
    
    addon_dg_1 = K/(G2+K)**2 *sigma_G*(sigma_G_tilde -rho * sigma_F)* (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)
    addon_dg_2 = dg_dG*(-3)/(np.sqrt(sigma_tilde**2))**4*((sigma_G_tilde  - rho * sigma_F)**2) * sigma_G_tilde * (sigma_G * K) / (G2 + K) * (X_t - Y_t)
    addon_dg_4 = (K/(G2+K)**2 - 2*G2*K/(G2+K)**3)*((sigma_G_tilde  - rho * sigma_F)**2) * (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G**2 * (X_t - Y_t)
    addon_dg_5 = -1/(G2+K)*((sigma_G_tilde  - rho * sigma_F)**2) * (1 / ((np.sqrt(sigma_tilde**2))**3)) * sigma_G_tilde * (sigma_G * K) / (G2 + K)
    
    dI_dG = dg_dG + 0.5* (addon_dg_1 + addon_dg_2 +addon_dg_4+ addon_dg_5)
    
    
    return np.exp(-r*T)*(-norm.cdf(d2) + (G2 + K) *n_f(d2)*np.sqrt(T)*dI_dG)


def mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r, num_simulations=10000000, alpha =0.95):
    dt = T
    Z1 = np.random.standard_normal(num_simulations)
    Z =np.random.standard_normal(num_simulations)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z
    
    F1_T = F1 * np.exp((- 0.5 * sigma_F**2) * T + sigma_F * np.sqrt(dt) * Z1)
    G2_T = G2 * np.exp((- 0.5 * sigma_G**2) * T + sigma_G * np.sqrt(dt) * Z2)
    
    F1_T_inv = F1 * np.exp((- 0.5 * sigma_F**2) * T + sigma_F * np.sqrt(dt) * (-Z1))
    G2_T_inv = G2 * np.exp((- 0.5 * sigma_G**2) * T + sigma_G * np.sqrt(dt) * (-Z2))
    
    
    payoffs = np.maximum(F1_T - G2_T - K, 0)
    payoffs_inv = np.maximum(F1_T_inv - G2_T_inv - K, 0)
    
    all_payoffs = np.concatenate((payoffs, payoffs_inv))
    discounted_payoffs = np.exp(-r * T) * all_payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = (np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs)))

    confidence_interval = stats.t.interval(alpha, len(discounted_payoffs) - 1, loc=price, scale=std_error)
    return price, confidence_interval


def delta_F_mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r, h=0.01, num_simulations=20000000, alpha =0.95):
    dt = T
    Z1 = np.random.standard_normal(num_simulations)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(num_simulations)
    
    F1_T = F1 * np.exp((- 0.5 * sigma_F**2) * T + sigma_F * np.sqrt(dt) * Z1)
    F1_T_minus = (F1-h) * np.exp((- 0.5 * sigma_F**2) * T + sigma_F * np.sqrt(dt) * Z1)

    G2_T = G2 * np.exp((- 0.5 * sigma_G**2) * T + sigma_G * np.sqrt(dt) * Z2)
    
    
    payoff_plus = np.maximum(F1_T - G2_T - K, 0)
    payoff_minus = np.maximum(F1_T_minus - G2_T - K, 0)
    
    price_plus = np.exp(-r * T) * payoff_plus
    price_minus = np.exp(-r * T) * payoff_minus
    
    delta_mc = (price_plus - price_minus) / h
    
    # Calculate mean and standard error
    delta_mc_mean = np.mean(delta_mc)
    delta_mc_std_error = np.std(delta_mc, ddof=1) / np.sqrt(num_simulations)
    
    # Calculate 95% confidence interval
    confidence_interval = stats.t.interval(alpha, df=num_simulations-1, 
                                           loc=delta_mc_mean, 
                                           scale=delta_mc_std_error)
    
    return delta_mc_mean.tolist(), confidence_interval

def delta_G_mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r, h=0.001, num_simulations=20000000, alpha =0.95):
    dt = T
    Z1 = np.random.standard_normal(num_simulations)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(num_simulations)
    
    F1_T = F1 * np.exp((- 0.5 * sigma_F**2) * T + sigma_F * np.sqrt(dt) * Z1)

    G2_T       = G2     * np.exp((- 0.5 * sigma_G**2) * T + sigma_G * np.sqrt(dt) * Z2)
    G2_T_minus = (G2-h) * np.exp((- 0.5 * sigma_G**2) * T + sigma_G * np.sqrt(dt) * Z2)
    
    payoff_plus = np.maximum(F1_T - G2_T - K, 0)
    payoff_minus = np.maximum(F1_T - G2_T_minus - K, 0)
    
    price_plus = np.exp(-r * T) * payoff_plus
    price_minus = np.exp(-r * T) * payoff_minus
    
    delta_mc = (price_plus - price_minus) / h
    
    # Calculate mean and standard error
    delta_mc_mean = np.mean(delta_mc)
    delta_mc_std_error = np.std(delta_mc, ddof=1) / np.sqrt(num_simulations)
    
    # Calculate 95% confidence interval
    confidence_interval = stats.t.interval(alpha, df=num_simulations-1, 
                                           loc=delta_mc_mean, 
                                           scale=delta_mc_std_error)
    
    return delta_mc_mean, confidence_interval

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
# K_range = np.arange(0, 21, 1)
K_range = np.arange(0, 21, 1)
rho_range = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.999]


kirk_prices = np.zeros((len(K_range), len(rho_range)))
mc_prices = np.zeros((len(K_range), len(rho_range)))
modif_kirk_prices = np.zeros((len(K_range), len(rho_range)))
relative_errors = np.zeros((len(K_range), len(rho_range)))
modif_relative_errors = np.zeros((len(K_range), len(rho_range)))
#Deltas
deltaF_kirk_prices = np.zeros((len(K_range), len(rho_range)))
deltaF_modifkirk_prices = np.zeros((len(K_range), len(rho_range)))
deltaF_mc = np.zeros((len(K_range), len(rho_range)))
deltaG_kirk_prices = np.zeros((len(K_range), len(rho_range)))
deltaG_modifkirk_prices = np.zeros((len(K_range), len(rho_range)))
deltaG_mc = np.zeros((len(K_range), len(rho_range)))
deltaF_relative_errors = np.zeros((len(K_range), len(rho_range)))
deltaF_modif_relative_errors = np.zeros((len(K_range), len(rho_range)))
deltaG_relative_errors = np.zeros((len(K_range), len(rho_range)))
deltaG_modif_relative_errors = np.zeros((len(K_range), len(rho_range)))

#execTimes
end_time_ij=[]

#IC
mc_prices_IC_up = np.zeros((len(K_range), len(rho_range)))
mc_prices_IC_down = np.zeros((len(K_range), len(rho_range)))
mc_delta_F_IC_up = np.zeros((len(K_range), len(rho_range)))
mc_delta_G_IC_up = np.zeros((len(K_range), len(rho_range)))
mc_delta_F_IC_down = np.zeros((len(K_range), len(rho_range)))
mc_delta_G_IC_down = np.zeros((len(K_range), len(rho_range)))


# for i, K in enumerate(K_range):
#     for j, rho in enumerate(rho_range):
#         print(str(i) + '::' + str(j))
#         kirk_prices[i, j] = kirk_approximation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
#         mc_prices[i, j] = mc_simulation(F1, G2, K, T, sigma_F, sigma_G, rho, r)
#         relative_errors[i, j] = (kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j]

start_time = time.time() 
for i, k in enumerate(K_range):
    for j, rho in enumerate(rho_range):
        print('K:' + str(k) + '::' + 'rho: ' + str(rho))
        
        #pricing the Call
        kirk_prices[i, j] = kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        modif_kirk_prices[i, j] = modif_kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        start_time_ij = time.time() 
        mc_prices[i, j] ,(mc_prices_IC_down[i,j], mc_prices_IC_up[i,j]) = mc_simulation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        end_time_ij.append(time.time()  - start_time_ij )
        
        relative_errors[i, j] = abs(kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j] * 100
        modif_relative_errors[i, j] = abs(modif_kirk_prices[i, j] - mc_prices[i, j]) / mc_prices[i, j] * 100
        
        #DeltaF:
        deltaF_kirk_prices[i, j] = delta_F_kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        deltaF_modifkirk_prices[i, j] = delta_F_modif_kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        deltaF_mc[i, j], (mc_delta_F_IC_down[i,j], mc_delta_F_IC_up[i,j]) = delta_F_mc_simulation(F1, G2, k, T, sigma_F, sigma_G, rho, r)

        deltaF_relative_errors[i, j] = abs((deltaF_kirk_prices[i, j] - deltaF_mc[i, j]) / deltaF_mc[i, j]) * 100
        deltaF_modif_relative_errors[i, j] = abs((deltaF_modifkirk_prices[i, j] - deltaF_mc[i, j]) / deltaF_mc[i, j]) * 100

        #DeltaG
        deltaG_kirk_prices[i, j] = delta_G_kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        deltaG_modifkirk_prices[i, j] = delta_G_modif_kirk_approximation(F1, G2, k, T, sigma_F, sigma_G, rho, r)
        deltaG_mc[i, j], (mc_delta_G_IC_down[i,j], mc_delta_G_IC_up[i,j]) = delta_G_mc_simulation(F1, G2, k, T, sigma_F, sigma_G, rho, r)

        deltaG_relative_errors[i, j] = abs(deltaG_kirk_prices[i, j]/ deltaG_mc[i, j] - 1) * 100
        deltaG_modif_relative_errors[i, j] = abs(deltaG_modifkirk_prices[i, j]/ deltaG_mc[i, j] - 1)  * 100

end_time = time.time()      
execution_time = end_time - start_time 
print(f"Execution Time: {execution_time} seconds")    
mean_time_ij = np.sum(end_time_ij)/len(end_time_ij)
print(f"Execution Time to price mc Call: {mean_time_ij} seconds")    
       








############################## Call Price ####################################################################
############################# ############################################################################################
###################################################################################################################
############################## *Table ########

rho_to_plot = [0.70, 0.80, 0.90, 0.999]
K_to_plot = [5,10,20]

# Loop through each row
for i_graph, rho in enumerate(rho_to_plot):
    j = rho_range.index(rho)
    for i_, k in enumerate(K_to_plot):
        i = K_range.tolist().index(k)
        print('###################################################################################################################')
        print('K= '+ str(k))
        print('rho= '+ str(rho))
        kirk_price_ = kirk_prices[i, j]
        modif_kirk_price_ =  modif_kirk_prices[i, j]
        mc_price_ = mc_prices[i, j]
        error = abs(kirk_price_/mc_price_ - 1)
        modif_error = abs(modif_kirk_price_/mc_price_ - 1)

        print('kirk: ' + "{:.4f}".format(kirk_price_))
        print('modif_kirk_prices: ' + "{:.4f}".format(modif_kirk_price_))
        print('mc_price_: ' + "{:.4f}".format(mc_price_) + " ({:.4f}".format(mc_prices_IC_down[i,j]) + ",{:.4f})".format(mc_prices_IC_up[i,j]))
        print('error: ' + "{:.4f}".format(error*100))
        print('modif_error: ' + "{:.4f}".format(modif_error*100))
        # print('ic_up: ' + "{:.4f}".format(mc_prices_IC_up[i,j]))
        # print('ic_donw: ' + "{:.4f}".format(mc_prices_IC_down[i,j]))
        
        
        
############################## PLOT  2D Pice ####################################################################
# rho_to_plot = [0.80, 0.85, 0.90, 0.95, 0.999]
rho_to_plot = [0.80, 0.90, 0.999]

# Number of rows
num_rows = len(rho_to_plot)

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(18, 6 * num_rows))

plot_ = 0
# Loop through each row
for i_graph, rho in enumerate(rho_to_plot):
    i = rho_range.index(rho)
    print(rho_range[i])
    ax1 = axes[i_graph, 0]
    ax2 = axes[i_graph, 1]
    
    # Plot on the left graph
    ax1.plot(K_range, relative_errors[:, i], label='Original Kirk formula')
    ax1.plot(K_range, modif_relative_errors[:, i], label='Modified Kirk formula')
    ax1.set_title(f'Error of Approx Formula Call Price (rho = {rho})')
    ax1.legend()
    ax1.set_xlabel('K')
    ax1.set_ylabel('Error in %')

    # Plot on the right graph
    ax2.plot(K_range[plot_:],kirk_prices[plot_:, i], label='Kirk Approximation Price', color='blue', linestyle='--')
    ax2.plot(K_range[plot_:],modif_kirk_prices[plot_:, i], label='Moodified Kirk Approximation Price', color='orange', linestyle='--')
    ax2.plot(K_range[plot_:],mc_prices[plot_:, i], label='mc price', color='green', alpha = 0.5)
    
        # Add confidence interval
    ax2.fill_between(K_range[plot_:], mc_prices_IC_down[plot_:, i], mc_prices_IC_up[plot_:, i], 
                     alpha=0.2, color='lightgreen', label='95% Confidence Interval')
    
    # Plot confidence interval boundaries
    ax2.plot(K_range[plot_:], mc_prices_IC_up[plot_:, i], color='green', linestyle=':', alpha=0.7, label='CI Upper Bound')
    ax2.plot(K_range[plot_:], mc_prices_IC_down[plot_:, i], color='green', linestyle=':', alpha=0.7, label='CI Lower Bound')
    

    ax2.set_title(f'Call Price (rho = {rho})')
    ax2.legend()
    ax2.set_xlabel('K')
    ax2.set_ylabel('Values')


# Adjust layout
# plt.tight_yout()
plt.subplots_adjust(hspace=0.9)  
plt.show()





############################## PLOT  3D ####################################################################
# Create surface plot
save=0
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
ax.set_zlabel('Relative Error (%)')
ax.set_title("Relative Errors Call Price: Abs(Kirk - MC) / MC")

# Add a color bar
#fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#ax.view_init(elev=30, azim=60)  # elev = vertical angle, azim = horizontal angle

if save:
    def update(frame):
        ax.view_init(elev=30, azim=frame)  # Rotate the plot by changing azim
        return ax,
    
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    
    ani.save('C:/Users/Marouane/OneDrive/Echange/Uniper/numerical result/rotating_3d_surface.mp4', writer='ffmpeg', fps=10)  # For .mp4
    


plt.tight_layout()
plt.show()






############################## Call Delta  F ####################################################################
############################# ############################################################################################

############################## *Table ########

rho_to_plot = [0.70, 0.80, 0.90, 0.999]
K_to_plot = [0,5,10,20]

# Loop through each row
for i_graph, rho in enumerate(rho_to_plot):
    j = rho_range.index(rho)
    for i_, k in enumerate(K_to_plot):
        i = K_range.tolist().index(k)
        print('###################################################################################################################')
        print('K= '+ str(k))
        print('rho= '+ str(rho))
        kirk_price_ = deltaF_kirk_prices[i, j]
        modif_kirk_price_ =  deltaF_modifkirk_prices[i, j]
        mc_price_ = deltaF_mc[i, j]
        error = abs(kirk_price_/mc_price_ - 1)
        modif_error = abs(modif_kirk_price_/mc_price_ - 1)

        print('kirk: ' + "{:.4f}".format(kirk_price_))
        print('modif_kirk_prices: ' + "{:.4f}".format(modif_kirk_price_))
        print('mc_price_: ' + "{:.4f}".format(mc_price_) + " ({:.4f}".format(mc_delta_F_IC_down[i,j]) + ",{:.4f})".format(mc_delta_F_IC_up[i,j]))
        print('error: ' + "{:.4f}".format(error*100))
        print('modif_error: ' + "{:.4f}".format(modif_error*100))
        
        
        
        
############################## PLOT  2D Pice ####################################################################
        
rho_to_plot = [0.80, 0.90, 0.999]
# Number of rows
num_rows = len(rho_to_plot)

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(18, 6 * num_rows))

# Loop through each row
for i_graph, rho in enumerate(rho_to_plot):
    i = rho_range.index(rho)
    print(rho_range[i])
    ax1 = axes[i_graph, 0]
    ax2 = axes[i_graph, 1]

    # Plot on the left graph
    ax1.plot(K_range, deltaF_relative_errors[:, i], label='Original Kirk formula')
    ax1.plot(K_range, deltaF_modif_relative_errors[:, i], label='Modified Kirk formula')
    ax1.set_title(f'Error of Approx Formula Delta F Call (rho = {rho})')
    ax1.legend()
    ax1.set_xlabel('K')
    ax1.set_ylabel('Error in %')

    # Plot on the right graph
    ax2.plot(K_range,deltaF_kirk_prices[:, i], label='Kirk Approximation Price', color='blue', linestyle='--')
    ax2.plot(K_range,deltaF_modifkirk_prices[:, i], label='Moodified Kirk Approximation Price', color='orange', linestyle='--')
    ax2.plot(K_range,deltaF_mc[:, i], label='mc price', color='green', alpha = 0.5)
    ax2.set_title(f'Call Delta F (rho = {rho})')
    ax2.legend()
    ax2.set_xlabel('K')
    ax2.set_ylabel('Values')


# Adjust layout
# plt.tight_yout()
plt.subplots_adjust(hspace=0.9)  
plt.show()




############################## PLOT  3D ####################################################################
# Create surface plot
save=0
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

K, RHO = np.meshgrid(K_range, rho_range)
surf = ax.plot_surface(K, RHO, deltaF_relative_errors.T, cmap='coolwarm',alpha=0.7)

modif_surf = ax.plot_surface(K, RHO, deltaF_modif_relative_errors.T, cmap='viridis', alpha=0.4, edgecolor='k')


# Add colorbars for both surfaces
cbar1 = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar1.set_label('Error (%) (Original Kirk)')
cbar2 = fig.colorbar(modif_surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar2.set_label('Error (%) (Modified Kirk)')



ax.set_xlabel('Strike Delta (K)')
ax.set_ylabel('Correlation (ρ)')
ax.set_zlabel('Relative Error (%)')
ax.set_title("Relative Errors Call Delta F: Abs(Kirk - MC) / MC")

# Add a color bar
#fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#ax.view_init(elev=30, azim=60)  # elev = vertical angle, azim = horizontal angle

if save:
    def update(frame):
        ax.view_init(elev=30, azim=frame)  # Rotate the plot by changing azim
        return ax,
    
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    
    ani.save('C:/Users/Marouane/OneDrive/Echange/Uniper/numerical result/DeltaF_Kirk.mp4', writer='ffmpeg', fps=10)  # For .mp4
    


plt.tight_layout()
plt.show()









############################## Call Delta  G ####################################################################
############################# ############################################################################################
############################# ############################################################################################


rho_to_plot = [0.70, 0.80, 0.90, 0.999]
K_to_plot = [5,10,20]

# Loop through each row
for i_graph, rho in enumerate(rho_to_plot):
    j = rho_range.index(rho)
    for i_, k in enumerate(K_to_plot):
        i = K_range.tolist().index(k)
        print('###################################################################################################################')
        print('K= '+ str(k))
        print('rho= '+ str(rho))
        kirk_price_ = deltaG_kirk_prices[i, j]
        modif_kirk_price_ =  deltaG_modifkirk_prices[i, j]
        mc_price_ = deltaG_mc[i, j]
        error = abs(kirk_price_/mc_price_ - 1)
        modif_error = abs(modif_kirk_price_/mc_price_ - 1)

        print('kirk: ' + "{:.4f}".format(kirk_price_))
        print('modif_kirk_prices: ' + "{:.4f}".format(modif_kirk_price_))
        print('mc_price_: ' + "{:.4f}".format(mc_price_)+ " ({:.4f}".format(mc_delta_G_IC_down[i,j]) + ",{:.4f})".format(mc_delta_G_IC_up[i,j]))
        print('error: ' + "{:.4f}".format(error*100))
        print('modif_error: ' + "{:.4f}".format(modif_error*100))
        
        
        
        
        




############################## PLOT  2D Pice ####################################################################
rho_to_plot = [0.80, 0.90, 0.999]
# Number of rows
num_rows = len(rho_to_plot)

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(18, 6 * num_rows))

# Loop through each row
for i_graph, rho in enumerate(rho_to_plot):
    i = rho_range.index(rho)
    print(rho_range[i])
    ax1 = axes[i_graph, 0]
    ax2 = axes[i_graph, 1]

    # Plot on the left graph
    ax1.plot(K_range, deltaG_relative_errors[:, i], label='Original Kirk formula')
    ax1.plot(K_range, deltaG_modif_relative_errors[:, i], label='Modified Kirk formula')
    ax1.set_title(f'Error of Approx Formula Delta G Call (rho = {rho})')
    ax1.legend()
    ax1.set_xlabel('K')
    ax1.set_ylabel('Error in %')

    # Plot on the right graph
    ax2.plot(K_range,deltaG_kirk_prices[:, i], label='Kirk Approximation Price', color='blue', linestyle='--')
    ax2.plot(K_range,deltaG_modifkirk_prices[:, i], label='Moodified Kirk Approximation Price', color='orange', linestyle='--')
    ax2.plot(K_range,deltaG_mc[:, i], label='mc price', color='green', alpha = 0.5)
    ax2.set_title(f'Call Delta G (rho = {rho})')
    ax2.legend()
    ax2.set_xlabel('K')
    ax2.set_ylabel('Values')


# Adjust layout
# plt.tight_yout()
plt.subplots_adjust(hspace=0.9)  
plt.show()




############################## PLOT  3D ####################################################################
# Create surface plot
save=0
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

K, RHO = np.meshgrid(K_range, rho_range)
surf = ax.plot_surface(K, RHO, deltaG_relative_errors.T, cmap='coolwarm',alpha=0.7, edgecolor='k')

modif_surf = ax.plot_surface(K, RHO, deltaG_modif_relative_errors.T, cmap='viridis', alpha=0.5)


# Add colorbars for both surfaces
cbar1 = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar1.set_label('Error (%) (Original Kirk)')
cbar2 = fig.colorbar(modif_surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar2.set_label('Error (%) (Modified Kirk)')



ax.set_xlabel('Strike Delta (K)')
ax.set_ylabel('Correlation (ρ)')
ax.set_zlabel('Relative Error (%)')
ax.set_title("Relative Errors Call Delta G: Abs((Kirk - MC) / MC)")

# Add a color bar
#fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#ax.view_init(elev=30, azim=60)  # elev = vertical angle, azim = horizontal angle

if save:
    def update(frame):
        ax.view_init(elev=30, azim=frame)  # Rotate the plot by changing azim
        return ax,
    
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    
    ani.save('C:/Users/Marouane/OneDrive/Echange/Uniper/numerical result/deltaG_Kirk.mp4', writer='ffmpeg', fps=10)  # For .mp4
    


plt.tight_layout()
plt.show()







