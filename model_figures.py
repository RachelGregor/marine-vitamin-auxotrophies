#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:45:38 2024

@author: vercelli
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import seaborn as sns

# Graph of growth vs lysis rate
filename = 'simulation_results_lysis_E3M18'
Ks = 1
degrader_max_growth_rate = 0.26

with open(filename, 'rb') as f:
    results = pickle.load(f)
    

vit_value = []
sub_value = []
vit_lim = []
sub_lim = []
lysis_percentage = []

for key, item in results.items():
    sub, vit, bac = item['results']
    lysis_percentage.append(item['lysis_percentage'])
    Kv = item['vit_info'][2]
    
    mean_vit = np.mean(vit.data[bac.data > 0])
    mean_sub = np.mean(sub.data[bac.data > 0])
    vit_value.append(mean_vit)
    sub_value.append(mean_sub)
    vit_lim.append(Kv/(Kv + mean_vit))
    sub_lim.append(Ks/(Ks + mean_sub))

plt.figure("Growth vs Lysis")
x_data = np.asarray(lysis_percentage)*degrader_max_growth_rate
plt.plot(x_data, 100*(1-np.array(vit_lim)), '-', label='Vitamin limited')
plt.plot(x_data, 100*(1-np.array(sub_lim)), '-', label='Carbon limited')
plt.plot(x_data, 100*(1-np.array(vit_lim))*(1-np.array(sub_lim)), '-', label='Colimited')
plt.xlabel("Lysis rate per hour ($\mu$)", fontsize=14)
plt.xlim([0, 0.26])
plt.ylim([0, 100])
plt.ylabel("Percentage max growth", fontsize=14)
plt.legend(loc='upper right')
plt.grid()

print(filename.split('_')[3], 100*(1-np.array(vit_lim[20])))

plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(f"fig_lysis_{filename.split('_')[3]}.svg")

# Phasespace plot
filename = 'phasespace/simulation_results_phasespace'

with open(filename, 'rb') as f:
    results = pickle.load(f)

Km_values = []
for i in range(-12, -4):
    Km_values.append(f'$10^{str({i})}$')
    Km_values.append('')
Km_values.pop()

Yv_inv_values = []
for i in range(0, 7):
    Yv_inv_values.append(f'$10^{str({i})}$')
    Yv_inv_values.append('')
Yv_inv_values.pop()
Yv_inv_values = Yv_inv_values[::-1]

data = np.zeros((15, 13), dtype=float)
for key, item in results.items():
    i,j = [int(i) for i in key.split(',')]
    print(i, j)
    vit_info = item['vit_info']
    sub, vit, bac = item['results']
    
    mean_vit = np.mean(vit.data[bac.data > 0])
    percentage_max_growth = mean_vit/(vit_info[2] + mean_vit)
    data[i, -j-1] = percentage_max_growth*100

plt.figure("Phasespace")
ax = sns.heatmap(np.transpose(data), vmin=0, vmax=100, cmap='Blues', square=True, xticklabels=Km_values, yticklabels=Yv_inv_values)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Percentage max growth', fontsize=10)
plt.xlabel('Km (M)', fontsize=14)
plt.ylabel('Molecules per cell ($Y_v^{^{-1}}$)', fontsize = 14)
plt.tick_params(axis='x', labelrotation=0)
plt.tick_params(axis='y', labelrotation=0)

# Values measured for single isolates
Bvit_Km = [9.6E-09, 1.7E-10, 4.6E-06, 3.7E-08, 1.3E-10, 3.1E-10, 2.8E-11]
Bvit_Yv_inv = [20000, 20000, 100000, 100000, 100, 100, 5000]

Bvit_Km = 2*(np.log10(Bvit_Km) + 12) + 0.5
Bvit_Yv_inv = 12 - (2*np.log10(Bvit_Yv_inv) - 0.5)
plt.plot(Bvit_Km, Bvit_Yv_inv, 'r.')

plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(f"fig_phasespace.svg")
    
    
    