#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:52:14 2023

Simulation of biofilm growth on a chitin particle including vitamin dynamics

@author: vercelli
"""

import matplotlib.pyplot as plt
import numpy as np
from pde import MemoryStorage, ScalarField, SphericalSymGrid, \
FieldCollection, PDEBase, solvers
from pde.tools.numba import jit
import pickle
import os

class SubstrateVitaminPDE(PDEBase):
    
    def __init__(self, 
                 bac_info=[1, 1, 1], 
                 sub_info=[1, 1, 1], 
                 vit_info=[1, 1, 1], 
                 bc_sub="auto_periodic_neumann", 
                 bc_vit="auto_periodic_neumann"
                 ):
        
        self.max_growth_rate = bac_info[0]
        self.sub_diffusivity = sub_info[0]
        self.sub_yield = sub_info[1]
        self.sub_affinity = sub_info[2]
        self.vit_diffusivity = vit_info[0]
        self.vit_yield = vit_info[1]
        self.vit_affinity = vit_info[2]
        self.bc_sub = bc_sub
        self.bc_vit = bc_vit

    def evolution_rate(self, state, t=0):
        sub, vit, bac = state
        sub_monod = sub / (sub + 1)
        vit_monod = vit / (vit + 1)
        colimited_growth_rate = self.max_growth_rate * sub_monod * vit_monod
        
        sub_t = (self.sub_diffusivity * sub.laplace(bc=self.bc_sub)
                 - colimited_growth_rate * bac / (self.sub_affinity * self.sub_yield) ) # (mM_sub/s per Ks)
        vit_t = (self.vit_diffusivity * vit.laplace(bc=self.bc_vit)
                 - colimited_growth_rate * bac / (self.vit_affinity * self.vit_yield) ) # (pM_vit/s per Kv times Yv)
        bac_t = ScalarField(bac.grid, 0)
        
        return FieldCollection([sub_t, vit_t, bac_t])
    
    def _make_pde_rhs_numba(self, state):
        """ the numba-accelerated evolution equation """
        # make attributes locally available
        vit_diffusivity = self.vit_diffusivity
        vit_affinity = self.vit_affinity
        vit_yield = self.vit_yield
        sub_diffusivity = self.sub_diffusivity
        sub_affinity = self.sub_affinity
        sub_yield = self. sub_yield
        max_growth_rate = self.max_growth_rate

        # create operators
        laplace_sub = state.grid.make_operator("laplace", bc=self.bc_sub)
        laplace_vit = state.grid.make_operator("laplace", bc=self.bc_vit)

        @jit
        def pde_rhs(state_data, t=0):
            """ compiled helper function evaluating right hand side """
            results = np.zeros(state_data.shape)
            
            sub_laplacian = laplace_sub(state_data[0])
            vit_laplacian = laplace_vit(state_data[1])
            
            sub_monod = state_data[0] / (state_data[0] + 1)
            vit_monod = state_data[1] / (state_data[1] + 1)
            colimited_growth_rate = max_growth_rate * sub_monod * vit_monod
            
            results[0] = (sub_diffusivity * sub_laplacian
                     - colimited_growth_rate * state_data[2] / (sub_affinity * sub_yield) ) # (mM_sub/s per Ks)
            results[1] = (vit_diffusivity * vit_laplacian
                     - colimited_growth_rate * state_data[2] / (vit_affinity * vit_yield) ) # (pM_vit/s per Kv)
            
            return results
        
        return pde_rhs

def run_simulation(lysis_percentage, Ks, Kv, Yv_inv, m=1, sim_name='test'):
    # Define geometry
    particle_radius = 150e-4 # cm
    grid_radius = 10e-4 + particle_radius # cm
    N = 30
    
    # Define spherically symmetric grids
    grid = SphericalSymGrid(radius=[particle_radius, grid_radius], shape=N)  # generate grid
    
    # Auxotrophic cross-feeder info
    bac_info = [8.3e-5, 1100, 1e-12] # [max growth rate (1/s), density (g_cell/L), volume (cm^3/cell)]
    bac_density = 555*1 # 0.5 cells/um^3 * (9e8 CFU/ml/OD)^-1
    
    # Degrader info
    effective_growth_rate = 0.26/3600 # 1/s
    lysis_rate = lysis_percentage*effective_growth_rate # 1/s
    sigma_0 = 0.782 # OD*cm
    surface_cell_density = max(sigma_0*(1-lysis_rate/effective_growth_rate), 0) # OD*cm
    vitamin_storage_multiplier = m
    
    # Substrate info
    sub_info = [12e-6, 31*1e-3, Ks] # [Diffusivity (cm^2/s), yield (g_cell/mMol_sub), affinity (mM_sub)]
    sub_influx = surface_cell_density * 2.47*1e-1*1e-3/Ks # mM_sub * cm / s per Monod
    bc_lower = {"derivative": sub_influx/sub_info[0]} # mM_sub / cm
    Rf = grid_radius # cm
    bc_upper = {"type": "mixed", "value": 1/Rf, "const": 0}
    bc_sub = [bc_lower, bc_upper]
    
    # Vitamin info
    Yv = 0.6691/Yv_inv  # (molecules/CFU * (9e11) CFU/L / OD * (6.022e23)^-1 mol/molecules * (1e12) pmol/mol)^-1 = OD/pM
    vit_info = [3.67e-6, Yv, Kv] # [Diffusivity (cm^2/s), yield (OD/pM_vit), affinity (pM_vit)]
    vit_media_concentration = 0/Kv #26e-12 # pM (1e-10-1e-13)
    vit_influx = vitamin_storage_multiplier*surface_cell_density*lysis_rate/Yv/Kv # pM_vit*cm/s per Kv
    bc_lower = {"derivative": vit_influx/vit_info[0]} # pM_vit / cm
    bc_upper = {"type": "mixed", "value": 1/Rf, 
                "const": vit_media_concentration/Rf}
    bc_vit = [bc_lower, bc_upper]
    
    # Initialize biofilm
    biofilm = 1e-4 + particle_radius # cm
    bac = bac_density * ScalarField.from_expression(grid, f"r < {biofilm}") # OD
    
    # Initialize chemical fields
    x_data = grid.axes_coords[0] * 1e4 # um
    sub_analytic = (sub_influx/sub_info[0] * particle_radius**2 * 1e4)/x_data
    vit_analytic = (vit_influx/vit_info[0] * particle_radius**2 * 1e4)/x_data + \
    vit_media_concentration
    
    # Solve equation starting from analytical solution
    initial_sub = ScalarField(grid, sub_analytic)
    initial_vit = ScalarField(grid, vit_analytic)
    state = FieldCollection(
        [ScalarField(grid, list(map(max, zip(ScalarField(grid, 0).data, initial_sub.data)))), 
         ScalarField(grid, list(map(max, zip(ScalarField(grid, 0).data, initial_vit.data)))),
         bac]
        ) # [substrate (mM), vitamin (pM), bacteria density (g_cell/cm^3)]
    
    eq = SubstrateVitaminPDE(
        bac_info=bac_info,
        sub_info=sub_info,
        vit_info=vit_info,
        bc_sub=bc_sub,
        bc_vit=bc_vit
        )
    
    vit_storage = MemoryStorage()
    t_diff = 0.5*(grid_radius-particle_radius)**2/sub_info[0]
    t_range = 100*t_diff
    solver = solvers.explicit.ExplicitSolver(eq, scheme='rk', adaptive=True, tolerance=1e-4)
    trackers = [
        vit_storage.tracker(1),
        ]
    controller = solvers.Controller(solver, t_range=t_range, tracker=trackers)
    vit_result = controller.run(state)
    vit_result[0].data = vit_result[0].data*Ks
    vit_result[1].data = vit_result[1].data*Kv
    sub_analytic = sub_analytic*Ks
    vit_analytic = vit_analytic*Kv
    
    mean_vit = np.mean(vit_result[1].data[bac.data > 0])
    r_auxo = mean_vit/(mean_vit + Kv)
    results_dict[sim_name] = {'results': vit_result, 'vit_info': vit_info, 'sub_info': sub_info, 'lysis_percentage': lysis_percentage, 'r_auxo': r_auxo}
    print(sim_name, 'mean_vit: ', mean_vit, ' r: ', r_auxo)
    
    # Plot summary figures to evaluate simulation
    xmax = grid_radius*1e4
    xmin = particle_radius*1e4
    legend = []
    
    plot_analytic = True
    
    plt.figure("Substrate")
    if plot_analytic:
        plt.plot(x_data, sub_analytic, 'r--')
        #plt.plot(x_data, initial_sub.data, 'k--', label='_nolegend_')
        legend.extend(["analytic"])
    plt.vlines(xmin, 0, max(sub_analytic.data), colors='k', linestyles='dashed', label='_nolegend_')
    plt.xlabel("Radial distance (um)", fontsize=14)
    plt.ylabel("Substrate (mM)", fontsize=14)
    plt.xlim([xmin - 10, xmax])
    
    
    plt.figure("Vitamin")
    if plot_analytic:   
        plt.plot(x_data, vit_analytic, 'r--')
        #plt.plot(x_data, initial_vit.data, 'k--', label='_nolegend_')
    plt.vlines(xmin, 0, max(vit_analytic.data), colors='k', linestyles='dashed', label='_nolegend_')
    plt.xlabel("Radial distance (um)", fontsize=14)
    plt.ylabel("Vitamin (pM)", fontsize=14)
    plt.xlim([xmin - 10, xmax])
    
    plt.figure("Substrate limitation")
    if plot_analytic:
        plt.plot(x_data, sub_info[2]/(sub_info[2] + sub_analytic), 'r--')
    plt.vlines(xmin, 0, 1, colors='k', linestyles='dashed', label='_nolegend_')
    plt.xlabel("Radial distance (um)", fontsize=14)
    plt.ylabel("Substrate limitation", fontsize=14)
    plt.xlim([xmin - 10, xmax])
    plt.ylim([0, 1.1])
    
    plt.figure("Vitamin limitation")
    if plot_analytic:
        plt.plot(x_data, vit_info[2]/(vit_info[2] + vit_analytic), 'r--')
    plt.vlines(xmin, 0, 1, colors='k', linestyles='dashed', label='_nolegend_')
    plt.xlabel("Radial distance (um)", fontsize=14)
    plt.ylabel("Vitamin limitation", fontsize=14)
    plt.xlim([xmin - 10, xmax])
    plt.ylim([0, 1.1])
    
    plt.figure("Percent growth rate")
    if plot_analytic:
        sub_monod = sub_analytic/(sub_analytic + sub_info[2])
        vit_monod = vit_analytic/(vit_analytic + vit_info[2])
        colimited_growth_rate = sub_monod*vit_monod
        plt.plot(x_data, colimited_growth_rate, 'r--')
    plt.vlines(xmin, 0, 1, colors='k', linestyles='dashed', label='_nolegend_')
    plt.xlabel("Radial distance (um)", fontsize=14)
    plt.ylabel("Growth percentage (r/r_max)", fontsize=14)
    
    x_point = []
    y_point = []
    
    for key, item in sorted(results_dict.items()):
        if 'results' not in item:
            continue
        key = str(key)
        sub, vit, bac = item['results']
        sub_info = item['sub_info']
        vit_info = item['vit_info']
        
        plt.figure("Substrate")
        y_data = sub.data
        plt.plot(x_data, y_data)
        # legend.append(key)
        # plt.legend(legend, loc='upper right')
        # plt.savefig("sub_profiles.svg")
        
        plt.figure("Vitamin")
        y_data = vit.data
        plt.plot(x_data, y_data)
        # plt.legend(legend, loc='upper right')
        # plt.savefig("vit_profiles.svg")
        
        plt.figure("Substrate limitation")
        y_data = sub_info[2]/(sub_info[2] + sub.data)
        plt.plot(x_data, y_data)
        # plt.legend(legend, loc='lower right')
        # plt.savefig("sub_limitation.svg")
        
        plt.figure("Vitamin limitation")
        y_data = vit_info[2]/(vit_info[2] + vit.data)
        plt.plot(x_data, y_data)
        # plt.legend(legend, loc='upper left')
        # plt.savefig("vit_limitation.svg")
        
        plt.figure("Percent growth rate")
        sub_monod = sub.data/(sub.data + sub_info[2])
        vit_monod = vit.data/(vit.data + vit_info[2])
        colimited_growth_rate = sub_monod*vit_monod
        plt.plot(x_data, colimited_growth_rate)
        # plt.legend(legend, loc="upper right")
        # plt.savefig("growth_percentage.svg")
        
        plt.figure("Limitation landscape")
        mean_vit = np.mean(vit.data[bac.data > 0])
        mean_sub = np.mean(sub.data[bac.data > 0])
        x_point.append(vit_info[2]/(vit_info[2] + mean_vit))
        y_point.append(sub_info[2]/(sub_info[2] + mean_sub))
        plt.plot(x_point[-1], y_point[-1], 'x')
    
    
    plt.figure("Limitation landscape")
    #plt.plot(x_point, y_point, 'x-')
    plt.xlabel("Vitamin limitation", fontsize=14)
    plt.ylabel("Substrate limitation", fontsize=14)
    plt.xlim([-.1, 1.1])
    plt.ylim([0, 1.01])
    plt.grid()
    
    plt.show()


if __name__ == '__main__':
    # Growht vs lysis graph for B1 - E3M18
    results_dict = dict()
    acetate_affinity = 1 # in mM units
    Kv = 170 # in pM units
    Yv_inv = 20000 # in molecules/CFU
    lysis_percentage = np.linspace(0, 1, 41)
    
    for ind, lysis in enumerate(lysis_percentage):
        sim_name = f'{ind}'
        run_simulation(lysis, acetate_affinity, Kv, Yv_inv, sim_name=sim_name)
    
    # savefolder = "simulation_results"
    # with open(f'{savefolder}/simulation_results_lysis_E3M18', 'wb') as f:
    #     pickle.dump(results_dict, f)

    # Phasespace simulation
    results_dict = dict()
    acetate_affinity = 1 # in mM units
    Kv = np.logspace(0, 7, 15) # in pM units
    Yv_inv = np.logspace(0, 6, 13) # in molecules/CFU
    lysis_percentage = 0.5
    
    
    for i, k in enumerate(Kv):
        for j, y in enumerate(Yv_inv):
            sim_name = f'{i}, {j}'
            run_simulation(lysis_percentage, acetate_affinity, k, y, sim_name=sim_name)
    
    # savefolder = "simulation_results/phasespace"
    # with open(f'{savefolder}/simulation_results_phasespace', 'wb') as f:
    #     pickle.dump(results_dict, f)









    
    
