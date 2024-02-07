# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:10:01 2024

@author: Lukas
"""

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fdmsolver import load_map, init_linear_system, solve_linear_system


# Define a uniform grid of spatial points
N_x, N_y = 500, 500
h = 1E-3  # Distance between each sample
width_V, height_V = (N_x - 1) * h, (N_y - 1) * h
width_E, height_E = (N_x - 2) * h, (N_y - 2) * h  # E and eps_r are on a staggered grid from the voltage potentials

# Set Dirichlet boundary conditions
dirichlet = {
    'mask': np.zeros((N_x, N_y), dtype=bool),
    'value': np.zeros((N_x, N_y))}

mask = load_map('./data/mouth.png', False, True, N=(N_x, N_y), twobit=True)
dirichlet['mask'][mask] = True
dirichlet['value'][mask] = 40

mask = load_map('./data/head.png', False, True, (N_x, N_y), twobit=True)
dirichlet['mask'][mask] = True
dirichlet['value'][mask] = 0

# Set Neumann boundary conditions
neumann = {
    'mask': np.zeros((N_x, N_y), dtype=bool),
    'value': np.zeros((N_x, N_y)),
    'dir': np.zeros((N_x, N_y, 2), dtype=int)}

# Set charges
rho = np.zeros((N_x, N_y))
rho += load_map('./data/eye_left.png', 0, 1E-5, N=(N_x, N_y), twobit=True)
rho += load_map('./data/eye_right.png', 0, 1E-5, N=(N_x, N_y), twobit=True)

# Set permittivities
eps_r = load_map('./data/background.png', 1, 100, N=(N_x - 1, N_y - 1), twobit=False)

# Initialize the linear system
# Note: no boundary conditions for left and right means periodic
A, b = init_linear_system(N_x, N_y, h, dirichlet, neumann, rho, eps_r)

# Solve the linear system and compute E-field magnitude
V, E = solve_linear_system(A, b, N_x, N_y, h)
E_magn = norm(E, axis=-1)

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

im = axs[0].imshow(eps_r.T, extent=[0, width_E, 0, height_E], origin='lower', cmap='jet', interpolation='nearest')
axs[0].set_title('Dielectric Constant')
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('y (m)')
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.15)
fig.colorbar(im, label='Dielectric Constant', cax=cax)

im = axs[1].imshow(V.T, extent=[0, width_V, 0, height_V], origin='lower', cmap='jet', interpolation='nearest')
axs[1].set_title('Electric Potential')
axs[1].set_xlabel('x (m)')
axs[1].set_ylabel('y (m)')
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.15)
fig.colorbar(im, label='Electric Potential (V)', cax=cax)

im = axs[2].imshow(E_magn.T, extent=[0, width_E, 0, height_E], origin='lower', cmap='jet', interpolation='nearest')
axs[2].set_title('Electric Field Magnitude')
axs[2].set_xlabel('x (m)')
axs[2].set_ylabel('y (m)')
divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.15)
fig.colorbar(im, label='Electric Field (V/m)', cax=cax)

plt.tight_layout(pad=1.5)
plt.show()
