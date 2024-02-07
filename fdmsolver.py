# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:48:00 2024

@author: Lukas
"""

import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# Define constants
eps_0 = 8.8541878128E-12


# Load a map from a grayscale image
def load_map(filename, value_white, value_black, N=None, twobit=False):
    # Load and transform image to desired format
    img = np.array(mpimg.imread(filename))[:, :, :3]  # Crop first three channels
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, axis=1)
    
    # Normalize integer data to float 0.-1.
    if np.issubdtype(img.dtype, np.integer):
        img = np.asarray(img, dtype=float) / np.iinfo(img.dtype).max
    
    # Resize image
    if not N is None:
        resize_x = N[0] / img.shape[0]
        resize_y = N[1] / img.shape[1]
        img = rescale(img, scale=(resize_x, resize_y, 1), mode='constant')
    
    # Interpret image as grayscale
    img = rgb2gray(img)
        
    # Compute resulting map and return
    if twobit:
        out = np.where(img > 0.5, value_white,  value_black)
    else:
        out = img * (value_white - value_black) + value_black
    
    return out


# Initialize the linear system
def init_linear_system(N_x, N_y, h, dirichlet=None, neumann=None, rho=None, eps_r=1):
    # Set default data
    dirichlet = {
        'mask': np.zeros((N_x, N_y), dtype=bool),
        'value': np.zeros((N_x, N_y))} if dirichlet is None else dirichlet
    
    neumann = {
        'mask': np.zeros((N_x, N_y), dtype=bool),
        'value': np.zeros((N_x, N_y)),
        'dir': np.zeros((N_x, N_y, 2), dtype=int)} if neumann is None else neumann
    
    rho = np.zeros((N_x, N_y)) if rho is None else rho
    
    eps_r = np.ones((N_x, N_y)) * eps_r if not hasattr(eps_r, '__len__') else eps_r
    eps_r = np.hstack((eps_r, eps_r[:, 0].reshape(-1, 1)))  # Ensure wrapping around (periodic boundaries)
    eps_r = np.vstack((eps_r, eps_r[0, :]))  # "
    
    # Setup up the linear system
    A = lil_matrix((N_x * N_y, N_x * N_y))  # Init A as a sparse matrix (LIL format)
    b = np.zeros((N_x * N_y, 1))
    idx_map = np.arange(N_x * N_y).reshape((N_x, N_y)) # Map to convert 2D indices to 1D index
    
    # Iterate through indices
    for i in range(0, N_x):
        for j in range(0, N_y):
            # Apply Dirichlet boundary condition
            if dirichlet['mask'][i, j]:
                idx = idx_map[i, j]
                A[idx, idx] = 1
                b[idx] = dirichlet['value'][i, j]
            
            # Apply Neumann boundary condition
            elif neumann['mask'][i, j]:
                idx_1 = idx_map[i, j]
                idx_2 = idx_map[i + neumann['dir'][i, j][0], j + neumann['dir'][i, j][1]]  # Index of the "inner sample"
                A[idx_1, idx_1] = 1
                A[idx_1, idx_2] = -1
                b[idx_1] = neumann['value'][i, j] * h
        
            else:
                idx = idx_map[i, j]
                idx_l = idx_map[i - 1, j]
                idx_r = idx_map[(i + 1) % N_x, j]  # Ensure wrapping around (periodic boundaries)
                idx_t = idx_map[i, (j + 1) % N_y]  # Ensure wrapping around (periodic boundaries)
                idx_b = idx_map[i, j - 1]
                
                # Apply free charge
                b[idx] = -rho[i, j] * h**2 / eps_0
                
                # Apply remainder by five-point star
                a_0 = eps_r[i, j] + eps_r[i - 1, j] + eps_r[i, j - 1] + eps_r[i - 1, j - 1]
                a_1 = (eps_r[i, j] + eps_r[i, j - 1]) / 2
                a_2 = (eps_r[i - 1, j] + eps_r[i, j]) / 2
                a_3 = (eps_r[i - 1, j - 1] + eps_r[i - 1, j]) / 2
                a_4 = (eps_r[i, j - 1] + eps_r[i - 1, j - 1]) / 2
                
                A[idx, idx] = -a_0
                A[idx, idx_l] = a_3
                A[idx, idx_r] = a_1
                A[idx, idx_t] = a_2
                A[idx, idx_b] = a_4
            
    return A, b


# Solve sparse linear system
def solve_linear_system(A, b, N_x, N_y, h):
    x = spsolve(A.tocsc(), b)  # first convert A to CSC format
    V = x.reshape([N_x, N_y])
    
    # Compute E-field on a staggered grid from the voltage potentials
    Ex = -(V[1:, :] - V[:-1, :]) / h
    Ey = -(V[:, 1:] - V[:, :-1]) / h
    
    Ex = (Ex[:, 1:] + Ex[:, :-1]) / 2
    Ey = (Ey[1:, :] + Ey[:-1, :]) / 2
    
    E = np.transpose((Ex, Ey), [1, 2, 0])
    
    return V, E
