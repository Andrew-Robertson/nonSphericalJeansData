import numpy as np
import os
import simData.eagleSims as eagle
import simData.haloShapes as shapes
import sys


# make synthetic data
def make_deformed_isothermal_sphere(N=10000, Rmax=10.0, scale=[1.0, 1.0, 1.0], seed=None):
    """
    Generate N particles from a truncated isothermal sphere, then stretch/squash axes.

    Parameters
    ----------
    N : int
        Number of particles.
    Rmax : float
        Truncation radius of the spherical profile.
    scale : list of 3 floats
        Scaling factors for [x, y, z] directions.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pos : ndarray, shape (N, 3)
        3D positions of particles after deformation.
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample r from P(<r) ∝ r => r ∝ sqrt(u)
    u = np.random.rand(N)
    r = Rmax * u

    # Sample directions uniformly on a sphere
    cos_theta = 2*np.random.rand(N) - 1
    phi = 2*np.pi*np.random.rand(N)
    sin_theta = np.sqrt(1 - cos_theta**2)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta

    pos = np.vstack([x, y, z]).T

    # Apply deformation
    scale = np.asarray(scale)
    pos *= scale

    return pos

pos = make_deformed_isothermal_sphere(N=int(1e5), Rmax=20.0, scale=[1.0, 1.0, 0.5], seed=None)

rmaxs = np.geomspace(0.1,10,20) #Mpc
shell_fac = np.sqrt(rmaxs[0]/rmaxs[1])

dm_Q,dm_q,dm_e1,dm_e2,dm_a,dm_b,dm_flags,dm_Ns = shapes.axisymmetric_shape_with_radius(pos,rmaxs,centre=[0,0,0],zvec=[0,0,1],mass=None,print_outputs=False,tollerance=0.0001,max_attempts=100,shell=True,reduced=False,Boxsize=50,shell_fac=shell_fac)

q, N = shapes.shape_axisymmetric_with_radius(pos, rs=rmaxs, print_outputs=False, shell_fac=shell_fac)
Nboot = 25
dm_Q_array = np.zeros((Nboot, len(rmaxs)))
dm_N_array = np.zeros((Nboot, len(rmaxs)))
for i in np.arange(Nboot):
    Npart = pos.shape[0]
    indices = np.random.choice(Npart, size=Npart, replace=True)
    resampled_pos = pos[indices]
    dm_Q_array[i], dm_N_array[i]= shapes.shape_axisymmetric_with_radius(resampled_pos, rs=rmaxs, print_outputs=False, shell_fac=shell_fac)