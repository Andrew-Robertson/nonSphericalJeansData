import numpy as np
from scipy.spatial import cKDTree as KDTree

def density_profile(pos, mass, centre=[0, 0, 0], rmin=1e-4, rmax=2, bins=100, r200=None, Boxsize=50):
    """
    Compute spherical density profile and enclosed mass from particle data.

    Parameters:
    - pos: (N, 3) array of particle positions.
    - mass: (N,) array of particle masses.
    - centre: 3-element list for center of profile.
    - rmin, rmax: Min and max radius for profile.
    - bins: Number of radial bins.
    - r200: Optional radius for computing m200.

    Returns:
    - rs: Radius at bin centers.
    - rho: Density in each bin.
    - redges: Bin edges.
    - mass_within_radii: Enclosed mass at each edge.
    - (Optional) r200, m200 if r200 provided.
    """
    redges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    rs = np.sqrt(redges[1:] * redges[:-1])  # Geometric mean for bin centers
    bin_vol = (4/3) * np.pi * (redges[1:]**3 - redges[:-1]**3)

    p = pos - centre
    p[p > 0.5 * Boxsize] -= Boxsize
    p[p < -0.5 * Boxsize] += Boxsize
    r = np.linalg.norm(p, axis=1)
    mass_in_shells = np.histogram(r, bins=redges, weights=mass)[0]
    rho = mass_in_shells / bin_vol

    mass_within_radii = np.concatenate([[np.sum(mass[r < redges[0]])],
                                        np.cumsum(mass_in_shells) + np.sum(mass[r < redges[0]])])
    if r200 is None:
        return rs, rho, redges, mass_within_radii
    else:
        m200 = np.sum(mass[r < r200])
        return rs, rho, redges, mass_within_radii, r200, m200

def cylindrical_density_profile(pos,mass,centre=[0,0,0],zvec=[0,0,1],Rmin=0,Rmax=0.1,Rbins=100,Zmax=0.1,Zbins=200, Boxsize=50):
    p = pos - centre
    p[p > 0.5 * Boxsize] -= Boxsize
    p[p < -0.5 * Boxsize] += Boxsize
    r = np.sqrt(np.sum(p**2,axis=1))
    zhat = np.asarray(zvec)/np.linalg.norm(zvec)
    z = np.sum(zhat*p,axis=1)
    R = np.sqrt(r**2 - z**2)
    mass_in_cylindrical_shells, Redges, Zedges = np.histogram2d(R, z, bins=[Rbins,Zbins], range=[[Rmin,Rmax],[-Zmax,Zmax]], weights=mass)
    bin_vol = np.outer(np.pi*(Redges[1:]**2-Redges[:-1]**2),Zedges[1:]-Zedges[:-1])
    rho = mass_in_cylindrical_shells / bin_vol
    return rho,Redges,Zedges


def increase_res(pos, mass, res_fac=10, Nngb=32, smooth_fac=1, max_extent=None):
    """
    Refine particle resolution by cloning and scattering particles.

    Parameters:
    - pos: Particle positions.
    - mass: Particle masses.
    - res_fac: Number of clones per particle.
    - Nngb: Number of neighbors to define smoothing scale.
    - smooth_fac: Multiplier for smoothing kernel.
    - max_extent: Optional spatial limit for refinement.

    Returns:
    - high_res_pos, high_res_mass: Refined particle positions and masses.
    """
    if max_extent is not None:
        r = np.linalg.norm(pos, axis=1)
        mask = r < max_extent
        pos, mass = pos[mask], mass[mask]

    tree = KDTree(pos)
    dd, _ = tree.query(pos, k=Nngb)
    h = dd[:, -1]

    noise = np.random.randn(len(pos), res_fac, 3) * h[:, None, None] * smooth_fac
    high_res_pos = pos[:, None, :] + noise
    high_res_mass = np.repeat(mass, res_fac) / res_fac

    return high_res_pos.reshape(-1, 3), high_res_mass


def increase_res_adaptive(pos, mass, high_res_ind, res_fac=10, high_res_fac=100, Nngb=32, smooth_fac=1, max_extent=None):
    """
    Adaptive resolution refinement, using higher resolution for selected particles.

    Parameters:
    - high_res_ind: Boolean mask for which particles get higher refinement.
    - high_res_fac: Number of clones for high-res particles.

    Returns:
    - high_res_pos, high_res_mass: Refined particle data.
    """
    if max_extent is not None:
        r = np.linalg.norm(pos, axis=1)
        mask = r < max_extent
        pos, mass, high_res_ind = pos[mask], mass[mask], high_res_ind[mask]

    tree = KDTree(pos)
    dd, _ = tree.query(pos, k=Nngb)
    h = dd[:, -1]

    # Split into high and low res groups
    def scatter_particles(X, m, hval, nclone):
        noise = np.random.randn(len(X), nclone, 3) * hval[:, None, None] * smooth_fac
        return (X[:, None, :] + noise).reshape(-1, 3), np.repeat(m, nclone) / nclone

    X1, m1, h1 = pos[high_res_ind], mass[high_res_ind], h[high_res_ind]
    X2, m2, h2 = pos[~high_res_ind], mass[~high_res_ind], h[~high_res_ind]

    pos1, mass1 = scatter_particles(X1, m1, h1, high_res_fac)
    pos2, mass2 = scatter_particles(X2, m2, h2, res_fac)

    return np.vstack([pos1, pos2]), np.concatenate([mass1, mass2])


def cylindrical_density_profile_smooth(pos, mass, centre=[0,0,0], zvec=[0,0,1],
                                Rmin=0, Rmax=0.1, Rbins=100,
                                Zmax=0.1, Zbins=200, res_fac=10,
                                smooth_fac=1, Boxsize=50):
    """
    Compute cylindrical density profile with uniform refinement.

    Particles are wrapped within a periodic box before being resampled.
    """
    p = pos - centre
    p[p > 0.5 * Boxsize] -= Boxsize
    p[p < -0.5 * Boxsize] += Boxsize

    p, mass = increase_res(p, mass, res_fac=res_fac, Nngb=32,
                           smooth_fac=smooth_fac, max_extent=Rmax + Zmax)

    return cylindrical_density_profile(p,mass,centre=[0,0,0],zvec=zvec,Rmin=Rmin,Rmax=Rmax,Rbins=Rbins,Zmax=Zmax,Zbins=Zbins)



def cylindrical_density_profile_adaptive(pos, mass, centre=[0,0,0], zvec=[0,0,1],
                                         Rmin=0, Rmax=0.1, Rbins=100,
                                         Zmax=0.1, Zbins=200, res_fac=10,
                                         smooth_fac=1, Boxsize=50,
                                         refine_Zmin=0.3, refine_Rmax=0.1,
                                         refine_Fac=10):
    """
    Compute cylindrical density profile with adaptive refinement.

    Particles farther from center (along z) and close to R=0 are refined more.
    """
    p = pos - centre
    p[p > 0.5 * Boxsize] -= Boxsize
    p[p < -0.5 * Boxsize] += Boxsize

    zhat = np.asarray(zvec) / np.linalg.norm(zvec)
    z = np.dot(p, zhat)
    r = np.linalg.norm(p, axis=1)
    R = np.sqrt(r**2 - z**2)

    refine = (np.abs(z) > refine_Zmin * Zmax) & (R < refine_Rmax * Rmax)

    p, mass = increase_res_adaptive(p, mass, high_res_ind=refine,
                                     res_fac=res_fac,
                                     high_res_fac=res_fac * refine_Fac,
                                     Nngb=32,
                                     smooth_fac=smooth_fac,
                                     max_extent=Rmax + Zmax)
    return cylindrical_density_profile(p,mass,centre=[0,0,0],zvec=zvec,Rmin=Rmin,Rmax=Rmax,Rbins=Rbins,Zmax=Zmax,Zbins=Zbins)
