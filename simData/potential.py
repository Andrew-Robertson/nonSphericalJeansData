import numpy as np
import astropy.constants as const
import astropy.units as u
from scipy.special import ellipk

def pot_from_hoop(M,l,R,z,eps=0.5*u.kpc):
    # taken from https://iopscience.iop.org/article/10.1088/0143-0807/30/3/019
    # softened slightly (with softening length epsilon), so that self-potential is not infinite
    return (2*const.G*M*ellipk((4*R*l / ((R+l)**2 + z**2 + eps**2)).to_value('')) / (np.pi * np.sqrt((R+l)**2 + z**2))).to('km2 s-2')

def cylindrical_pot_from_dens(rho,Redges,Zedges,eps=0.5*u.kpc,rmax=None):
    Rmids = 0.5*(Redges[1:]+Redges[:-1])*u.Mpc
    Zmids = 0.5*(Zedges[1:]+Zedges[:-1])*u.Mpc
    bin_vol = np.outer(np.pi*(Redges[1:]**2-Redges[:-1]**2),Zedges[1:]-Zedges[:-1]) # Mpc**3
    mass = bin_vol * rho * u.Msun
    Rgrid, Zgrid = np.meshgrid(Rmids,Zmids,indexing='ij')
    rgrid = np.sqrt(Rgrid**2 + Zgrid**2)
    if rmax != None:
        mass[rgrid>rmax*u.Mpc]=0
    potential =pot_from_hoop(0*u.Msun,0*u.Mpc,Rgrid,Zgrid,eps) # initialise potential grid (from 0 mass) 
    for i in np.arange(len(Rmids)):
        for j in np.arange(len(Zmids)):
            potential += pot_from_hoop(mass[i,j],Rmids[i],Rgrid,Zgrid-Zmids[j],eps)
            if np.sum(np.isnan(potential))>0:
                print("Problem with [i,j] =", i, j)
    return potential