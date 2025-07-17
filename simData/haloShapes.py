import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy import stats
from scipy.optimize import curve_fit
import simData.read_eagle_files_mpi as read_eagle
from numpy.linalg import LinAlgError


#########################################################
def sep(pos1,pos2):
    # input pos in Mpc, sep in kpc
    return 1e3* np.sqrt(np.sum((pos1-pos2)**2))



def com(part_pos, part_mass):
    mtotal = np.sum(part_mass) 
    xcoord = np.sum(part_mass * part_pos[:,0])/mtotal
    ycoord = np.sum(part_mass * part_pos[:,1])/mtotal
    zcoord = np.sum(part_mass * part_pos[:,2])/mtotal
    return np.array([xcoord,ycoord,zcoord])


def shrinking_sphere(part_pos, part_mass,start_centre,start_rad,Nfin=200,shrinkage = 0.95,verbose=True):
    com_pos = start_centre
    r = start_rad * 2

    while True:
        rs = np.sqrt(np.sum((part_pos-com_pos)**2,axis=1))
        part_pos = part_pos[rs<r]
        part_mass = part_mass[rs<r]
        if len(part_pos) < Nfin:
            break
        com_pos = com(part_pos,part_mass)
        if verbose:
            print('particles: %d, radius: %.5f, centre:' % (len(part_pos), r), com_pos)
        r = r*shrinkage
    
    return com_pos



def ellipsoid(pos, rn=None, mass=None, centre=None, print_outputs=False, plot_graph=False,reduced=False):
   """Fit ellipsoid to 3D distribution of points and return eigenvectors
       and eigenvalues of the result.

   Args:
      pos          :  3xN array of positionsfield (in Mpc). If centre is not specified
                      then a shrinking circles procedure will be run on the particles
                      to find the halo centre.
      centre       : 

      print outputs : Bool True/False. Optional flag to print outputs.
                       Default: False.
      plot_graph    : Bool True/False. Optional flag to plot data in
                       eigenvector basis. Default: False.

   Returns:
      Tuple containing the eigenvectors and eigenvalues of the ellipsoid.
   """

   if mass is None:
       mass = np.ones(pos.shape[0], np.float64)

   if centre is None:
       # shrinking spheres starting with 1 Mpc radius
       centre=shrinking_sphere(pos,mass,com(pos,mass),start_rad=1.0,shrinkage=0.9) 

   pos = pos - centre

   # Compute a weight for each point
   w = mass
   if reduced:
    print("Computing the reduced shape ellipsoid.")
    if rn is not None:
        w = mass * rn**-2
    else:
        w = mass * (pos[:,:] * pos[:,:]).sum( axis=1 )**-1

   # Compute Moment of Intertia
   shape = np.empty((3,3), np.float64)
   for i in range(0, 3):
      shape[i] = [(w * pos[:,i] * pos[:,0]).sum(),
                   (w * pos[:,i] * pos[:,1]).sum(),
                   (w * pos[:,i] * pos[:,2]).sum()]
   shape /= (pos.shape[0] * np.mean(mass))

   # Obtain eigenvalues and eigenvectors
   eigenValues, eigenVectors = np.linalg.eig(shape)

   # Sort eigenvalues in increasing order
   order = eigenValues.argsort()
   eigenValues.sort()
   eigenVectors = eigenVectors[:,order]

   # Get the shape
   c, b, a = eigenValues ** 0.5
   if print_outputs:
      print("Eigenvectors")
      print("v_a = {}, v_b = {}, v_c = {}".format(eigenVectors[:,2], eigenVectors[:,1], eigenVectors[:,0]))
      print("Eigenvalues")
      print("a = {}, b = {}, c = {}".format(a, b, c))

   if plot_graph:
      # get the coordinates of the points in this new coordinate system, i.e. the one in which x,y and z are the principal axes of the inertia tensor
      mat = eigenVectors.transpose()
      res = np.zeros( (pos.shape[0],3+3+2), np.float64 )
      res[:,:3] = pos
      for i in range( pos.shape[0] ):     # the coordinates of the point in the new reference system
          res[i,5:2:-1] = np.linalg.solve( mat, pos[i,:] )
      res[:,6] = (pos[:,:] * pos[:,:]).sum( axis=1 )**0.5     # the distance of the point from the center
      res[:,7] = np.abs( res[:,5] )                           # the distance of the point from the plane

      # plot the points in a coordinate system where x,y and z are the principal axes of the inertia tensor
      indexList = [ [3,4], [3,5], [4,5] ]
      title     = [ "v_a, v_b", "v_a, v_c", "v_b, v_c",  ]
      maxDis    = res[:,6].max()              # maximum distance from center, used as min-max range for all the plots
      for i in range(3):
          plt.figure()
          plt.xlim( [-maxDis,maxDis] )
          plt.ylim( [-maxDis,maxDis] )
          plt.plot( res[:,indexList[i][0]], res[:,indexList[i][1]], 'o', color="k" )
          plt.title( "Projection on eigenvector plane {0}".format( title[i] ) )

   return (eigenVectors, eigenValues)


def ellipsoid_iterative(pos, eff_radius, mass=None, centre=None, print_outputs=False, plot_graph=False,tollerance=5e-3,max_attempts=None,shell=False,shell_fac=0.8):
    """Fit ellipsoid to 3D distribution of points and return eigenvectors
       and eigenvalues of the result.

   Args:
      pos          :  3xN array of positionsfield (in Mpc). If centre is not specified
                      then a shrinking circles procedure will be run on the particles
                      to find the halo centre.
      centre       : 

      print outputs : Bool True/False. Optional flag to print outputs.
                       Default: False.
      plot_graph    : Bool True/False. Optional flag to plot data in
                       eigenvector basis. Default: False.

   Returns:
      Tuple containing the eigenvectors and eigenvalues of the ellipsoid.
   """

    a = eff_radius
    # start with sphere along coordinate axes
    b,c = a,a
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])


    if mass is None:
        mass = np.ones(pos.shape[0], np.float64)

    if centre is None:
        # shrinking spheres starting with 1 Mpc radius
        centre=shrinking_sphere(pos,mass,com(pos,mass),start_rad=1.0,shrinkage=0.9)

    pos = pos - centre

    toll = 1

    if max_attempts is not None:
        ii = 0
        flag = 0

    while toll > tollerance:
        q,s = b/a, c/a
        if print_outputs:
            print("a = ",a,", b = ",b,", c = ",c)
        xn,yn,zn = np.sum(pos*e1,axis=1),np.sum(pos*e2,axis=1),np.sum(pos*e3,axis=1)
        rn = np.sqrt(xn**2 + (yn/q)**2 + (zn/s)**2)
        if shell:
            inEllipse = (rn<a)*(rn>a*shell_fac)
        else:
            inEllipse = (rn<a)
        if mass is not None:
            massEllipse = mass[inEllipse]
        else:
            massEllipse = None
        eigenVectors, eigenValues = ellipsoid(pos[inEllipse], rn=rn[inEllipse], mass=massEllipse, centre=[0,0,0], print_outputs=False, plot_graph=plot_graph,reduced=True)
        c2,b2,a2 = eigenValues**0.5
        e3,e2,e1 = eigenVectors.T
        # re-scale ellipse axes to keep 'a' fixed
        q2,s2 = b2/a2, c2/a2
        toll = (q2-q)**2/q**2 + (s2-s)**2/s**2
        if print_outputs:
            print("(q2-q)^2/q^2+(s2-s)^2/s^2 = ", toll)
        scale = eff_radius/((a2*b2*c2)**(1./3))
        c,b,a = c2*scale,b2*scale,a2*scale
        if max_attempts is not None:
            ii += 1
            if ii > max_attempts:
                flag = 1
                break

    q,s = b/a, c/a

    xn,yn,zn = np.sum(pos*e1,axis=1),np.sum(pos*e2,axis=1),np.sum(pos*e3,axis=1)
    rn = np.sqrt(xn**2 + (yn/q)**2 + (zn/s)**2)
    r = np.sqrt(xn**2 + yn**2 + zn**2)
    if shell:
        Nr = sum((rn<a)*(rn>a*shell_fac))
    else:
        Nr = sum(rn<a)

    if print_outputs:
        print("q = ",q,", s = ",s)

    if max_attempts is not None:
        return eigenVectors, eigenValues, flag, Nr
    else:
        return eigenVectors, eigenValues, Nr
       

def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)

def read_FOF_M200(simpath,snap):
    filepath = simpath+'/data/M200s/'
    filename = filepath+'FOF_M200_'+str(snap).zfill(3)+'.npy'
    try:
        data = np.load(filename)
    except IOError:
        subfind = read_eagle.read_eagle_file(simpath,'sub',snap,gadgetunits=True)
        GROUPm200 = subfind.read_data_array('FOF/Group_M_Crit200',gadgetunits=True)*1e10/subfind.h
        ensure_dir(filepath)
        np.save(filename,GROUPm200)
        data = GROUPm200
    return data


def shape_with_radius(pos,rmaxs,centre,mass=None,print_outputs=True,tollerance=0.001,max_attempts=100,shell=False):
    a,b,c = np.zeros((3,len(rmaxs)))
    e1,e2,e3 = np.zeros((3,len(rmaxs),3))
    flags = np.zeros((len(rmaxs),))
    Nrs = np.zeros((len(rmaxs),))

    i=0
    for i in np.arange(len(rmaxs)):
        rmax = rmaxs[i]
        try:
            eigenVectors, eigenValues, flag, Nr = ellipsoid_iterative(pos,eff_radius=rmax,mass=mass,centre=centre,print_outputs=print_outputs,tollerance=tollerance,max_attempts=max_attempts,shell=shell)
        except LinAlgError:
            eigenVectors = np.array([[np.nan,0,0],[0,np.nan,0],[0,0,np.nan]])
            eigenValues = np.array([np.nan,np.nan,np.nan])
            flag = 2
            Nr = np.nan
        a[i] = eigenValues[2]**0.5
        b[i] = eigenValues[1]**0.5
        c[i] = eigenValues[0]**0.5
        e1[i] = eigenVectors[:,2]
        e2[i] = eigenVectors[:,1]
        e3[i] = eigenVectors[:,0]
        flags[i] = flag
        Nrs[i] = Nr
    return a,b,c,e1,e2,e3,flags,Nrs
    





def shape2D(pos, rn=None, mass=None, centre=None, print_outputs=False,reduced=False):
    """Fit ellipsoid to 3D distribution of points and return eigenvectors
    and eigenvalues of the result.

    Args:
    pos          :  Nx2 array of positions. If centre is not specified
                      then it is assumed to be [0,0]
    centre       : [x,y] position of the centre

    print outputs : Bool True/False. Optional flag to print outputs.
                       Default: False.
    
    Returns:
    Tuple containing the eigenvectors and eigenvalues of the ellipse.
    """

    if mass is None:
        mass = np.ones(pos.shape[0], np.float64)

    if centre is None:
       # shrinking spheres starting with 1 Mpc radius
       centre=[0,0] 

    pos = pos - centre

    # Compute a weight for each point
    if reduced:
        print("Computing the reduced shape ellipsoid.")
        if rn is not None:
            w = mass * rn**-2
        else:
            w = mass * (pos[:,:] * pos[:,:]).sum( axis=1 )**-1
    else:
        w = mass

    # Compute Moment of Intertia
    shape = np.empty((2,2), np.float64)
    for i in range(0, 2):
        shape[i] = [(w * pos[:,i] * pos[:,0]).sum(),
                   (w * pos[:,i] * pos[:,1]).sum()]
    shape /= (pos.shape[0] * np.mean(mass))

    # Obtain eigenvalues and eigenvectors
    eigenValues, eigenVectors = np.linalg.eig(shape)

    # Sort eigenvalues in increasing order
    order = eigenValues.argsort()
    eigenValues.sort()
    eigenVectors = eigenVectors[:,order]

    # Get the shape
    b, a = eigenValues ** 0.5
    if print_outputs:
        print("Eigenvectors")
        print("v_a = {}, v_b = {}".format(eigenVectors[:,1], eigenVectors[:,0]))
        print("Eigenvalues")
        print("a = {}, b = {}".format(a, b))

    return (eigenVectors, eigenValues)


def shape2D_iterative(pos, eff_radius, mass=None, centre=None, print_outputs=False,tollerance=1e-4,max_attempts=None,shell=False,shell_fac=0.9,reduced=False):
    """Fit ellipsoid to 3D distribution of points and return eigenvectors
    and eigenvalues of the result.

    Args:
    pos          :  3xN array of positionsfield (in Mpc). If centre is not specified
                      then a shrinking circles procedure will be run on the particles
                      to find the halo centre.
    centre       : 

    print outputs : Bool True/False. Optional flag to print outputs.
                       Default: False.
    plot_graph    : Bool True/False. Optional flag to plot data in
                       eigenvector basis. Default: False.

    Returns:
    Tuple containing the eigenvectors and eigenvalues of the ellipsoid.
    """

    a = eff_radius
    # start with circle along coordinate axes
    b = a
    e1 = np.array([1,0])
    e2 = np.array([0,1])


    if mass is None:
        mass = np.ones(pos.shape[0], np.float64)
        
    if centre is None:
        centre=np.array([0,0])

    pos = pos - centre

    toll = 1

    if max_attempts is not None:
        ii = 0
        flag = 0

    while toll > tollerance:
        axRat = b/a
        if print_outputs:
            print("a = ",a,", b = ",b)
        xn,yn = np.sum(pos*e1,axis=1),np.sum(pos*e2,axis=1)
        rn = np.sqrt(xn**2 + yn**2 / axRat**2)
        if shell:
            inEllipse = (rn<a/np.sqrt(shell_fac))*(rn>a*np.sqrt(shell_fac))
        else:
            inEllipse = (rn<a)
        if mass is not None:
            massEllipse = mass[inEllipse]
        else:
            massEllipse = None
        eigenVectors, eigenValues = shape2D(pos[inEllipse], rn=rn[inEllipse], mass=massEllipse, centre=[0,0], print_outputs=False, reduced=reduced)
        b2,a2 = eigenValues**0.5
        e2,e1 = eigenVectors.T

        axRat2 = b2/a2
        toll = (axRat2-axRat)**2/axRat**2
        if print_outputs:
            print("(axRat2-axRat)^2/axRat^2 = ", toll)
        # re-scale ellipse to keep enclosed area fixed
        scale = eff_radius/((a2*b2)**(1./2))
        b,a = b2*scale,a2*scale
        if max_attempts is not None:
            ii += 1
            if ii > max_attempts:
                flag = 1
                break

    axRat = b/a

    xn,yn = np.sum(pos*e1,axis=1),np.sum(pos*e2,axis=1)
    rn = np.sqrt(xn**2 + yn**2 / axRat**2)
    r = np.sqrt(xn**2 + yn**2)
    if shell:
        Nr = np.sum((rn<a/shell_fac)*(rn>a*shell_fac))
    else:
        Nr = np.sum(rn<a)

    if print_outputs:
        print("axRat = ",axRat)

    if max_attempts is not None:
        return eigenVectors, eigenValues, flag, Nr
    else:
        return eigenVectors, eigenValues, Nr
       


def shape2D_with_radius(pos,rmaxs,centre,mass=None,print_outputs=True,tollerance=0.0001,max_attempts=100,shell=False,reduced=False,shell_fac=0.9):
    a,b = np.zeros((2,len(rmaxs)))
    e1,e2 = np.zeros((2,len(rmaxs),2))
    flags = np.zeros((len(rmaxs),))
    Nrs = np.zeros((len(rmaxs),))

    i=0
    for i in np.arange(len(rmaxs)):
        rmax = rmaxs[i]
        try:
            eigenVectors, eigenValues, flag, Nr = shape2D_iterative(pos,eff_radius=rmax,mass=mass,centre=centre,print_outputs=print_outputs,tollerance=tollerance,max_attempts=max_attempts,shell=shell,reduced=reduced,shell_fac=shell_fac)
        except LinAlgError:
            eigenVectors = np.array([[np.nan,0],[0,np.nan]])
            eigenValues = np.array([np.nan,np.nan])
            flag = 2
            Nr = np.nan
        a[i] = eigenValues[1]**0.5
        b[i] = eigenValues[0]**0.5
        e1[i] = eigenVectors[:,1]
        e2[i] = eigenVectors[:,0]
        flags[i] = flag
        Nrs[i] = Nr
    return a,b,e1,e2,flags,Nrs



def axisymmetric_shape_with_radius(pos,rmaxs,centre=[0,0,0],zvec=[0,0,1],mass=None,print_outputs=True,tollerance=0.0001,max_attempts=100,shell=False,reduced=False,Boxsize=50,shell_fac=0.9):
    p = pos-centre
    # wrap particles that are over the box edges
    # note that Boxsize needs to be in the same units as pos
    while (p > .5 * Boxsize).any():
        p[p > .5 * Boxsize] -= Boxsize
    while (p < -.5 * Boxsize).any():
        p[p < -.5 * Boxsize] += Boxsize
    # measure the halo shape with assumed azimuthal symmetry (about a z axis specified by zvec)
    r = np.sqrt(np.sum(p**2,axis=1))
    zhat = np.asarray(zvec)/np.linalg.norm(zvec)
    z = np.sum(zhat*p,axis=1)
    R = np.sqrt(r**2 - z**2)
    if mass is None:
        mass = np.ones_like(r)
    weights = mass / R # to account for volume effects ("pixel size" is 2*pi*R*dR*dz) 
    # extend to -ve R
    R = np.concatenate((-R,R),axis=0)
    z = np.concatenate((z,z),axis=0)
    weights = np.concatenate((weights,weights),axis=0)
    # make into 2D coordinates
    pos2D = np.vstack((R.flatten(),z.flatten())).T
    # Let's set up some result arrays
    a,b,q = np.zeros((3,len(rmaxs)))
    e1,e2 = np.zeros((2,len(rmaxs),2))
    flags = np.zeros((len(rmaxs),))
    Nrs = np.zeros((len(rmaxs),))
    # And now let's measure some shapes :-)
    for i, rmax in enumerate(rmaxs):
        try:
            eigenVectors, eigenValues, flag, Nr = shape2D_iterative(pos2D,eff_radius=rmax,mass=weights,centre=np.array([0,0]),print_outputs=print_outputs,tollerance=tollerance,max_attempts=max_attempts,shell=shell,reduced=reduced,shell_fac=shell_fac) # Note that centre is already accounted for in pos2D
        except LinAlgError:
            eigenVectors = np.array([[np.nan,0],[0,np.nan]])
            eigenValues = np.array([np.nan,np.nan])
            flag = 2
            Nr = np.nan
        a[i] = eigenValues[1]**0.5
        b[i] = eigenValues[0]**0.5
        e1[i] = eigenVectors[:,1]
        e2[i] = eigenVectors[:,0]
        flags[i] = flag
        Nrs[i] = Nr
        q = b/a
        Q = np.copy(q)
        where_prolate = np.isclose(np.abs(e1[:,1]),1)
        Q[where_prolate] = 1/q[where_prolate]

    return Q,q,e1,e2,a,b,flags,Nrs


############################################
# Bespoke way of doing shapes with axisymmetry (rather than re-weighting a projected-density 2D shape code)
############################################

def shape_axisymmetric(pos, weights=None):
    # calculate q=sqrt(M_zz/M_RR), where the z-axis is indexed by "2". this corresponds to the shape of a halo assuming azimuthal symmetry about the z-axis. Note that the assumption is that the pos array is centred on the origin.
    if weights is None:
        weights = np.ones(len(pos))
    R2 = np.sum(pos[:,:2]**2,axis=1)
    z2 = pos[:,2]**2
    M_RR = 0.5*np.sum(weights*R2)
    M_zz = np.sum(weights*z2)
    q = np.sqrt(M_zz/M_RR)
    return q
    

def shape_axisymmetric_iterative(pos, eff_radius, mass=None, centre=None, print_outputs=False, tollerance=1e-4, shell=True, shell_fac=0.9, max_attempts=100):
    if mass is None:
        mass = np.ones(len(pos))
    if centre is None:
        centre = np.array([0,0,0])
    pos = pos - centre
    R = np.sqrt(np.sum(pos[:,:2]**2,axis=1))
    z = np.sqrt(pos[:,2]**2)
    toll = 1
    attempts = 0
    # begin using a circular aperture
    q = 1
    while toll>tollerance:
        a = eff_radius * q**(-1./3) # stretch effective radius in R-direction (for q<1)
        b = eff_radius * q**(2./3) # squash effective radius in z-direction (for q<1)
        if print_outputs:
            print("a = ",a,", b = ",b)
        rn = np.sqrt((R/a)**2 + (z/b)**2) # dimensionless radius, ~1 on surface
        if shell:
            inc = (rn<1/shell_fac)*(rn>shell_fac)
        else:
            inc = (rn<1)
        axRat = shape_axisymmetric(pos[inc], weights=mass[inc])
        toll = (axRat-q)**2/q**2
        attempts += 1
        q = axRat
        N = np.sum(inc)
        if max_attempts is not None:
            if attempts > max_attempts:
                break
    return q, N

def rotate_to_z_axis(pos, zvec):
    """
    Rotate 3D positions so that the z-axis aligns with `zvec`.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Input positions.
    zvec : array-like, shape (3,)
        Vector defining the new z-axis direction.

    Returns
    -------
    pos_rot : ndarray, shape (N, 3)
        Rotated positions such that column 2 is distance along zvec.
    """
    zvec = np.asarray(zvec)
    zhat = zvec / np.linalg.norm(zvec)  # normalize

    # Choose arbitrary vector not parallel to zhat
    if np.allclose(zhat, [0, 0, 1]):
        x_temp = np.array([1, 0, 0])
    else:
        x_temp = np.array([0, 0, 1])

    # Compute new x and y axes (orthonormal to zhat)
    xhat = np.cross(x_temp, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)

    # Rotation matrix: rows are new basis vectors
    R = np.vstack([xhat, yhat, zhat])  # shape (3, 3)

    # Rotate all positions into this frame
    pos_rot = pos @ R.T

    return pos_rot

def shape_axisymmetric_with_radius(pos, rs, centre=[0,0,0],zvec=[0,0,1], mass=None,  print_outputs=False, tollerance=1e-4, shell=True, shell_fac=0.9, max_attempts=100, Boxsize=50):
    p = pos - centre
    # wrap particles that are over the box edges
    # note that Boxsize needs to be in the same units as pos
    while (p > .5 * Boxsize).any():
        p[p > .5 * Boxsize] -= Boxsize
    while (p < -.5 * Boxsize).any():
        p[p < -.5 * Boxsize] += Boxsize
    # rotate pos to have z-axis along zvec
    p = rotate_to_z_axis(p, zvec)
    # Let's set up some result arrays
    q, N = np.zeros((2,len(rs)))
    # for each radius, find the shape
    for i, eff_radius in enumerate(rs):
        try:
            qi, Ni = shape_axisymmetric_iterative(p, eff_radius, mass=mass, centre=None, print_outputs=print_outputs, tollerance=tollerance, shell=shell, shell_fac=shell_fac, max_attempts=max_attempts)
        except LinAlgError:
            qi=np.nan; Ni=np.nan
        q[i]=qi; N[i]=Ni
    return q, N