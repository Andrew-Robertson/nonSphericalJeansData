
import simData.read_eagle_files_mpi as read_eagle
import simData.density as density
import simData.potential as potential
import os
import numpy as np
import astropy.units as u

basePath = "/cosma7/data/Eagle/EAGLE_SIDM/L0050N0752"
eagleBoxsize = 50 # cMpc/h

simDict = {
     'CDMb':  'CDM+baryons',
     'CDM': 'CDM',
     'SIDM1b':  'SIDM1+baryons',
     'SIDM1': 'SIDM1',
     'vdSIDMb':  'vdSIDM_3.04_w560+baryons',
     'vdSIDM': 'vdSIDM_3.04_w560',
}


class eagleSim():
     
    def __init__(self, model="CDMb", snapNum=28, GroupNumbers=np.arange(250)+1):
          self.model = model
          self.simPath = os.path.join(basePath, simDict[self.model])
          self.snapNum = snapNum
          self.GroupNumbers = GroupNumbers

    def load_data(self,):
        self.group = read_eagle.read_eagle_file(self.simPath, 'group', self.snapNum,gadgetunits=True)
        self.subfind = read_eagle.read_eagle_file(self.simPath, 'sub', self.snapNum,gadgetunits=True)
        self.particles = read_eagle.read_eagle_file(self.simPath, 'particles', self.snapNum,gadgetunits=True)
        self.a, self.h = self.group.a, self.group.h

    def read_group_data(self,):
        self.GROUPm200 = self.subfind.read_data_array('FOF/Group_M_Crit200',gadgetunits=True)
        self.GROUPpos = self.subfind.read_data_array('FOF/GroupCentreOfPotential',gadgetunits=True)
        self.GROUPr200 = self.subfind.read_data_array('FOF/Group_R_Crit200',gadgetunits=True)
        self.GROUPsub0id = self.subfind.read_data_array('FOF/FirstSubhaloID').astype('int')
        self.GROUPstarL = self.subfind.read_data_array('Subhalo/Stars/Spin',gadgetunits=True)[self.GROUPsub0id]

    def read_particle_data(self,):
        # Read in DM data for the relevant FOF groups
        self.DM_group_no = np.abs(self.particles.read_data_array('PartType1/GroupNumber',gadgetunits=True)).astype('int')      
        self.DM_mass = 1e10 * self.particles.DM_Particle_Mass / self.h # msun 
        self.DM_pos = self.particles.read_data_array('PartType1/Coordinates',gadgetunits=True)[self.DM_group_no <= self.GroupNumbers.max()] # h-inv cMpc 
        self.DM_group_no = self.DM_group_no[self.DM_group_no <= self.GroupNumbers.max()]
        if "baryons" in simDict[self.model]:
            # Read all the GAS particle masses and positions
            self.GAS_group_no = np.abs(self.particles.read_data_array('PartType0/GroupNumber',gadgetunits=True)).astype('int')
            self.GAS_pos = self.particles.read_data_array('PartType0/Coordinates',gadgetunits=True)[self.GAS_group_no <= self.GroupNumbers.max()]      
            self.GAS_mass = 1e10 * self.particles.read_data_array('PartType0/Mass',gadgetunits=True)[self.GAS_group_no <= self.GroupNumbers.max()]  / self.h
            self.GAS_group_no = self.GAS_group_no[self.GAS_group_no <= self.GroupNumbers.max()] 
            # Read all the STAR particle masses and positions
            self.STAR_group_no = np.abs(self.particles.read_data_array('PartType4/GroupNumber',gadgetunits=True)).astype('int')
            self.STAR_pos = self.particles.read_data_array('PartType4/Coordinates',gadgetunits=True)[self.STAR_group_no <= self.GroupNumbers.max()]       
            self.STAR_mass = 1e10 * self.particles.read_data_array('PartType4/Mass',gadgetunits=True)[self.STAR_group_no <= self.GroupNumbers.max()]  / self.h
            self.STAR_group_no = self.STAR_group_no[self.STAR_group_no <= self.GroupNumbers.max()] 
            # Read all the BH particle masses and positions
            self.BH_group_no = np.abs(self.particles.read_data_array('PartType5/GroupNumber',gadgetunits=True)).astype('int')
            self.BH_pos = self.particles.read_data_array('PartType5/Coordinates',gadgetunits=True)[self.BH_group_no <= self.GroupNumbers.max()]       
            self.BH_mass = 1e10 * self.particles.read_data_array('PartType5/Mass',gadgetunits=True)[self.BH_group_no <= self.GroupNumbers.max()]  / self.h
            self.BH_group_no = self.BH_group_no[self.BH_group_no <= self.GroupNumbers.max()] 

    def cylindrical_density_profile(self, pos, mass, centre, zvec):
        # pos in physical Mpc
        return density.cylindrical_density_profile(pos, mass, centre=centre, zvec=zvec, Rmin=0, Rmax=0.1, Rbins=100, Zmax=0.1, Zbins=200, Boxsize=eagleBoxsize*self.a/self.h)

    def calculate_properties_for_GrNm(self, GrNm, includeSmoothDarkMatterDensity=True, res_fac=300, smooth_fac=1, refine_Fac=30):
        M200 = self.GROUPm200[GrNm-1]*1e10/self.h # Msun
        R200 = self.GROUPr200[GrNm-1]* self.a /self. h
        CoP = self.GROUPpos[GrNm-1]*self.a/self.h
        L = self.GROUPstarL[GrNm-1] # angular momentum vector of stars
        # get data for the GroupNumber
        dm_pos = self.DM_pos[self.DM_group_no==GrNm]*self.a/self.h
        dm_mass = np.ones(len(dm_pos))*self.DM_mass
        if "baryons" in simDict[self.model]:
            gas_pos = self.GAS_pos[self.GAS_group_no==GrNm]*self.a/self.h
            gas_mass = self.GAS_mass[self.GAS_group_no==GrNm]
            star_pos = self.STAR_pos[self.STAR_group_no==GrNm]*self.a/self.h
            star_mass = self.STAR_mass[self.STAR_group_no==GrNm]
            bh_pos = self.BH_pos[self.BH_group_no==GrNm]*self.a/self.h
            bh_mass = self.BH_mass[self.BH_group_no==GrNm]
        # make dictionary of quantitites for this halo, we will add the density, potential, etc. to this
        properties = {'M200':M200, 'r200':R200, 'L':L, 'pos':self.GROUPpos[GrNm-1]}
        # cylindrical density profiles
        properties['dm_rho'], properties['Redges'], properties['Zedges'] = self.cylindrical_density_profile(dm_pos,dm_mass,centre=CoP,zvec=L)
        if "baryons" in simDict[self.model]:
            properties['star_rho'], Redges, Zedges = self.cylindrical_density_profile(star_pos,star_mass,centre=CoP,zvec=L)
            properties['gas_rho'], Redges, Zedges = self.cylindrical_density_profile(gas_pos,gas_mass,centre=CoP,zvec=L)
            properties['bh_rho'], Redges, Zedges = self.cylindrical_density_profile(bh_pos,bh_mass,centre=CoP,zvec=L)
        # cylindrical potentials
        # they (Sean and Adam) appear to be using 0.5 kpc softening
        properties['dm_potential_sphere'] = potential.cylindrical_pot_from_dens(properties['dm_rho'], properties['Redges'], properties['Zedges'], eps=0.5*u.kpc, rmax=properties['Redges'].max())
        if "baryons" in simDict[self.model]:
            properties['star_potential_sphere'] = potential.cylindrical_pot_from_dens(properties['star_rho'], properties['Redges'], properties['Zedges'], eps=0.5*u.kpc, rmax=properties['Redges'].max())
            properties['gas_potential_sphere'] = potential.cylindrical_pot_from_dens(properties['gas_rho'], properties['Redges'], properties['Zedges'], eps=0.5*u.kpc, rmax=properties['Redges'].max())
            properties['bh_potential_sphere'] = potential.cylindrical_pot_from_dens(properties['bh_rho'], properties['Redges'], properties['Zedges'], eps=0.5*u.kpc, rmax=properties['Redges'].max())
            properties['baryon_potential_sphere'] = properties['star_potential_sphere'] + properties['gas_potential_sphere'] + properties['bh_potential_sphere']
            properties['total_potential_sphere'] = properties['baryon_potential_sphere'] + properties['dm_potential_sphere']
        # smooth DM density (smooth particles out by splitting them into multiple particles and scattering these within some kernel centred on the original particle position)
        if includeSmoothDarkMatterDensity:  
            properties['smooth_dm_rho'], Redges, Zedges = density.cylindrical_density_profile_adaptive(dm_pos,dm_mass,centre=CoP,zvec=L,
                                                Rmin=properties['Redges'].min(), Rmax=properties['Redges'].max(), Rbins=len(properties['Redges'])-1, Zmax=properties['Zedges'].max(), Zbins=len(properties['Zedges'])-1,
                                                res_fac=res_fac, smooth_fac=smooth_fac, Boxsize=eagleBoxsize*self.a/self.h, refine_Fac=refine_Fac) # res_fac=300, smooth_fac=1, Boxsize=eagleBoxsize*self.a/self.h, refine_Fac=30)
        # return info
        return properties
