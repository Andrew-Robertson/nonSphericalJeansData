import simData.read_eagle_files_mpi as read_eagle
import simData.eagleSims as eagle
import os
import numpy as np


def M200_and_CoP(path,snap,Nfof=None):
    group = read_eagle.read_eagle_file(path,'group',snap,gadgetunits=True)
    subfind = read_eagle.read_eagle_file(path,'sub',snap,gadgetunits=True)
    GROUPm200 = subfind.read_data_array('FOF/Group_M_Crit200',gadgetunits=True)
    GROUPpos = subfind.read_data_array('FOF/GroupCentreOfPotential',gadgetunits=True)
    if Nfof is None:
        return GROUPm200, GROUPpos
    else:
        return GROUPm200[:Nfof], GROUPpos[:Nfof]

def one_way_match(M1,p1,M2,p2,Mdist=0.1,pdist=1, boxsize=eagle.eagleBoxsize):
    # distance in M in log-space
    Matching_indices = np.zeros(len(M1))
    for ind, M in enumerate(M1):
        p = p1[ind]
        sep = p-p2
        # wrap things due to the periodic box
        sep[sep > 0.5 * boxsize] -= boxsize
        sep[sep < -0.5 * boxsize] += boxsize
        d2 = np.sum((sep)**2,axis=1) / pdist**2 + np.log10(M/M2)**2 / Mdist**2
        Matching_indices[ind] = np.argmin(d2)
    return Matching_indices.astype('int')

def bijective_match(model1,model2,snap=28,Mdist=0.1,pdist=1,Nfof=1000):
    # models should be "CDM", "SIDM1b", etc.
    path1 = os.path.join(eagle.basePath, eagle.simDict[model1])
    path2 = os.path.join(eagle.basePath, eagle.simDict[model2])
    M1, P1 = M200_and_CoP(path1,snap,Nfof=Nfof)
    M2, P2 = M200_and_CoP(path2,snap,Nfof=Nfof)
    match1 = one_way_match(M1,P1,M2,P2,Mdist=Mdist,pdist=pdist)
    match2 = one_way_match(M2,P2,M1,P1,Mdist=Mdist,pdist=pdist)
    GoodMatch1 = (match1[match2]-np.arange(Nfof)==0)
    GrNm1 = np.arange(Nfof)[GoodMatch1]+1
    GrNm2 = match1[GoodMatch1]+1
    DeltaLogM = np.log10(M1[GrNm1-1]/M2[GrNm2-1])
    DeltaP = np.sqrt(np.sum((P1[GrNm1-1]-P2[GrNm2-1])**2,axis=1))
    return GrNm1, GrNm2, DeltaLogM, DeltaP, path1, path2

