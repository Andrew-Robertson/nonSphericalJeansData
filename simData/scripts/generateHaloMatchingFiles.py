import numpy as np
import os
import simData.eagleSims as eagle
import sys
import simData.haloMatching as matching

model1 = sys.argv[1] # e.g. "CDMb", "vdSIDMb", etc.
model2 = sys.argv[2] # e.g. "CDMb", "vdSIDMb", etc.
maxDeltaLogM = 0.1 # masses within 25%
maxSeparation = 0.05 # positions within 50 kpc
Mdist=0.1 # 0.1 dex in halo mass, is as different as...
pdist=1 # 1 cMpc/h in position
Nfof=1000 # number of FOF halos to consider from each simulation

output_dir = './haloMatchingData'
os.makedirs(output_dir, exist_ok=True)

GrNm1,GrNm2,DeltaLogM,separation, path1, path2 = matching.bijective_match('CDMb','SIDM1b', Mdist=Mdist, pdist=pdist, Nfof=Nfof)
keep = (DeltaLogM < maxDeltaLogM) * (separation < maxSeparation)
GalaxyID = np.arange(np.sum(keep))
fname = os.path.join(output_dir, f"EAGLE_{model1}_{model2}_matching_Mdist{Mdist}.npz")
np.savez(fname, path1=path1, path2=path2, GalaxyID=GalaxyID, GrNm1=GrNm1[keep], GrNm2=GrNm2[keep], DeltaLogM=DeltaLogM[keep], separation=separation[keep])