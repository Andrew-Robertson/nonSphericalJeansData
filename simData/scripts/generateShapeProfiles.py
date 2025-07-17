import numpy as np
import os
import simData.eagleSims as eagle
import simData.haloShapes as shapes
import sys

model = sys.argv[1] # e.g. "CDMb", "vdSIDMb", etc.
output_base = './shapeProfiles'
output_dir = os.path.join(output_base, model)
os.makedirs(output_dir, exist_ok=True)

dd = eagle.eagleSim(model=model)
dd.load_data()
dd.read_group_data()
dd.read_particle_data()

for GrNm in np.arange(250)+1:
    print("Analysing Group Number", GrNm)
    data = dd.calculate_axisymmetric_shapes_for_GrNm(GrNm)
    np.savez(os.path.join(output_dir,model+'_'+str(GrNm).zfill(5)+'.npz'), **data)