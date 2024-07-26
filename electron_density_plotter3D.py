import numpy as np
from pyscf import gto, scf, dft, tools
from pyscf.tools import cubegen

#-----------------------------------------------------------------------------------------------
# Read molden file with PySCf.tools.molden module
# mol        --> it contains the geometry and corresponding info of the applied basis set
# mo_energy  --> energy of the molecule orbitals (MO)
# mo_coeff   --> expansion coeffcient of the wave fucntion representing the contribution of MOs
# mo_occ     --> occupancy of the molecular orbitals (mo_occ = 0 or 2)
# irrep      --> irreducibly representation in the given symmetry group of MOs...not used here
# spin       --> spin of the orbitals --> not used here
#-----------------------------------------------------------------------------------------------
# Read Molden file
#filename = "Nitro-Benzene_step_5.molden"
filename = "Ringclosing-135triene_step_0.molden"
#filename = "saved_water_test_HF_STO-3G.molden"
mol, mo_energy, mo_coeff, mo_occ, irrep , spin = tools.molden.load(filename)
nocc = np.count_nonzero(mo_occ)

#============ If you want to average either "over time" or "over trajectories" you have to average this C_occ 
#This are the MO coeffs scaled with the occupancy of the given orbital:
C_occ = mo_coeff[:,:nocc] * np.sqrt(mo_occ[:nocc])

print()
print("Occupancy:", mo_occ)
print("MO energies:", mo_energy)
#print("MO coeffs before scaling (all --> occupied + unoccupied):", mo_coeff)
#print("MO coeffs after scaling (only occupied):", C_occ)
#-----------------------------------------------------------------------------------------------

no_atoms = 14

#Density matrix:
dm = np.einsum('ij,jk', C_occ, C_occ.T)
#print("Density matrix:", dm)


#========== You have to change the file name too in order to process all trajectories ==================
#Electron density saved into a Cube file
filename1 = filename + ".cube"

# generate electron density on a grid from the density matrix and atomic basis function included in "mol"
cubegen.density(mol, filename1, dm)
print("\n Cube file is created for electron density \n")

# molecular electrostatic potential
#cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

# or just a single orbital --> here you may change the index of the mo_coeff[:index]
#cubegen.orbital(mol, filename1, mo_coeff[:,15])


# Retrieve the electron density distribution data for H2 ######################
import numpy as np
from mayavi import mlab

mlab.figure(1, bgcolor=(1.0, 1.0, 1.0), size=(350, 350))
#bgcolor=(0.0, 0.0, 0.0) ---> black background
mlab.clf()


#------------------------------------------------
# Load the data, we need to remove the 
# first 6 lines and the space after
#------------------------------------------------
with open(filename1, 'r') as file:
    content = file.readlines()

no_atoms = int(content[2].split()[0])
nx = int(content[3].split()[0])  #number of grid points on X-axis
ny = int(content[4].split()[0])  #number of grid points on Y-axis
nz = int(content[5].split()[0])  #number of grid points on Z-axis

# Print the extracted values
print(f'Number of atoms: {no_atoms}')
print(f'Grid points (nx, ny, nz): ({nx}, {ny}, {nz})')

# Remove the first 6 lines plus the number of atoms
content = content[(6 + no_atoms):]
# Join the remaining lines into a single string
data_str = ' '.join(content)
data = np.fromstring(data_str, sep=' ')
# Reshape the numpy array to the desired dimensions
data.shape = (nx, ny, nz)
#------------------------------------------------

# Display the electron density distribution
source = mlab.pipeline.scalar_field(data)
#vol = mlab.pipeline.volume(source, vmin=0.0001, vmax=1.0)
vol = mlab.pipeline.volume(source, vmin=0.0001, vmax=0.20)

#=============== here above you may play with the vmin and vmax values =============


# Can change the position and direction of camera
#mlab.view(132, 54, 45, [21, 20, 21.5])
mlab.view(azimuth=90, elevation=90, focalpoint='auto')

mlab.show()
