import numpy as np
import matplotlib.pyplot as plt
import tqdm

from tdhf import HartreeFock, TimeDependentHartreeFock, RestrictedHartreeFock
from quantum_systems import construct_psi4_system, construct_pyscf_system
from quantum_systems.time_evolution_operators import LaserField
from configuration_interaction import TDCISD, TDCISDT, TDCISDTQ
from configuration_interaction import CISDTQ, CISD
from configuration_interaction.integrators import GaussIntegrator
from quantum_systems import CustomSystem
import psi4

# Memory for numpy in GB
numpy_memory = 2
molecule = """
Be 0.0 0.0 0.0
symmetry c1
"""

options = {'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8}


psi4.core.be_quiet()
psi4.set_options(options)

mol = psi4.geometry(molecule)
nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

wavefunction = psi4.core.Wavefunction.build(
    mol, psi4.core.get_global_option("BASIS")
)

molecular_integrals = psi4.core.MintsHelper(wavefunction.basisset())

kinetic = np.asarray(molecular_integrals.ao_kinetic())
potential = np.asarray(molecular_integrals.ao_potential())

H = kinetic + potential
I = np.asarray(molecular_integrals.ao_eri()).transpose(0, 2, 1, 3)
S = np.asarray(molecular_integrals.ao_overlap())
n_elec = 4

rhf = RestrictedHartreeFock(n_elec,H,I,S)
rhf.set_Econv(1e-15)

C,SCF_E = rhf.doRhf()

h_new = np.dot(C.T,np.dot(H,C))
# abcd, ds -> abcs
I = np.tensordot(I, C, axes=(3, 0))
# abcs, cr -> absr -> abrs
I = np.tensordot(I, C, axes=(2, 0)).transpose(0, 1, 3, 2)
# abrs, qb -> arsq -> aqrs
I = np.tensordot(I, C.T, axes=(1, 1)).transpose(0, 3, 1, 2)
# pa, aqrs -> pqrs
I = np.tensordot(C.T, I, axes=(1, 0))

n_spin_orbitals = 2*I.shape[0]

system = CustomSystem(n_elec,n_spin_orbitals)
#print(h_new)
system.set_h(h_new,add_spin=True)
#print(system.h)
system.set_u(I,add_spin=True,anti_symmetrize=True)

cisdtq = CISDTQ(system, verbose=True, np=np)
cisdtq.setup_ci_space()

cisdtq.spin_reduce_states()
cisdtq.compute_ground_state(k=1)
print("Efci: %.10f" % cisdtq.energies[0])