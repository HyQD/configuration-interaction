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

import pyscf


mol = pyscf.gto.Mole()
mol.unit = "bohr"
mol.build(atom="be 0.0 0.0 0.0", basis="cc-pvdz", symmetry=False)
mol.set_common_origin(np.array([0.0, 0.0, 0.0]))

n = mol.nelectron
l = mol.nao * 2

H = pyscf.scf.hf.get_hcore(mol)
S = mol.intor_symmetric("int1e_ovlp")
I = (
    mol.intor("int2e")
    .reshape(l // 2, l // 2, l // 2, l // 2)
    .transpose(0, 2, 1, 3)
)


rhf = RestrictedHartreeFock(n, H, I, S)
rhf.set_Econv(1e-10)

C, SCF_E = rhf.doRhf()

h_new = np.dot(C.T, np.dot(H, C))
# abcd, ds -> abcs
I = np.tensordot(I, C, axes=(3, 0))
# abcs, cr -> absr -> abrs
I = np.tensordot(I, C, axes=(2, 0)).transpose(0, 1, 3, 2)
# abrs, qb -> arsq -> aqrs
I = np.tensordot(I, C.T, axes=(1, 1)).transpose(0, 3, 1, 2)
# pa, aqrs -> pqrs
I = np.tensordot(C.T, I, axes=(1, 0))

n_spin_orbitals = 2 * I.shape[0]

system = CustomSystem(n, n_spin_orbitals)
# print(h_new)
system.set_h(h_new, add_spin=True)
# print(system.h)
system.set_u(I, add_spin=True, anti_symmetrize=True)

cisdtq = CISDTQ(system, verbose=True, np=np)
cisdtq.setup_ci_space()

cisdtq.spin_reduce_states()
cisdtq.compute_ground_state(k=1)
print("Efci: %.10f" % cisdtq.energies[0])
