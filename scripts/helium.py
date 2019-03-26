import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tdhf import HartreeFock, TimeDependentHartreeFock
from quantum_systems import construct_psi4_system
from configuration_interaction import CIS, CID, CISD

He = """
He 0 0 0
symmetry c1
units bohr
"""

options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(He, options)
hf = HartreeFock(system)
C = hf.scf(max_iters=100, tolerance=1e-15)
system.change_basis(C)
print("E_total = E_hf + E_nuc: %g" % (hf.e_hf + system.Enuc))

cisd = CISD(system, brute_force=False, verbose=True, np=np)
cisd.setup_ci_space()
cisd.compute_ground_state()
print("CISD ground state energy: {0}".format(cisd.energies[0] + system.Enuc))

rho_qp = cisd.compute_one_body_density_matrix()
print("tr(rho): {0}".format(np.trace(rho_qp)))

eps, U = np.linalg.eigh(rho_qp)
print(eps)
print(2 * eps)

rho_qp_red = np.zeros((system.l // 2, system.l // 2))
for p in range(system.l // 2):
    for q in range(system.l // 2):
        rho_qp_red[p, q] = rho_qp[2 * p, 2 * q] + rho_qp[2 * p + 1, 2 * q + 1]
eps_red, U_red = np.linalg.eigh(rho_qp_red)
print(eps_red)
