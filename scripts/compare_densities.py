import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import ODQD
from configuration_interaction import CISD

od = ODQD(2, 12, 11, 201)
od.setup_system()

cisd = CISD(od, verbose=True)
cisd.compute_ground_state()

rho_cisd = cisd.compute_particle_density()
rho_qp = cisd.compute_one_body_density_matrix()
rho_qs = od.compute_particle_density(rho_qp)

plt.plot(od.grid, rho_cisd.real, label="CISD")
plt.plot(od.grid, rho_qs.real, label="QS")
plt.legend()
plt.show()

np.testing.assert_allclose(rho_cisd, rho_qs)
