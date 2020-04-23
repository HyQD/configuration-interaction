import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import GeneralOrbitalSystem, ODQD
from configuration_interaction import CISD

odho = GeneralOrbitalSystem(2, ODQD(10, 8, 401, potential=ODQD.HOPotential(1)))

fci = CISD(odho, verbose=True).compute_ground_state()

print(fci.compute_energy())

plt.plot(odho.grid, fci.compute_particle_density().real)
plt.show()
