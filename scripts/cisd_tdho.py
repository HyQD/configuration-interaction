import os
import time
import numpy as np
import matplotlib.pyplot as plt


from quantum_systems import TwoDimensionalHarmonicOscillator
from configuration_interaction import CISD

os.environ["QS_CACHE_TDHO"] = "1"

tdho = TwoDimensionalHarmonicOscillator(2, 66, 10, 401)
t_0 = time.time()
tdho.setup_system(verbose=True)
t_1 = time.time()

print(f"Time spent setting up system: {t_1 - t_0} sec")

cisd = CISD(tdho, verbose=True)

cisd.compute_ground_state()

print(cisd.energies[0])
