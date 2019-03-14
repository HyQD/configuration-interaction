import time

from configuration_interaction.ci import ConfigurationInteraction
from configuration_interaction.ci_helper import (
    create_reference_state,
    create_singles_states,
    get_diff_lists,
    BITTYPE,
    BITSTRING_SIZE,
)


class CIS(ConfigurationInteraction):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np

        self.num_states = self.n * self.m + 1

        if self.verbose:
            print("Number of states to create: {0}".format(self.num_states))

        # Find the shape of the states array
        # Each state is represented as a bit string padded to the nearest
        # 32-bit boundary
        shape = (self.num_states, self.l // BITSTRING_SIZE + 1)

        if self.verbose:
            print(
                "Size of a state in bytes: {0}".format(
                    np.dtype(BITTYPE).itemsize * 1
                )
            )

        self.states = np.zeros(shape, dtype=BITTYPE)

    def setup_ci_space(self):
        t0 = time.time()
        create_reference_state(self.n, self.l, self.states)
        create_singles_states(self.n, self.l, self.states, index=1)
        t1 = time.time()

        if self.verbose:
            print("Time spent setting up CIS space: {0} sec".format(t1 - t0))
