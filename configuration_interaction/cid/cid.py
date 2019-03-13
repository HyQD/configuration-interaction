import time

from configuration_interaction.ci import ConfigurationInteraction
from configuration_interaction.ci_helper import (
    create_reference_state,
    create_doubles_states,
    get_diff_lists,
    BITTYPE,
    BITSTRING_SIZE,
)


class CID(ConfigurationInteraction):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np

        self.num_states = (
            self.n * (self.n - 1) // 2 * self.m * (self.m - 1) // 2 + 1
        )

        if self.verbose:
            print("Number of states to create: {0}".format(self.num_states))

        # Find the shape of the states array
        # Each state is represented as a bit string padded to the nearest
        # 32-bit boundary
        shape = (self.num_states, self.l // BITSTRING_SIZE + 1)

        if self.verbose:
            print(
                "Size of a state in bytes: {0}".format(
                    (self.l // BITSTRING_SIZE + 1) * 4
                )
            )

        self.states = np.zeros(shape, dtype=BITTYPE)

    def setup_ci_space(self):
        t0 = time.time()
        create_reference_state(self.n, self.l, self.states)
        create_doubles_states(self.n, self.l, self.states, index=1)
        t1 = time.time()

        if self.verbose:
            print("Time spent setting up CID space: {0} sec".format(t1 - t0))

        diff_by_one, diff_by_two = get_diff_lists(self.states)
