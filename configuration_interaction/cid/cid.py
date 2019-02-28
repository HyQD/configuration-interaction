from configuration_interaction.ci import ConfigurationInteraction
from configuration_interaction.ci_helper import (
    create_reference_state,
    create_doubles_states,
    get_diff_lists,
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
        shape = (self.num_states, self.l // 32 + 1)

        if self.verbose:
            print(
                "Size of a state in bytes: {0}".format((self.l // 32 + 1) * 4)
            )

        self.states = np.zeros(shape, dtype=np.uint32)

    def setup_ci_space(self):
        create_reference_state(self.n, self.l, self.states)
        create_doubles_states(self.n, self.l, self.states, 1)
        diff_by_one, diff_by_two = get_diff_lists(self.states)
