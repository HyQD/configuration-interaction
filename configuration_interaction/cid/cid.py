from configuration_interaction.ci import ConfigurationInteraction


class CID(ConfigurationInteraction):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np

        self.num_states = self.n * (self.n - 1) // 2
        self.num_states *= self.l * (self.l - 1) // 2

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
        pass
