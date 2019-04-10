from configuration_interaction import CID
from configuration_interaction.tdci import TimeDependentConfigurationInteraction


class TDCID(TimeDependentConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        super().__init__(CID, *args, **kwargs)
