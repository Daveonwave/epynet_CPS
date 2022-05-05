class GenericAttacker:
    def __init__(self, name, target, event_start, event_end, tags=None):
        """

        """
        self.name = name
        self.target = target
        self.event_start = event_start
        self.event_end = event_end
        self.tags = tags

        self.old_readings = None

    def apply_attack(self, variables):
        pass


class MITM(GenericAttacker):
    def apply_attack(self, variables):
        """
        Inject the fake value for each tag declared in tags dictionary
        """
        for var in self.tags:
            variables[var['tag']][var['property']] = var['value']
        return variables


class DOS(GenericAttacker):
    def apply_attack(self, variables):
        """
        # TODO: starts the DOS from the second timestep, otherwise we need to get the previous env state
        """
        if self.old_readings:
            variables = self.old_readings
        else:
            self.old_readings = variables
        return variables


class NetworkDelay(GenericAttacker):
    def apply_attack(self, variables):
        return variables


class AttackerGenerator:
    """
    Generates randomized attacks parsing the attackers' configuration file provided
    """
    def parse_configuration(self, attackers_config):
        pass


