from physical_process import WaterDistributionNetwork

# TODO: discuss the hierarchy (can differ between sensor and actuators).
# Used when we have concurrent attacks to decide which one has to be applied before.
attackers_hierarchy = ['NetworkDelay', 'DOS', 'MITM']


class GenericPLC:
    def __init__(self, name, wn: WaterDistributionNetwork, plc_variables):
        """
        :param name:
        :param wn:
        :param plc_variables:
        :param attackers:
        """
        self.name = name
        self.wn = wn

        self.owned_vars = plc_variables
        self.attackers = None
        self.active_attackers = []

        self.elapsed_time = 0
        self.ongoing_attack_flag = 0

        self.ground_readings = {}
        self.altered_readings = {}

    def get_own_vars(self):
        # initialize self.owned_vars through a config file
        pass

    def set_attackers(self, attackers=None):
        """
        Set the attackers as a list of attacker objects
        """
        self.attackers = attackers

    def apply(self, var_list):
        pass

    def save_ground_data(self):
        pass

    def check_for_ongoing_attacks(self):
        """
        Check if there are ongoing attacks and return 1 in that case. Moreover it extracts active attackers to apply
        the injection of the attack.
        """
        self.active_attackers = []
        self.ongoing_attack_flag = 0

        if self.attackers:
            curr_attackers = [attacker for attacker in self.attackers
                              if attacker.event_start <= self.elapsed_time < attacker.event_end]
            if curr_attackers:
                # Order the attackers following a predefined hierarchy
                for item in attackers_hierarchy:
                    self.active_attackers.extend([attacker for attacker in curr_attackers if type(attacker).__name__ == item])
                print(self.name, [type(att).__name__ for att in self.active_attackers])
                self.ongoing_attack_flag = 1

        return self.ongoing_attack_flag


class SensorPLC(GenericPLC):

    def init_readings(self, var, prop):
        """

        """
        if var in self.owned_vars['nodes']:
            return getattr(self.wn.nodes[var], prop)
        else:
            raise Exception("Variable {} is not controlled by {}".format(var, self.name))

    def apply(self, readings=None):
        """
        Reads data from the physical process
        """
        readings = {}
        # time used to print results
        self.elapsed_time = self.wn.times[-1]

        # TODO: list of data passed as parameter in config file
        for object_type in self.owned_vars.keys():
            for var in self.owned_vars[object_type].keys():
                readings[var] = {}
                for prop in self.owned_vars[object_type][var]:
                    readings[var][prop] = getattr(self.wn, object_type)[var].results[prop][-1]
        # self.save_ground_data()

        # Apply attacks
        if self.check_for_ongoing_attacks():
            for attacker in self.active_attackers:
                readings = attacker.apply_attack(readings)
            # save altered readings

        readings['under_attack'] = self.ongoing_attack_flag
        return readings


class ActuatorPLC(GenericPLC):

    def apply(self, action):
        """
        Reads data from agent's action space and apply them into the physical process
        """
        # Action translated in binary value with one bit for each pump status
        new_status_dict = {pump_id: 0 for pump_id in self.owned_vars}
        bin_action = '{0:0{width}b}'.format(action[0], width=len(self.owned_vars))

        for i, key in enumerate(new_status_dict.keys()):
            new_status_dict[key] = int(bin_action[i])

        # Apply attacks
        if self.check_for_ongoing_attacks():
            for attacker in self.attackers:
                new_status_dict = attacker.apply_attack(new_status_dict)

        # TODO: understand how to handle attacks from actuators perspective

        # Update pump status
        n_updates = self.wn.update_actuators_status(new_status=new_status_dict)
        return n_updates
