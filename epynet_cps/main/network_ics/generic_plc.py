from physical_process_new import WaterDistributionNetwork


class GenericPLC:
    def __init__(self, name, wn: WaterDistributionNetwork, plc_variables, attacks=None):
        """
        :param name:
        :param wn:
        :param plc_variables:
        :param attacks:
        """
        self.name = name
        self.wn = wn

        self.owned_vars = plc_variables
        self.attacks = attacks

        self.elapsed_time = 0
        self.ongoing_attack_flag = 0

        self.ground_readings = {}
        self.altered_readings = {}

    def get_own_vars(self):
        # initialize self.owned_vars through a config file
        pass

    def apply(self, var_list):
        pass

    def save_ground_data(self):
        pass

    def check_for_ongoing_attacks(self):
        if self.attacks:
            return 1
        else:
            return 0

    # modifies the captured readings depending on the kind of attack
    def inject_attack(self):
        pass


class SensorPLC(GenericPLC):

    def apply(self, readings=None):
        """
        Reads data from the physical process
        """
        readings = {}
        # time used to print results
        self.elapsed_time = self.wn.times[-1]

        # TODO: list of data passed as parameter in config file
        for var in self.owned_vars.keys():
            readings[var] = {}
            for prop in self.owned_vars[var]:
                readings[var][prop] = self.wn.nodes[var].results[prop][-1]

        #print(self.name + ' reads...')
        # self.save_ground_data()

        # Apply attacks
        if self.check_for_ongoing_attacks():
            self.inject_attack()
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

        #print(new_status_dict)

        # Apply attacks
        if self.check_for_ongoing_attacks():
            self.inject_attack()

        # Update pump status
        n_updates = self.wn.update_actuators_status(new_status=new_status_dict)
        return n_updates
