if __name__ == '__main__':
    from scripts import network, epynetUtils, objFunction, differentialEvolution
    import pandas as pd
    import datetime

    net = network.WaterDistributionNetwork("anytown_pd.inp")
    print(net.junctions['20'].pattern.values)

    # Put report step equal to hyd_step because if it is lower it leads the timestep interval
    net.set_time_params(duration=3600*24*7, hydraulic_step=300, report_step=1200)

    pattern_csv = "../demand_patterns/demand_patterns_test_low.csv"
    junc_demands = pd.read_csv(pattern_csv)
    net.set_demand_pattern('junc_demand', junc_demands['0'], net.junctions)

    status = [1.0, 1.0]
    if net.valves:
        actuators_update_dict = {uid: status for uid in net.pumps.uid.append(net.valves.uid)}
    else:
        actuators_update_dict = {uid: status for uid in net.pumps.uid}

    net.run(interactive=True, status_dict=actuators_update_dict)
    # print(net.junctions.actual_demand.iloc[-1])

    print(net.tanks.overflow)
    print(net.df_nodes_report['tanks'])
    print(net.df_links_report['pumps'])

    exit(0)

    total_basedemand = net.df_nodes_report.iloc[:, net.df_nodes_report.columns.get_level_values(2) == 'basedemand'].iloc[-1].sum()
    total_demand = net.df_nodes_report.iloc[:, net.df_nodes_report.columns.get_level_values(2) == 'actual_demand'].iloc[-1].sum()
    print('basedemand', total_basedemand)
    print('demand', total_demand)
    print('ratio', total_demand/total_basedemand)

    ratio = objFunction.supply_demand_ratio(net)

    print(ratio)
    exit(0)

    net.demand_model_summary()

    #for junc in net.junctions:
    #    print("basedemand: " + str(junc.basedemand) + ";    demand: " + str(junc.emitter))
    #exit(0)

    status = [1.0, 0.0]
    if net.valves:
        actuators_update_dict = {uid: status for uid in net.pumps.uid.append(net.valves.uid)}
    else:
        actuators_update_dict = {uid: status for uid in net.pumps.uid}

    net.run(interactive=True, status_dict=actuators_update_dict)

    print(net.tanks['T1'].inflow)
    exit(0)

    pumps_energy = sum([net.df_links_report['pumps', pump_id, 'energy'].sum() for pump_id in net.pumps.uid])
    print(pumps_energy)
    exit(0)

    times = [datetime.timedelta(seconds=time) for time in net.times]
    # net.set_basedemand_pattern(2)
    # net.set_time_params(duration=3600)
    # print(net.pumps.results.index)

    # print(net.df_links_report.iloc[:, net.df_links_report.columns.get_level_values(2) == 'status'])
    for time in times[:5]:
        print(str(time) + ": " +
              str(net.df_nodes_report.loc[time]['junctions', 'J147', 'demand']))
