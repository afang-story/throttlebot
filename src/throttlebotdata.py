'''
List of options for percentage of allocations
For each setup
- update resource_config_csv
- run experiment on it
- record results
Convert results to binary hyperparameters
'''

import csv
import run_experiment, modify_resources, weighting_conversions, run_throttlebot, poll_cluster_state
import os
import argparse
import numpy as np

from copy import deepcopy
from collections import namedtuple
from collections import Counter
from mr_gradient import *
from stress_analyzer import *
from weighting_conversions import *
from remote_execution import *
from run_experiment import *
from container_information import *
from filter_policy import *
from poll_cluster_state import *
from instance_specs import *
from mr	import MR
from run_throttlebot import *

import redis.client
import redis_client as tbot_datastore
import redis_resource as resource_datastore
import modify_resources as resource_modifier
import visualizer as chart_generator

# configs = []
# r_config = [['SERVICE', 'RESOURCE', 'AMOUNT', 'REPR'],
#             ['haproxy:1.7', 'CPU-QUOTA', '0', 'PERCENT'],
#             ['quilt/mongo', 'CPU-QUOTA', '0', 'PERCENT'],
#             ['node-app:node-todo.git', 'CPU-QUOTA', '0', 'PERCENT'],
#             ['haproxy:1.7', 'DISK', '0', 'PERCENT'],
#             ['quilt/mongo', 'DISK', '0', 'PERCENT'],
#             ['node-app:node-todo.git', 'DISK', '0', 'PERCENT'],
#             ['haproxy:1.7', 'MEMORY', '0', 'PERCENT'],
#             ['quilt/mongo', 'MEMORY', '0', 'PERCENT'],
#             ['node-app:node-todo.git', 'MEMORY', '0', 'PERCENT'],
#             ['haproxy:1.7', 'NET', '0', 'PERCENT'],
#             ['quilt/mongo', 'NET', '0', 'PERCENT'],
#             ['node-app:node-todo.git', 'NET', '0', 'PERCENT'],
#             ]
# print "Configuration being parsed"
# my_config = "test_mean.cfg"
# sys_config, workload_config, filter_config = run_throttlebot.parse_config_file(my_config)
# print "Configuration parsed"

# with open("p_configs.csv") as csvfile:
#     reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
#     for row in reader: # each row is a list
#         configs.append(row)
# configs = np.loadtxt("p_configs.csv", delimiter=",")
# configs = np.loadtxt("testpcon.csv", delimiter=",")
# print "Data Loaded"
# print str(len(configs)) + " configs"

# for loop_num in range(len(configs)):
#     print "Loop: " + str(loop_num)
#     for i in range(len(r_config) - 1):
#         r_config[i+1][2] = configs[loop_num][i]
#     np.savetxt("tbot_collect.csv", r_config, fmt='%s', delimiter=",")
#     mr_allocation = run_throttlebot.parse_resource_config_file("tbot_collect.csv", sys_config)
#
#     mr_allocation = run_throttlebot.filter_mr(mr_allocation,
#                               sys_config['stress_these_resources'],
#                               sys_config['stress_these_services'],
#                               sys_config['stress_these_machines'])
#
#     measurement = getMeasurement(sys_config, workload_config, filter_config, mr_allocation)
#     if measurement is None:
#         bench = None
#     else:
#         bench = (sum(measurement['latency_99'])/len(measurement['latency_99']))
#     print bench
#     with open('improvementlist.csv', 'ab') as f:
#         writer = csv.writer(f)
#         writer.writerows([[bench]])

def runCopy(sys_config, workload_config, filter_config, default_mr_config, last_completed_iter=0):
    redis_host = sys_config['redis_host']
    baseline_trials = sys_config['baseline_trials']
    experiment_trials = sys_config['trials']
    stress_weights = sys_config['stress_weights']
    stress_policy = sys_config['stress_policy']
    resource_to_stress = sys_config['stress_these_resources']
    service_to_stress = sys_config['stress_these_services']
    vm_to_stress = sys_config['stress_these_machines']
    machine_type = sys_config['machine_type']
    quilt_overhead = sys_config['quilt_overhead']
    gradient_mode = sys_config['gradient_mode']
    setting_mode = sys_config['setting_mode']
    fill_services_first = sys_config['fill_services_first']

    preferred_performance_metric = workload_config['tbot_metric']
    optimize_for_lowest = workload_config['optimize_for_lowest']

    filter_policy= filter_config['filter_policy']

    redis_db = redis.StrictRedis(host=redis_host, port=6379, db=0)
    if last_completed_iter == 0:
        redis_db.flushall()

    init_cluster_capacities_r(redis_db, machine_type, quilt_overhead)
    init_service_placement_r(redis_db, default_mr_config)
    temp_val = init_resource_config(redis_db, default_mr_config, machine_type, workload_config)
    if (temp_val == -11):
        return -11

    print 'INFO: INSTALLING DEPENDENCIES'
    install_dependencies(workload_config)

    # Initialize time for data charts
    time_start = datetime.datetime.now()

    print '*' * 20
    print 'INFO: RUNNING BASELINE'

    # Get the Current Performance -- not used for any analysis, just to benchmark progress!!
    current_performance = measure_baseline(workload_config,
                                           baseline_trials,
                                           workload_config['include_warmup'])

    current_performance[preferred_performance_metric] = remove_outlier(current_performance[preferred_performance_metric])
    return current_performance
    # current_time_stop = datetime.datetime.now()
    # time_delta = current_time_stop - time_start
    #
    # print 'Current (non-analytic) performance measured: {}'.format(current_performance)

    # if last_completed_iter == 0:
    #     tbot_datastore.write_summary_redis(redis_db,
    #                                        0,
    #                                        MR('initial', 'initial', []),
    #                                        0,
    #                                        {},
    #                                        mean_list(current_performance[preferred_performance_metric]),
    #                                        mean_list(current_performance[preferred_performance_metric]),
    #                                        time_delta.seconds, 0)
    #
    # print '============================================'
    # print '\n' * 2
    #
    # # Initialize the current configurations
    # # Initialize the working set of MRs to all the MRs
    # mr_working_set = resource_datastore.get_all_mrs(redis_db)
    # resource_datastore.write_mr_working_set(redis_db, mr_working_set, 0)
    # cumulative_mr_count = 0
    # experiment_count = last_completed_iter + 1
    #
    # while experiment_count < 1:
    #     # Calculate the analytic baseline that is used to determine MRs
    #     analytic_provisions = prepare_analytic_baseline(redis_db, sys_config, min(stress_weights))
    #     print 'The Analytic provisions are as follows {}'.format(analytic_provisions)
    #     for mr in analytic_provisions:
    #         resource_modifier.set_mr_provision(mr, analytic_provisions[mr], workload_config)
    #
    #     if len(analytic_provisions) != 0:
    #         analytic_baseline = measure_runtime(workload_config, experiment_trials)
    #     else:
    #         analytic_baseline = deepcopy(current_performance)
    #     print analytic_baseline
    #     analytic_mean = mean_list(analytic_baseline[preferred_performance_metric])
    #     print 'The analytic baseline is {}'.format(analytic_baseline)
    #     print 'This current performance is {}'.format(current_performance)
    #     analytic_baseline[preferred_performance_metric] = remove_outlier(analytic_baseline[preferred_performance_metric])
    #
    #     # Get a list of MRs to stress in the form of a list of MRs
    #     mr_to_consider = apply_filtering_policy(redis_db,
    #                                           mr_working_set,
    #                                           experiment_count,
    #                                           sys_config,
    #                                           workload_config,
    #                                           filter_config)
    #
    #     for mr in mr_to_consider:
    #         print '\n' * 2
    #         print '*' * 20
    #         print 'Current MR is {}'.format(mr.to_string())
    #         increment_to_performance = {}
    #         current_mr_allocation = resource_datastore.read_mr_alloc(redis_db, mr)
    #         print 'Current MR allocation is {}'.format(current_mr_allocation)
    #
    #         for stress_weight in stress_weights:
    #             # Calculate Gradient Schedule and provision resources accordingly
    #             mr_gradient_schedule = calculate_mr_gradient_schedule(redis_db, [mr],
    #                                                                   sys_config,
    #                                                                   stress_weight)
    #             for change_mr in mr_gradient_schedule:
    #                 resource_modifier.set_mr_provision(change_mr, mr_gradient_schedule[change_mr], workload_config)
    #
    #             experiment_results = measure_runtime(workload_config, experiment_trials)
    #
    #             # Write results of experiment to Redis
    #             # preferred_results = remove_outlier(experiment_results[preferred_performance_metric])
    #             preferred_results = experiment_results[preferred_performance_metric]
    #             mean_result = mean_list(preferred_results)
    #             tbot_datastore.write_redis_ranking(redis_db, experiment_count,
    #                                                preferred_performance_metric,
    #                                                mean_result, mr, stress_weight)
    #
    #             # Revert the Gradient schedule and provision resources accordingly
    #             mr_revert_gradient_schedule = revert_mr_gradient_schedule(redis_db,
    #                                                                       [mr],
    #                                                                       sys_config,
    #                                                                       stress_weight)
    #             for change_mr in mr_revert_gradient_schedule:
    #                 resource_modifier.set_mr_provision(change_mr, mr_revert_gradient_schedule[change_mr], workload_config)
    #
    #             increment_to_performance[stress_weight] = experiment_results
    #
    #         # Write the results of the iteration to Redis
    #         tbot_datastore.write_redis_results(redis_db, mr, increment_to_performance, experiment_count, preferred_performance_metric)
    #         print '*' * 20
    #         print '\n' * 2
    #
    #     # Timing Information for the purpose of experiments
    #     current_time_stop = datetime.datetime.now()
    #     time_delta = current_time_stop - time_start
    #     cumulative_mr_count += len(mr_to_consider)
    #     chart_generator.get_summary_mimr_charts(redis_db, workload_config,
    #                                             current_performance, mr_working_set,
    #                                             experiment_count, stress_weights,
    #                                             preferred_performance_metric, time_start)
    #
    #     # Recover the results of the experiment from Redis
    #     max_stress_weight = min(stress_weights)
    #     mimr_list = tbot_datastore.get_top_n_mimr(redis_db, experiment_count,
    #                                               preferred_performance_metric,
    #                                               max_stress_weight, gradient_mode,
    #                                               optimize_for_lowest=optimize_for_lowest,
    #                                               num_results_returned=-1)
    #
    #     imr_list, nimr_list = seperate_mr(mimr_list, mean_list(analytic_baseline[preferred_performance_metric]), optimize_for_lowest)
    #     if len(imr_list) == 0:
    #         print 'INFO: IMR list length is 0. Please choose a metric with more signal. Exiting...'
    #         break
    #     print 'INFO: IMR list is {}'.format([mr.to_string() for mr in imr_list])
    #     print 'INFO: NIMR list is {}'.format([mr.to_string() for mr in nimr_list])
    #
    #     # Try all the MIMRs in the list until a viable improvement is determined
    #     # Improvement Amount
    #     mimr = None
    #     action_taken = {}
    #
    #     for imr in imr_list:
    #         imr_improvement_percent = improve_mr_by(redis_db, imr, max_stress_weight)
    #         current_imr_alloc = resource_datastore.read_mr_alloc(redis_db, imr)
    #         new_imr_alloc = convert_percent_to_raw(imr, current_imr_alloc, imr_improvement_percent)
    #         imr_improvement_proposal = int(new_imr_alloc - current_imr_alloc)
    #
    #         # If the the Proposed MR cannot be improved by the proposed amount, there are two options
    #         # - Max out the resources to fill up the remaining resources on the machine
    #         # - Resource Stealing from NIMRs
    #         # Both functions will return VIABLE improvements to the IMR deployment
    #         nimr_diff_proposal = {}
    #         if check_improve_mr_viability(redis_db, imr, imr_improvement_proposal) is False:
    #             print 'INFO: MR {} to increase {} by {} is not viable'.format(imr.to_string(),
    #                                                                           current_imr_alloc,
    #                                                                           imr_improvement_proposal)
    #             print 'INFO: Attempting to max out the machines resources...'
    #             imr_improvement_proposal = fill_out_resource(redis_db, imr)
    #
    #             if imr_improvement_proposal <= 0:
    #                 print 'INFO: No more space to fill out resources. Stealing from NIMRs'
    #                 # Calculate a plan to reduce the resource provisioning of NIMRs
    #                 nimr_diff_proposal,imr_improvement_proposal = create_decrease_nimr_schedule(redis_db,
    #                                                                                             imr,
    #                                                                                             nimr_list,
    #                                                                                             max_stress_weight)
    #                 print 'INFO: Proposed NIMR {}'.format(nimr_diff_proposal)
    #                 print 'INFO: New IMR improvement {}'.format(imr_improvement_proposal)
    #
    #                 if len(nimr_diff_proposal) == 0 or imr_improvement_proposal == 0:
    #                     if filter_policy is None:
    #                         action_taken[imr] = 0
    #                         continue
    #
    #                     # Special actions for Filtered results
    #                     filtered_nimr_list = find_colocated_nimrs(redis_db, imr, mr_working_set, analytic_mean, sys_config, workload_config)
    #                     nimr_diff_proposal,imr_improvement_proposal = create_decrease_nimr_schedule(redis_db,
    #                                                                                                 imr,
    #                                                                                                 filtered_nimr_list,
    #                                                                                                 max_stress_weight)
    #                     if len(nimr_diff_proposal) == 0 or imr_improvement_proposal == 0:
    #                         continue
    #
    #         # Decrease the amount of resources provisioned to the NIMR
    #         for nimr in nimr_diff_proposal:
    #             action_taken[nimr] = nimr_diff_proposal[nimr]
    #             new_nimr_alloc = resource_datastore.read_mr_alloc(redis_db, nimr) + nimr_diff_proposal[nimr]
    #             print 'NIMR stealing: imposing a change of {} on {}'.format(action_taken[nimr],
    #                                                                         nimr.to_string())
    #             finalize_mr_provision(redis_db, nimr, new_nimr_alloc, workload_config)
    #
    #         print 'Taking an accounting before decreasing...'
    #         print 'IMR {} is currently at {}, trying to improve by {}'.format(imr.to_string(), current_imr_alloc, imr_improvement_proposal)
    #         # Improving the resource should always be viable at this step
    #         if check_improve_mr_viability(redis_db, imr, imr_improvement_proposal):
    #             new_imr_alloc = imr_improvement_proposal + current_imr_alloc
    #             action_taken[imr] = imr_improvement_proposal
    #             finalize_mr_provision(redis_db, imr, new_imr_alloc, workload_config)
    #             print 'Improvement Calculated: MR {} increase from {} to {}'.format(imr.to_string(), current_imr_alloc, new_imr_alloc)
    #             mimr = imr
    #             break
    #         else:
    #             action_taken[imr] = 0
    #             new_imr_alloc = imr_improvement_proposal + current_imr_alloc
    #             print 'Improvement Calculated: MR {} failed to improve from {} to {}'.format(imr.to_string(),
    #                                                                                          current_imr_alloc,
    #                                                                                          imr_improvement_proposal)
    #             print 'This IMR cannot be improved. Printing some debugging before exiting...'
    #
    #             print 'Current MR allocation is {}'.format(current_imr_alloc)
    #             print 'Proposed (failed) allocation is {}, improved by {}'.format(new_imr_alloc, imr_improvement_proposal)
    #
    #             for deployment in imr.instances:
    #                 vm_ip,container = deployment
    #                 capacity = resource_datastore.read_machine_capacity(redis_db, vm_ip)
    #                 consumption = resource_datastore.read_machine_consumption(redis_db, vm_ip)
    #                 print 'Machine {} Capacity is {}, and consumption is currently {}'.format(vm_ip, capacity, consumption)
    #
    #     if mimr is None:
    #         print 'No viable improvement found'
    #         break
    #
    #     # Move back into the normal operating basis by removing the baseline prep stresses
    #     reverted_analytic_provisions = revert_analytic_baseline(redis_db, sys_config)
    #     for mr in reverted_analytic_provisions:
    #         resource_modifier.set_mr_provision(mr, reverted_analytic_provisions[mr], workload_config)
    #
    #     #Compare against the baseline at the beginning of the program
    #     improved_performance = measure_runtime(workload_config, baseline_trials)
    #     # improved_performance[preferred_performance_metric] = remove_outlier(improved_performance[preferred_performance_metric])
    #     improved_mean = mean_list(improved_performance[preferred_performance_metric])
    #     print improved_mean
    #     return improved_mean
    #     previous_mean = mean_list(current_performance[preferred_performance_metric])
    #     performance_improvement = improved_mean - previous_mean
    #
    #     # Write a summary of the experiment's iterations to Redis
    #     tbot_datastore.write_summary_redis(redis_db, experiment_count, mimr,
    #                                        performance_improvement, action_taken,
    #                                        analytic_mean, improved_mean,
    #                                        time_delta.seconds, cumulative_mr_count)
    #     current_performance = improved_performance
    #
    #     # Generating overall performance improvement
    #     chart_generator.get_summary_performance_charts(redis_db, workload_config, experiment_count, time_start)
    #
    #     results = tbot_datastore.read_summary_redis(redis_db, experiment_count)
    #     print 'Results from iteration {} are {}'.format(experiment_count, results)
    #
    #     # Checkpoint MR configurations and print
    #     current_mr_config = resource_datastore.read_all_mr_alloc(redis_db)
    #     print_csv_configuration(current_mr_config)
    #     experiment_count += 1
    #
    # print '{} experiments completed'.format(experiment_count)
    # print_all_steps(redis_db, experiment_count, sys_config, workload_config, filter_config)
    #
    # current_mr_config = resource_datastore.read_all_mr_alloc(redis_db)
    # for mr in current_mr_config:
    #     print '{} = {}'.format(mr.to_string(), current_mr_config[mr])
    #
    # print_csv_configuration(current_mr_config)

def queryPoint(config, num_worker):
    # r_config = [['SERVICE', 'RESOURCE', 'AMOUNT', 'REPR'],
    #             ['haproxy:1.7', 'CPU-QUOTA', '0', 'PERCENT'],
    #             ['quilt/mongo', 'CPU-QUOTA', '0', 'PERCENT'],
    #             ['node-app:node-todo.git', 'CPU-QUOTA', '0', 'PERCENT'],
    #             ['haproxy:1.7', 'DISK', '0', 'PERCENT'],
    #             ['quilt/mongo', 'DISK', '0', 'PERCENT'],
    #             ['node-app:node-todo.git', 'DISK', '0', 'PERCENT'],
    #             ['haproxy:1.7', 'MEMORY', '0', 'PERCENT'],
    #             ['quilt/mongo', 'MEMORY', '0', 'PERCENT'],
    #             ['node-app:node-todo.git', 'MEMORY', '0', 'PERCENT'],
    #             ['haproxy:1.7', 'NET', '0', 'PERCENT'],
    #             ['quilt/mongo', 'NET', '0', 'PERCENT'],
    #             ['node-app:node-todo.git', 'NET', '0', 'PERCENT'],
    #             ]
    r_config = [['SERVICE', 'RESOURCE', 'AMOUNT', 'REPR'],
                ['hantaowang/bcd-spark-master', 'CPU-CORE', '0', 'RAW'],
                ['hantaowang/bcd-spark', 'CPU-CORE', '0', 'RAW'],
                ['hantaowang/bcd-spark-master', 'DISK', '0', 'PERCENT'],
                ['hantaowang/bcd-spark', 'DISK', '0', 'PERCENT'],
                ['hantaowang/bcd-spark-master', 'NET', '0', 'PERCENT'],
                ['hantaowang/bcd-spark', 'NET', '0', 'PERCENT'],
                ]
    sys_config, workload_config, filter_config = run_throttlebot.parse_config_file("test_spark.cfg")
    for i in range(len(r_config) - 1):
        r_config[i+1][2] = config[i]
    np.savetxt("tbot_collect.csv", r_config, fmt='%s', delimiter=",")
    mr_allocation = run_throttlebot.parse_resource_config_file("tbot_collect.csv", sys_config)

    mr_allocation = run_throttlebot.filter_mr(mr_allocation,
                              sys_config['stress_these_resources'],
                              sys_config['stress_these_services'],
                              sys_config['stress_these_machines'])
    if workload_config['type'] == 'bcd':
        all_vm_ip = poll_cluster_state.get_actual_vms()
        service_to_deployment = poll_cluster_state.get_service_placements(all_vm_ip)
        workload_config['request_generator'] = [service_to_deployment['hantaowang/bcd-spark-master'][0][0]]
        workload_config['frontend'] = [service_to_deployment['hantaowang/bcd-spark-master'][0][0]]
        workload_config['additional_args'] = {'container_id': service_to_deployment['hantaowang/bcd-spark-master'][0][1]}
        workload_config['resources'] = {
            'spark.executor.cores': '8',
            'spark.driver.cores': '8',
            'spark.executor.memory': str(int(32 * 0.8)) + 'g',
            'spark.driver.memory': str(int(32 * 0.8)) + 'g',
            'spark.cores.max': '48'
        }
        workload_config['instances'] = service_to_deployment['hantaowang/bcd-spark'] + service_to_deployment['hantaowang/bcd-spark-master']
        print workload_config

    # measurement = run_experiment.measure_bcd(workload_config, sys_config['baseline_trials'], num_worker) # getMeasurement(sys_config, workload_config, filter_config, mr_allocation)
    measurement = runCopy(sys_config, workload_config, filter_config, mr_allocation)
    print measurement
    if measurement is None:
        bench = None
    else:
        bench = (sum(measurement['latency'])/len(measurement['latency']))
    print bench
    return bench
