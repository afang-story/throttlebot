import numpy as np
import scipy as scp
import csv
import run_experiment, modify_resources, weighting_conversions, run_throttlebot, poll_cluster_state, throttlebotdata
import os
import argparse
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from scipy.interpolate import griddata

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

def queryPoint(config):
    r_config = [['SERVICE', 'RESOURCE', 'AMOUNT', 'REPR'],
                ['haproxy:1.7', 'CPU-QUOTA', '0', 'PERCENT'],
                ['quilt/mongo', 'CPU-QUOTA', '0', 'PERCENT'],
                ['node-app:node-todo.git', 'CPU-QUOTA', '0', 'PERCENT'],
                ['haproxy:1.7', 'DISK', '0', 'PERCENT'],
                ['quilt/mongo', 'DISK', '0', 'PERCENT'],
                ['node-app:node-todo.git', 'DISK', '0', 'PERCENT'],
                ['haproxy:1.7', 'MEMORY', '0', 'PERCENT'],
                ['quilt/mongo', 'MEMORY', '0', 'PERCENT'],
                ['node-app:node-todo.git', 'MEMORY', '0', 'PERCENT'],
                ['haproxy:1.7', 'NET', '0', 'PERCENT'],
                ['quilt/mongo', 'NET', '0', 'PERCENT'],
                ['node-app:node-todo.git', 'NET', '0', 'PERCENT'],
                ]
    temp = [20,20,20]
    config = np.concatenate((config[:3], temp, config[3:]))
    sys_config, workload_config, filter_config = run_throttlebot.parse_config_file("test_mean.cfg")
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

    bench = 0
    z = 0
    while z < 1:
        measurement = throttlebotdata.runCopy(sys_config, workload_config, filter_config, mr_allocation)
        if (measurement == -11):
            return -11
        if measurement is None:
            z -= 1
        else:
            bench = (sum(measurement['latency_99'])/len(measurement['latency_99']))
        z += 1
    return bench

def perToRaw(amount, machine_type, resource):
    max_capacity = get_instance_specs(machine_type)[resource]
    amount = (amount / 100.0) * max_capacity
    return amount

def ptToRaw(pt):
    pt[0] = perToRaw(pt[0], 'm4.large', 'CPU-QUOTA')
    pt[1] = perToRaw(pt[1], 'm4.large', 'CPU-QUOTA')
    pt[2] = perToRaw(pt[2], 'm4.large', 'CPU-QUOTA')
    pt[3] = perToRaw(pt[3], 'm4.large', 'MEMORY')
    pt[4] = perToRaw(pt[4], 'm4.large', 'MEMORY')
    pt[5] = perToRaw(pt[5], 'm4.large', 'MEMORY')
    pt[6] = perToRaw(pt[6], 'm4.large', 'NET')
    pt[7] = perToRaw(pt[7], 'm4.large', 'NET')
    pt[8] = perToRaw(pt[8], 'm4.large', 'NET')
# tester=[]
# tester.append(queryPoint([30,30,30,30,30,30,30,30,30]))
# tester.append(queryPoint([5,20,20,20,20,20,20,20,20]))
# tester.append(queryPoint([35,20,20,20,20,20,20,20,20]))
# tester.append(queryPoint([20,5,20,20,20,20,20,20,20]))
# tester.append(queryPoint([20,35,20,20,20,20,20,20,20]))
# tester.append(queryPoint([20,20,5,20,20,20,20,20,20]))
# tester.append(queryPoint([20,20,35,20,20,20,20,20,20]))
# [2409.75, 3434.3333333333335, 3592.5, 3131.25, 3302.6666666666665, 5061.5, 5208.0]
# print tester
# print queryPoint([30,30,30,30,30,30,30,30,30])
# exit()

# Let centre be all 30, let box be +- 20, alpha be 8
feats = 9
alpha = 15
alpha2 = 5
box_len = 10
steps = 2
dist = 5
points= 1 + 2*feats + 2*feats + 2*feats*(feats-1)
center = [20, 20, 20, 20, 20, 20, 20, 20, 20]
'''training = np.ndarray(shape=(points,9))
z = 0
while (z < feats):
    training[2*z] = center[:]
    training[2*z, z] -= alpha
    training[2*z+1] = center[:]
    training[2*z+1, z] += alpha
    z += 1
z = 0
while (z < feats):
    training[2*feats+2*z] = center[:]
    training[2*feats+2*z, z] -= alpha2
    training[2*feats+2*z+1] = center[:]
    training[2*feats+2*z+1, z] += alpha2
    z += 1
z *= 4
for i in range(feats):
    j = i + 1
    while j < feats:
        if i != j:
            training[z] = center[:]
            training[z, i] += box_len
            training[z, j] -= box_len
            training[z+1] = center[:]
            training[z+1, i] -= box_len
            training[z+1, j] += box_len
            training[z+2] = center[:]
            training[z+2, i] += box_len
            training[z+2, j] += box_len
            training[z+3] = center[:]
            training[z+3, i] -= box_len
            training[z+3, j] -= box_len
            z += 4
        j += 1
training[-1] = center

k = 0
Xall = np.ndarray(shape=(2*steps,feats))
for i in range(steps):
    Xall[k] = center[:]
    Xall[k + 1] = center[:]
    for j in range(feats):
        Xall[k, j] -= (i + 1) * dist
        Xall[k + 1, j] += (i + 1) * dist
    k += 2

training = np.concatenate((training, Xall), axis=0)

np.savetxt("mean_training_c20_a15_5_b10_d5_s2_pm.csv", training, delimiter=",")

X = training'''
X = np.genfromtxt('mean_training_c20_a15_5_b10_d5_s2_pm.csv', delimiter=",")

# X = np.genfromtxt('mean_training_c20_a15_5_b10_d5_s2_pm.csv', delimiter=',')

'''Y = []
Y = np.genfromtxt('mean_training_results_c20_a15_5_b10_d5_s2_pm.csv', delimiter=',')
count = 5
Y = Y[0: count].tolist()
while count < len(X):
    pt = X[count]
    print count
    print pt
    Y.append(queryPoint(pt))
    count += 1
    np.savetxt("mean_training_results_c20_a15_5_b10_d5_s2_pm.csv", Y, delimiter=",")'''
Y = np.genfromtxt("mean_training_results_c20_a15_5_b10_d5_s2_pm.csv", delimiter=",")
for pt in X:
    ptToRaw(pt)
# np.savetxt("mean_training_c20_a15_5_b10_d5_s2_pm_raw.csv", training, delimiter=",")

lr = linear_model.LinearRegression()
logistic = linear_model.LogisticRegression()
lq = linear_model.LogisticRegression()
qr = make_pipeline(PolynomialFeatures(2), lr)
qrdg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=0.5, fit_intercept = False))
# cr = make_pipeline(PolynomialFeatures(3), lq)

# for clf, name in [(lr, 'Linear'), (qr, 'Quadratic'), (cr, "Cubic"), (logistic, 'Logistic')]:
for clf, name in [(qr, 'Quadratic'), (qrdg, "Qudratic Ridge")]:
    if name is 'Logistic':
        clf.fit(np.round(X), np.round(Y))
    else:
        clf.fit(X, Y)

training_y = qr.predict(X)
training_y_rdg = qrdg.predict(X)
training_avg_err = 0
training_avg_err_rdg = 0
for t in range(len(training_y)):
    training_avg_err += abs(training_y[t]-Y[t])
    training_avg_err_rdg += abs(training_y_rdg[t]-Y[t])
training_avg_err /= float(len(training_y))
training_avg_err_rdg /= float(len(training_y_rdg))

print training_avg_err
print training_avg_err_rdg

# test = np.genfromtxt('mean_test_X.csv', delimiter=',')
# test_true_y = np.genfromtxt('mean_test_Y.csv', delimiter=',')

'''test = np.ndarray(shape=(80,9))
test_true_y = []
for t in range(len(test)):
    for num in range(9):
        test[t, num] = np.random.randint(40) + 10
    temp_val = queryPoint(test[t])
    while (temp_val == -11):
        for num in range(9):
            test[t, num] = np.random.randint(40) + 10
        temp_val = queryPoint(test[t])
    test_true_y.append(temp_val)
np.savetxt("mean_test_X2.csv", test, delimiter=",")
np.savetxt("mean_test_Y2.csv", test_true_y, delimiter=",")'''
test = np.genfromtxt("mean_test_X2.csv", delimiter=",")
test_true_y = np.genfromtxt("mean_test_Y2.csv", delimiter=",")

for pt in test:
    ptToRaw(pt)

test_fn_y = qr.predict(test)
test_fn_y_rdg = qrdg.predict(test)
test_avg_err = 0
test_avg_err_rdg = 0
for t in range(len(test_true_y)):
    test_avg_err += abs(test_true_y[t]-test_fn_y[t])
    test_avg_err_rdg += abs(test_true_y[t]-test_fn_y_rdg[t])
test_avg_err /= float(len(test_true_y))
test_avg_err_rdg /= float(len(test_true_y))

training_diff = []
test_diff = []
training_diff_rdg = []
test_diff_rdg = []
for t in range(len(training_y)):
    training_diff.append(abs(training_y[t]-Y[t]))
    training_diff_rdg.append(abs(training_y_rdg[t]-Y[t]))

for t in range(len(test_true_y)):
    test_diff.append(abs(test_true_y[t]-test_fn_y[t]))
    test_diff_rdg.append(abs(test_true_y[t]-test_fn_y_rdg[t]))

# results_out = []
# results_out.append(training_avg_err)
# results_out.append(np.var(training_diff))
# results_out.append(max(training_diff))
# results_out.append(test_avg_err)
# results_out.append(np.var(test_diff))
# results_out.append(max(test_diff))
# np.savetxt("general_results.csv", results_out, delimiter=",")

temp_max = [30, 30, 30, 30, 30, 30, 30, 30, 30]
# original = queryPoint(temp_max)
changes_true = []
changes_fn = []
changes_ridge = []
diff = 15
# increase_true = []
# increase_fn = []
for i in range(len(temp_max)):
    temp = temp_max[:]
    temp[i] -= diff
    trial = queryPoint(temp)
    if (trial == -11):
        print ("BROKEN")
        exit()
    ptToRaw(temp)
    changes_true.append(trial)
    changes_fn.append(qr.predict(temp))
    changes_ridge.append(qrdg.predict(temp))
# for i in range(len(temp_max)):
#     temp = temp_max[:]
#     temp[i] += diff
#     trial = queryPoint(temp)
#     if (trial == -11):
#         print ("BROKEN")
#         exit()
#     increase_true.append(trial)
#     increase_fn.append(qr.predict(temp))
print "Original True"
# print original
print "Original Function"
ptToRaw(temp_max)
print qr.predict(temp_max)
print qrdg.predict(temp_max)
print "Decrease True:"
print changes_true
print "Decrease Function:"
print changes_fn
print changes_ridge

# print "Increase True:"
# print increase_true
# print "Increase Function:"
# print increase_fn

print "General Stats:"
print training_avg_err
print np.var(training_diff)
print max(training_diff)
print test_avg_err
print np.var(test_diff)
print max(test_diff)
print training_avg_err_rdg
print np.var(training_diff_rdg)
print max(training_diff_rdg)
print test_avg_err_rdg
print np.var(test_diff_rdg)
print max(test_diff_rdg)

#
# all15 = []
# for u in range(9):
#     all15.append(30)
# ptToRaw(all15)
# print all15
# l = 0
# while (not np.array_equal(X[l],all15)):
#     l+=1
# print(Y[l])
# print(qr.predict(all15))
# print len(X)
# print training_diff[-1]

# print lq.coef_
# print "train diff"
# sorted_train = sorted(training_diff)
# for i in range(10):
#     nth = -(i+1)
#     # print sorted_test[nth]
#     print X[training_diff.index(sorted_train[nth])]
#     print Y[training_diff.index(sorted_train[nth])]
#     print qr.predict(X[training_diff.index(sorted_train[nth])])
# print "test diff"
# sorted_test = sorted(test_diff)
# for i in range(10):
#     nth = -(i+1)
#     # print sorted_test[nth]
#     print test[test_diff.index(sorted_test[nth])]
#     print test_true_y[test_diff.index(sorted_test[nth])]
#     print qr.predict(test[test_diff.index(sorted_test[nth])].reshape(1,-1))


'''base = [10, 10, 10, 10, 10, 10, 10, 10, 10]
base_y = qr.predict(base)
base.append(base_y)

knee_pts = np.ndarray(shape=(len(training_y)+len(test_fn_y),10))

ind = 0
for i in range(len(training_y)):
    for n in range(9):
        knee_pts[ind, n] = X[i, n]
    knee_pts[ind, 9] = qr.predict(X[i])
    ind += 1
for i in range(len(test_fn_y)):
    for n in range(9):
        knee_pts[ind, n] = test[i, n]
    knee_pts[ind, 9] = qr.predict(test[i])
    ind += 1
min_dist = [scp.spatial.distance.euclidean(s, base) for s in knee_pts]
min_ind = min_dist.index(min(min_dist))
print "pre-normalization"
print knee_pts[min_ind]

mycopy = knee_pts.copy()
mycopy = normalize(mycopy, axis=0)
min_dist = [scp.spatial.distance.euclidean(s, base) for s in mycopy]
min_ind = min_dist.index(min(min_dist))
print "post-normalization"
print knee_pts[min_ind] '''
