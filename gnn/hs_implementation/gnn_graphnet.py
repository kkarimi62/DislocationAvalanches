
import functools
from typing import Any, Dict, Text,Tuple,Optional


import enum
import pickle
from absl import logging
import numpy as np


import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import sonnet as snt

from graph_nets import graphs
from graph_nets import modules as gn_modules
from graph_nets import utils_tf
from gnn_graphnet_model import *

from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys

SEED =4441666
np.random.seed(SEED)
tf.set_random_seed(SEED)



pure_lab='' 
data_split = None
data_split_lab=''
if len(sys.argv[1:]):
	if sys.argv[1]=='pure':
		pure_lab='_pure'
		print('PURE Mg')
	else:
		data_split =float(sys.argv[1])
		data_split_lab='_'+str(data_split)
else:
	print('ALLOY Mg')


def base_graph(logtrans=False):
	"""
	
	This here loads the data and forms a graph structure. This should be implemented as it is dataset-dependent.
	Output should be:
		a dict with  globals (dummy), nodes (nodal data in numpy), edges (edge data), receivers (indices of receiving node in int), senders (int)
		train_mask   array of size (n_nodes) with bool or 0/1 indicating training nodes
		val_mask     same for validation nodes 
		test_mask    same for testing nodes
		target 	     here the array containing the nodal target data
		weight	     if needed, a weight parameter given for the (training) nodes
	"""
	return {"globals": [0.0],  "nodes": nodes_data[:,1:], "edges": edges_data[:,2:],  "receivers": edges_data[:,1].astype(int), "senders": edges_data[:,0].astype(int)  }, train_mask, val_mask, test_mask, target, weight 


def create_loss_ops(target_op, output_op, mask, weight=None):
	"""Create supervised loss operations from targets and outputs.
	
	Args:
	  target_op: The target tf.Tensor.
	  output_ops: The output graph from the model.
	
	Returns:
	  loss values (tf.Tensor)
	"""
	if weight is None:
		loss_op = tf.reduce_mean(  (  tf.boolean_mask(output_op.nodes, mask) - tf.boolean_mask(target_op, mask))**2)
	else:
		loss_op = tf.reduce_mean( tf.boolean_mask(weight, mask)* (  tf.boolean_mask(output_op.nodes, mask) - tf.boolean_mask(target_op, mask))**2)
	
	return loss_op


def create_corr_ops(target_op, output_op, mask):
	corr_op = tfp.stats.correlation(tf.boolean_mask(target_op, mask), tf.boolean_mask(output_op.nodes, mask))
	return corr_op


num_processing_steps_tr = 3
# Data / training parameters.
num_training_iterations = 20000
learning_rate = 7.5e-6


static_graph_tr, train_mask_np, val_mask_np, test_mask_np, target_nodes_np, weight_np = base_graph(logtrans=data_split)
print("NUM PROCESSING STEPS ", num_processing_steps_tr)
print("LEARNING RATE ", learning_rate)
print(static_graph_tr.keys())
for k in static_graph_tr.keys():
	try:
		print(k, static_graph_tr[k].shape)
	except AttributeError:
		print(k)


input_graph = utils_tf.data_dicts_to_graphs_tuple([static_graph_tr])

#print(input_graph)

train_mask = tf.constant(train_mask_np, dtype=tf.bool)
test_mask  = tf.constant(test_mask_np , dtype=tf.bool)
val_mask   = tf.constant(val_mask_np  , dtype=tf.bool)

target_nodes = tf.constant(target_nodes_np)
weight = tf.constant(weight_np)
weight = None ##NOTE comment out if weights wanted 

model = EncodeProcessDecode(node_output_size=1)
#print(model.summary())


output_ops, latent_ops = model(input_graph, num_processing_steps_tr) #[-1]

loss_op_tr = []
loss_op_va = []
loss_op_ts = []
corr_op_tr = []
corr_op_va = []
corr_op_ts = []

for op in output_ops:
	loss_op_tr.append( create_loss_ops(target_nodes, op, train_mask,weight))
	loss_op_va.append( create_loss_ops(target_nodes, op,   val_mask,weight))
	loss_op_ts.append( create_loss_ops(target_nodes, op,  test_mask,weight))
	corr_op_tr.append( create_corr_ops(target_nodes, op, train_mask))
	corr_op_va.append( create_corr_ops(target_nodes, op,   val_mask))
	corr_op_ts.append( create_corr_ops(target_nodes, op,  test_mask))


loss_op_tr_sum = sum(loss_op_tr) / num_processing_steps_tr


# Optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr_sum)


training_history = np.zeros((num_training_iterations, 3*num_processing_steps_tr)) 
correlat_history = np.zeros((num_training_iterations, 3*num_processing_steps_tr)) 
counter = 0

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(snt.format_variables(model.variables))

best_corr = np.zeros((num_processing_steps_tr,3))
best_val_loss = np.inf
best_corr_loss = 0.0
best_val_loss_all = np.inf*np.ones(num_processing_steps_tr)
best_corr_loss_all = np.zeros(num_processing_steps_tr)
last_improved = 0
early_stopping_crit = 500


measure_val_by_loss= True
print("MEASURE VALIDATION BY LOSS ", measure_val_by_loss)

for iteration in range(num_training_iterations):
	last_iteration = iteration
	train_values = sess.run({
	    "step": step_op,
	    "loss": loss_op_tr,
	    "outputs": output_ops,
	    "latents": latent_ops,
	    "corr": corr_op_tr

	})


	test_values = sess.run({
		"loss_val":  loss_op_va,
		"loss_test": loss_op_ts,
	    	"corr_val": corr_op_va,
	    	"corr_test": corr_op_ts
	})
	training_history[counter, 0:num_processing_steps_tr] = train_values['loss']
	training_history[counter, num_processing_steps_tr:2*num_processing_steps_tr] = test_values['loss_val']
	training_history[counter, 2*num_processing_steps_tr:] = test_values['loss_test']

	correlat_history[counter, 0:num_processing_steps_tr] = train_values['corr']
	correlat_history[counter, num_processing_steps_tr:2*num_processing_steps_tr] = test_values['corr_val']
	correlat_history[counter, 2*num_processing_steps_tr:] = test_values['corr_test']



	if (iteration+1) %10==0:
		print("# {:05d}, training {:.4f}, validation {:.4f}, test {:.4f}".format(iteration+1,training_history[counter,num_processing_steps_tr-1], training_history[counter,2*num_processing_steps_tr-1],training_history[counter,-1]  ))

		for i in range(num_processing_steps_tr):
			if measure_val_by_loss:
				cond =  (training_history[counter,num_processing_steps_tr+i] < best_val_loss_all[i])
				cond_best = (training_history[counter,num_processing_steps_tr+i] < best_val_loss)
			else:
				cond =  (correlat_history[counter,num_processing_steps_tr+i] > best_corr_loss_all[i])
				cond_best = (correlat_history[counter,num_processing_steps_tr+i] > best_corr_loss)
			if cond:
				step_output =  sess.run(output_ops[i].nodes) # sess.run(output_ops)	
				best_corr[i,0] = stats.pearsonr( step_output[train_mask_np].flatten(),  target_nodes_np[train_mask_np].flatten() )[0]
				best_corr[i,1] = stats.pearsonr( step_output[val_mask_np].flatten(),  target_nodes_np[val_mask_np].flatten() )[0]
				best_corr[i,2] = stats.pearsonr( step_output[test_mask_np].flatten(),  target_nodes_np[test_mask_np].flatten() )[0]
				#print("      best val res, r: training {:.4f}, validation {:.4f}, test {:.4f}".format( best_corr[0], best_corr[1], best_corr[2]  ))
				best_val_loss_all[i] = training_history[counter,num_processing_steps_tr+i]
				best_corr_loss_all[i] = correlat_history[counter,num_processing_steps_tr+i]
				if cond_best:
					best_output = np.copy(step_output)
					best_latent = sess.run(latent_ops[i])
					#print(best_latent.shape)
					best_val_loss = training_history[counter,num_processing_steps_tr+i]
					best_corr_loss = correlat_history[counter,num_processing_steps_tr+i]
					last_improved = counter
	counter+=1 
	if counter > last_improved + early_stopping_crit:
		print('NO IMPROVEMENT IN {} STEPS, STOPPING TRAINING...'.format(int(early_stopping_crit)))
		break


f_label = "{}_{}_{}{}".format(pure_lab, learning_rate,num_processing_steps_tr,data_split_lab)

training_history = training_history[:counter]
correlat_history = correlat_history[:counter]
for i in range(num_processing_steps_tr):
	print("    {} steps:  best val res, r: training {:.4f}, validation {:.4f}, test {:.4f}".format(i+1, best_corr[i,0], best_corr[i,1], best_corr[i,2]  ))
	latest_output =  sess.run(output_ops[i].nodes) # sess.run(output_ops)	
	np.savetxt("/scratch/work/salmenh1/gnn/training_results_deepmind/graphnet_all_proc_latest_pred_{}_{}.dat".format(f_label,i+1), latest_output)



np.savetxt(   "/scratch/work/salmenh1/gnn/training_results_deepmind/graphnet_all_proc_training_history_{}.dat".format(f_label), training_history)
np.savetxt("/scratch/work/salmenh1/gnn/training_results_deepmind/graphnet_all_proc_correlation_history_{}.dat".format(f_label), correlat_history)
np.savetxt(          "/scratch/work/salmenh1/gnn/training_results_deepmind/graphnet_all_proc_best_pred_{}.dat".format(f_label), best_output)
np.save(           "/scratch/work/salmenh1/gnn/training_results_deepmind/graphnet_all_proc_best_latent_{}.npy".format(f_label), best_latent)








