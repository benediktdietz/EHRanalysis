# Description: Federated learning by model aggregation
#  Copyright (C) 2020  Arash Mehrjou, Max Planck Institute for Intelligent Systems
# Copyright (C) 2020  Arash Mehrjou, ETH Zürich


# Python packages
import numpy as np
import pandas as pd
import random
import cv2
import os
from imutils import paths
from matplotlib import pyplot as plt
import logging
import re
from os import listdir
from os.path import isfile, join
from syft.frameworks.torch import fl

# Sklearn packages
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
from federated_utils import ClassificationNN, get_data_from_DataManager
from torch.utils.data import TensorDataset, DataLoader
# Hospital packages
from data_management import eICU_DataLoader, DataProcessor, DataSetIterator, DataManager

args = {
	# 'mydata_path_processed' : '../mydata/nomeds_20k_processed.csv',
	'mydata_path_processed' : '../mydata/newset_processed.csv',
	'datapath_processed' : '../mydata/newset_processed.csv',
	'test_data_path' : '../mydata/federated/hospital_59.csv',
	'validation_data_path' : '../mydata/federated/hospital_59.csv',
	'federated_path' : '../mydata/federated',
	'OUTPATH' : '../results/FL_24_9_5/',
	'train_split' : .7,
	'create_new_federated_data' : True,
	'num_of_included_patients' : 20000,
	'client_ids' : "all", # or 'all' to load all client from the federated_path
	'n_clients' : 10,
	'use_cuda' : False,
	'batch_size' : 32,
	'test_batch_size' : 1000,
	'lr' : 0.001,
	'log_interval' : 5,
	'epochs' : 100,
    'aggregation_iters' : 10,
    'worker_iters' : 5,
	'task' : 'classification',
	'predictive_attributes' : ["length_of_stay", "will_die"],
	'target_attributes' : ["will_die"],
	'target_label' : "will_die",
	# 'split_strategy' : 'trainN_testN', #'trainNminus1_test1'
	'split_strategy' : 'trainNminus1_test1', #'trainNminus1_test1'
	'test_hospital_id' : 73 #'trainNminus1_test1'
}


datapath_processed = args["datapath_processed"]
federated_path = args["federated_path"]
OUTPATH = args["OUTPATH"]

use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


try:
	os.makedirs(args['OUTPATH'] + args['target_label'] + '/')
	os.makedirs(args['OUTPATH'] + args['target_label'] + '/roc/')
except FileExistsError: pass


eICU_data = DataManager(args)
available_client_IDs = eICU_data.sampling_df['hospital_id'].unique()
print('available_client_IDs: ', available_client_IDs, available_client_IDs.shape, available_client_IDs.dtype)
if args['split_strategy'] == 'trainNminus1_test1':
	available_client_IDs = available_client_IDs[available_client_IDs != args['test_hospital_id']]
	print('available_client_IDs: ', available_client_IDs)


# Create virtual workers as data owners and a secure worker
hook = sy.TorchHook(torch)
virtual_workers = {}
if type(args["client_ids"]) is list:
	client_ids =  args["client_ids"]
	for client_id in client_ids:
		virtual_workers["hospital_{}".format(client_id)] = sy.VirtualWorker(hook, id="hospital_{}".format(client_id))
else:
	client_ids = []
	fnames = [f for f in listdir(args["federated_path"]) if isfile(join(args["federated_path"], f)) and re.search("hospital", f)]
	for f in fnames:
		id = int(re.search(r"\d\w+", f).group())
		client_ids.append(id)

	client_ids = list(available_client_IDs)

	for client_id in client_ids:
		virtual_workers["hospital_{}".format(client_id)] = sy.VirtualWorker(hook, id="hospital_{}".format(client_id))
secure_worker = sy.VirtualWorker(hook, id="secure_worker")


# Make Syft federated dataset
client_datapair_dict = {}
datasets = []

logging.info("Load federated dataset")
for client_id in client_ids:
	tmp_path = federated_path + '/hospital_' + str(client_id) + '.csv'
	# x, y = get_data_from_DataManager(args["target_attributes"], args)
	x, y = eICU_data.get_train_data_from_hopital(client_id)
	client_datapair_dict["hospital_{}".format(client_id)] = (x, y)
#     client_data_list.append((pd.read_csv(federated_path + '/hospital_' + str(client_id) + '.csv')[predictive_attributes], )

for client_id in client_ids:
	tmp_tuple = client_datapair_dict["hospital_{}".format(client_id)]
	datasets.append(
		fl.BaseDataset(torch.tensor(tmp_tuple[0], dtype=torch.float32), torch.tensor(tmp_tuple[1].squeeze(), dtype=torch.long))
		.send(virtual_workers["hospital_{}".format(client_id)]))

fed_dataset = sy.FederatedDataset(datasets)
fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=args["batch_size"])


# Load test data
# x, y = get_data_from_DataManager(args["target_attributes"], args)
if args['split_strategy'] == 'trainN_testN':
	x, y = eICU_data.get_full_test_data()
if args['split_strategy'] == 'trainNminus1_test1':
	x, y = eICU_data.get_test_data_from_hopital(args['test_hospital_id'])
x_pt = torch.tensor(x, dtype=torch.float32) # transform to torch tensor
y_pt = torch.tensor(y.squeeze(), dtype=torch.long)
my_dataset = TensorDataset(x_pt, y_pt) # create your datset
test_loader = DataLoader(my_dataset, batch_size=10) # create your dataloader


# Load validation data
# x, y = get_data_from_DataManager(args["target_attributes"], args)
if args['split_strategy'] == 'trainN_testN':
	x, y = eICU_data.get_full_val_data()
if args['split_strategy'] == 'trainNminus1_test1':
	x, y = eICU_data.get_val_data_from_hopital(args['test_hospital_id'])
x_pt = torch.tensor(x, dtype=torch.float32) # transform to torch tensor
y_pt = torch.tensor(y.squeeze(), dtype=torch.long)
my_dataset = TensorDataset(x_pt, y_pt) # create your datset
validation_loader = DataLoader(my_dataset, batch_size=10) # create your dataloader


# Train local models in parallel
local_data = {}
local_target = {}
for worker_id in virtual_workers.keys():
    local_data[worker_id] = torch.tensor(client_datapair_dict[worker_id][0]).send(virtual_workers[worker_id])
    local_target[worker_id] = torch.tensor(client_datapair_dict[worker_id][1].squeeze(), dtype=torch.long).send(virtual_workers[worker_id])


def average_models(list_of_models, target_model):
    """take a list of models with the same architecture
    and avergae their weigths to produce a new model with
    a similar architecture."""
    state_dict = target_model.state_dict()
    for param_name in target_model.state_dict().keys():
        if list_of_models[0].state_dict()[param_name].location == target_model.location:
            
            avg_param = 0
            for model in list_of_models:
                avg_param += model.state_dict()[param_name]
            avg_param = avg_param / len(virtual_workers)
            state_dict[param_name] = avg_param
            target_model.load_state_dict(state_dict)
    return target_model


# Build the model and optimizer
logging.info("Build the client model")
input_dim = x.shape[1]
output_dim = y.shape[1]
model = ClassificationNN(input_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)


def validate(model, validation_loader):
	validation_loss = 0.
	validation_label_cache, validation_prediction_cache = [], []
	dummy = 0
	with torch.no_grad():
		model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in validation_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)

				# add losses together
				validation_loss += F.nll_loss(output, target, reduction='sum').item() 

				# get the index of the max probability class
				pred = output.argmax(dim=1, keepdim=True)  
				correct += pred.eq(target.view_as(pred)).sum().item()


				if dummy == 0:
					validation_label_cache = np.reshape(np.asarray(target.detach().numpy()), (-1,1))
					validation_prediction_cache = np.reshape(np.asarray(output.detach().numpy()), (-1,2))
					dummy += 1
				else:
					validation_label_cache = np.concatenate((validation_label_cache, np.reshape(np.asarray(target.detach().numpy()), (-1,1))), axis=0)
					validation_prediction_cache = np.concatenate((validation_prediction_cache, np.reshape(np.asarray(output.detach().numpy()), (-1,2))), axis=0)

		validation_loss /= len(validation_loader.dataset)

		print('\nValidation set: Validation loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			validation_loss, correct, len(validation_loader.dataset),
			100. * correct / len(validation_loader.dataset)))


# Training and validation loop
for a_iter in range(args["aggregation_iters"]):
    
    # Send the model to workers
    local_models = {}
    local_optimizers = {}
    for worker_id in virtual_workers.keys():
        local_models[worker_id] = model.copy().send(virtual_workers[worker_id])
        local_optimizers[worker_id] = optim.SGD(params=local_models[worker_id].parameters(),lr=args["lr"])

    # Train local models in parallel
    for wi in range(args["worker_iters"]):
        local_worker_pred = {}
        local_worker_loss = {}
        for worker_id in virtual_workers.keys():
            local_optimizers[worker_id].zero_grad()
            local_worker_pred[worker_id] = local_models[worker_id](local_data[worker_id])
            local_worker_loss[worker_id] = F.nll_loss(local_worker_pred[worker_id], local_target[worker_id])
            local_worker_loss[worker_id].backward()
            local_optimizers[worker_id].step()
            local_worker_loss[worker_id] = local_worker_loss[worker_id].get().data
    # Send models to the secure worker
    for worker_id in virtual_workers.keys():
        local_models[worker_id].move(secure_worker)
#     print(list(model.parameters())[0])
    model = average_models(list(local_models.values()), model.send(secure_worker)).get()
    validate(model, validation_loader)
#     print(list(model.parameters())[0])





