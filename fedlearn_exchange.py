# Description: Federated learning by model exchange
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
	'mydata_path_processed' : '../mydata/processed_data_consolidated.csv',
	'datapath_processed' : '../mydata/processed_data.csv',
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
	'batch_size' : 128,
	'test_batch_size' : 1000,
	'lr' : 0.001,
	'log_interval' : 20,
	'epochs' : 100,
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

# Generate federated data from the original eICU dataset
# loads orignal CSV files and builds combined DataFrame. uncomment if not yet available
# eICU_DataLoader(eICU_path, mydata_path, num_patients=args[number_of_included_patients])
# DataProcessor(mydata_path, mydata_path_processed)


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



# Create virtual workers

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

# Build the model and optimizer
logging.info("Build the client model")
input_dim = x.shape[1]
output_dim = y.shape[1]
model = ClassificationNN(input_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=args['lr'])


roc_df = []


# Training loop
def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	#iterate over federated data
	for batch_idx, (data, target) in enumerate(train_loader):

		#send the model to the remote location
		model = model.send(data.location)
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)

		# this loss is a pointer to the tensor loss 
		# at the remote location
		# print(output.dtype)
		loss = F.nll_loss(output, target)

		# call backward() on the loss pointer,
		# that will send the command to call
		# backward on the actual loss tensor
		# present on the remote machine
		loss.backward()

		optimizer.step()

		# retrieve the model to the local 
		# client and remove it from the 
		# distant client
		model.get()

		if batch_idx % args['log_interval'] == 0:

			# a thing to note is the variable loss was
			# also created at remote worker, so we need to
			# explicitly get it back
			loss = loss.get()

			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch,
					batch_idx * args['batch_size'], # no of images done
					len(train_loader) * args['batch_size'], # total images left
					100. * batch_idx / len(train_loader), 
					loss.item()
				)
			)

def test(model, device, test_loader, epoch, roc_df):
	model.eval()
	test_loss = 0
	correct = 0
	dummy = 0
	softmax = nn.Softmax(dim=1)

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			# add losses together
			test_loss += F.nll_loss(output, target, reduction='sum').item() 

			# get the index of the max probability class
			pred = output.argmax(dim=1, keepdim=True)  
			correct += pred.eq(target.view_as(pred)).sum().item()

			if dummy == 0:
				preds_bin = np.reshape(pred.numpy(), (-1))
				preds = np.reshape(softmax(output).numpy()[:,1], (-1))
				labels = np.reshape(target.numpy(), (-1))
				dummy += 1
			else:
				preds_bin = np.concatenate((preds_bin, np.reshape(pred.numpy(), (-1))), axis=0)
				preds = np.concatenate((preds, np.reshape(softmax(output).numpy()[:,1], (-1))), axis=0)
				labels = np.concatenate((labels, np.reshape(target.numpy(), (-1))), axis=0)

	# if args['split_strategy'] != 'trainNminus1_test1':
	test_loss /= len(test_loader.dataset)

	preds_bin = np.reshape(preds_bin, (-1))
	preds = np.reshape(preds, (-1))
	labels = np.reshape(labels, (-1))


	roc_df = evaluate(labels, preds, preds_bin, test_loss, epoch, roc_df)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

	return roc_df

def evaluate(y_true, predictions, predictions_binary, validation_loss, epoch_counter_train, roc_df):
  
	fp_rate, tp_rate, thresholds = roc_curve(y_true, predictions)

	roc_auc = auc(fp_rate, tp_rate)

	y_true = np.ravel(np.nan_to_num(y_true))
	predictions = np.ravel(np.nan_to_num(predictions))

	roc_dummy = pd.DataFrame({
		'fp_rate': fp_rate,
		'tp_rate': tp_rate,
		'threshold': thresholds,
		'tp_dummy': tp_rate,
		})

	set_recall = .9
	roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] > set_recall] = 0.
	roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] < roc_dummy['tp_dummy'].max()] = 0.
	roc_dummy['tp_dummy'] /= roc_dummy['tp_dummy'].max()
	roc_dummy = np.asarray(roc_dummy['threshold'].loc[roc_dummy['tp_dummy'] > .5].values)
	
	if roc_dummy.shape[0] > 1: roc_dummy = roc_dummy[0]

	recall_threshold = roc_dummy

	# predictions_binary = predictions.copy()
	# predictions_binary[predictions_binary >= recall_threshold] = 1.
	# predictions_binary[predictions_binary < recall_threshold] = 0.

	num_true_positives = np.sum(np.abs(predictions_binary) * np.abs(y_true))
	num_false_positives = np.sum(np.abs(predictions_binary) * np.abs(1. - y_true))
	num_true_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(1. - y_true))
	num_false_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(y_true))

	num_total_positives = num_true_positives + num_false_negatives
	num_total_negatives = num_true_negatives + num_false_positives

	num_total_positives_predicted = np.sum(np.abs(predictions_binary))

	recall = num_true_positives / num_total_positives
	selectivity = num_true_negatives / num_total_negatives
	precision = num_true_positives / (num_true_positives + num_false_positives)
	accuracy = (num_true_positives + num_true_negatives) / (num_total_positives + num_total_negatives)
	f1score = (2 * num_true_positives) / (2 * num_true_positives + num_false_positives + num_false_negatives)
	informedness = recall + selectivity - 1.

	roc_df.append({
		'epoch': epoch_counter_train,
		# 'train_loss': np.round(self.last_train_epoch_loss, 5),
		'val_loss': np.round(validation_loss, 5),
		'auroc': np.round(100*roc_auc, 2),
		'recall': np.round(100*recall, 2),
		'selectivity': np.round(100*selectivity, 2),
		'precision': np.round(100*precision, 2),
		'accuracy': np.round(100*accuracy, 2),
		'f1score': np.round(100*f1score, 2),
		'informedness': np.round(100*informedness, 2),
		'#TP': num_true_positives,
		'#FP': num_false_positives,
		'#TN': num_true_negatives,
		'#FN': num_false_negatives,
		})

	plt.figure(figsize=(10,10))
	plt.title('epoch ' + str(epoch_counter_train) + ' | auroc ' + str(np.round(100*roc_auc, 2)))
	plt.plot(fp_rate, tp_rate, c='darkgreen', linewidth=4, alpha=.6)
	plt.grid()
	plt.xlim(0.,1.)
	plt.ylim(0.,1.)
	plt.xlabel('FalsePositives [%]')
	plt.ylabel('TruePositives [%]')
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/roc/' + args['target_label'] + '_epoch_' + str(epoch_counter_train) + '.pdf')
	plt.close()

	performance_x_vec = np.linspace(0, epoch_counter_train, len(pd.DataFrame(roc_df)))
	plt.figure()
	plt.plot(performance_x_vec, pd.DataFrame(roc_df)['auroc'], c='darkgreen', label='auroc', linewidth=4, alpha=.6)
	plt.plot(performance_x_vec, pd.DataFrame(roc_df)['selectivity'], c='darkred', label='selectivity @.9recall', linewidth=4, alpha=.6)
	plt.plot(performance_x_vec, pd.DataFrame(roc_df)['precision'], c='orange', label='precision @.9recall', linewidth=4, alpha=.6)
	plt.plot(performance_x_vec, pd.DataFrame(roc_df)['accuracy'], c='darkblue', label='accuracy @.9recall', linewidth=4, alpha=.6)
	plt.xlabel('epochs')
	plt.ylabel('[%]')
	plt.title('Validation Performance Metrics for ' + args['target_label'])
	plt.grid()
	plt.ylim(50.,100.)
	plt.legend()
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/performance.pdf')
	plt.close()

	plt.figure()
	plt.scatter(np.arange(len(predictions[y_true == 0])), predictions[y_true == 0], c='darkred', alpha=.4)
	plt.scatter(np.arange(len(predictions[y_true == 1])), predictions[y_true == 1], c='darkgreen', alpha=.4)
	plt.grid()
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/predictions.pdf')
	plt.close()

	plt.figure()
	plt.scatter(np.arange(len(predictions)), predictions, c='darkred', alpha=.4)
	plt.grid()
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/preds.pdf')
	plt.close()

	plt.figure()
	plt.scatter(np.arange(len(predictions_binary)), predictions_binary, c='darkred', alpha=.4)
	plt.grid()
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/preds_binary.pdf')
	plt.close()

	plt.figure()
	plt.scatter(np.arange(len(y_true)), y_true, c='darkred', alpha=.4)
	plt.grid()
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/y_true.pdf')
	plt.close()

	plt.figure()
	plt.plot(performance_x_vec, pd.DataFrame(roc_df)['val_loss'], c='darkgreen', label='val loss', linewidth=4, alpha=.6)
	plt.grid()
	plt.savefig(args['OUTPATH'] + args['target_label'] + '/val_loss.pdf')
	plt.close()

	print(pd.DataFrame(roc_df))

	return roc_df

def validate(mode, validation_loader):
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

		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			validation_loss, correct, len(validation_loader.dataset),
			100. * correct / len(validation_loader.dataset)))


for epoch in range(1, args['epochs'] + 1):
		train(args, model, device, fdataloader, optimizer, epoch)
		roc_df = test(model, device, test_loader, epoch, roc_df)