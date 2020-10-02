# Copyright (C) 2020  Arash Mehrjou, Max Planck Institute for Intelligent Systems
# Copyright (C) 2020  Arash Mehrjou, ETH Zürich

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import logging
from data_management import DataManager

class MLP_classifier(nn.Module):
	
	def __init__(self, input_size, outpath):
		super(MLP_classifier, self).__init__()
		
		self.fc1 = nn.Linear(input_size, 512)
		self.dp = nn.Dropout(0.2)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, 2)

		torch.nn.init.xavier_uniform_(self.fc1.weight)
		torch.nn.init.xavier_uniform_(self.fc2.weight)
		torch.nn.init.xavier_uniform_(self.fc3.weight)
		torch.nn.init.xavier_uniform_(self.fc4.weight)

		self.writer = SummaryWriter(log_dir=outpath)
	
	def forward(self, x):

		x = self.dp(self.fc1(x))
		x = F.sigmoid(x)
		x = self.dp(self.fc2(x))
		x = F.sigmoid(x)
		x = self.dp(self.fc3(x))
		x = F.sigmoid(x)
		x = self.fc4(x)
		x = F.log_softmax(x, dim=1)

		return x

	def write_data(self):

		self.writer.add_scalar('gradients/fc1', self.fc1.weight.grad)

class ClassificationNN(nn.Module):

	def __init__(self, input_size):

		super(ClassificationNN, self).__init__()

		outdim_layer0 = 2048
		outdim_layer1 = 1024
		outdim_layer2 = 512
		outdim_layer3 = 128

		self.activation = torch.sigmoid
		# self.activation = F.relu
		# self.activation = F.elu
		# self.activation = torch.tanh

		self.dropout = nn.Dropout
		self.softmax = F.softmax

		self.fully_connected_0 = nn.Linear(input_size, outdim_layer0)
		self.fully_connected_1 = nn.Linear(outdim_layer0, outdim_layer1)
		self.fully_connected_2 = nn.Linear(outdim_layer1, outdim_layer2)
		self.fully_connected_3 = nn.Linear(outdim_layer2, outdim_layer3)
		self.fully_connected_final = nn.Linear(outdim_layer3, 2)

		self.bn_00 = nn.BatchNorm1d(input_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bn_01 = nn.BatchNorm1d(outdim_layer0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bn_12 = nn.BatchNorm1d(outdim_layer1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bn_23 = nn.BatchNorm1d(outdim_layer2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bn_3final = nn.BatchNorm1d(outdim_layer3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

		torch.nn.init.xavier_uniform_(self.fully_connected_0.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_1.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_2.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_3.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_final.weight)


	def forward(self, x, is_training=True):
		
		x = self.bn_00(x.float())

		x = self.activation(self.fully_connected_0(x))
		x = self.bn_01(x)
		x = F.dropout(x, p=0.5, training=is_training, inplace=False)

		x = self.activation(self.fully_connected_1(x))
		x = self.bn_12(x)
		x = F.dropout(x, p=0.5, training=is_training, inplace=False)

		x = self.activation(self.fully_connected_2(x))
		x = self.bn_23(x)
		x = F.dropout(x, p=0.5, training=is_training, inplace=False)

		x = self.activation(self.fully_connected_3(x))
		x = self.bn_3final(x)
		x = F.dropout(x, p=0.5, training=is_training, inplace=False)

		x = F.log_softmax(self.fully_connected_final(x), dim=1)

		return x


def get_data_from_DataManager(path, target_labels):
	eICU_data = DataManager(
	path, target_labels)
	print(eICU_data.training_data.keys())
	return eICU_data.training_data['x_full'], eICU_data.training_data['y_full']