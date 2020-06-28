import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.optim


class ClassificationNN(nn.Module):

	def __init__(self, DataManager):

		super(ClassificationNN, self).__init__()


		self.fully_connected_0 = nn.Linear(DataManager.num_input_features, 1024)
		self.fully_connected_1 = nn.Linear(1024, 1024)
		self.fully_connected_2 = nn.Linear(1024, 512)
		self.fully_connected_3 = nn.Linear(512, 128)

		self.fully_connected_final = nn.Linear(128, 2)

		self.relu = F.relu
		self.elu = F.elu
		self.sigmoid = sigmoid
		self.softmax = F.softmax
		self.dropout = nn.Dropout

		torch.nn.init.xavier_uniform_(self.fully_connected_0.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_1.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_2.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_3.weight)
		torch.nn.init.xavier_uniform_(self.fully_connected_final.weight)


	def forward(self, x, is_training=True):
		# TODO dropout, other activations

		x = self.sigmoid(self.fully_connected_0(x.float()))
		x = F.dropout(x, p=0.2, training=is_training, inplace=False)

		x = self.sigmoid(self.fully_connected_1(x))
		x = F.dropout(x, p=0.2, training=is_training, inplace=False)

		x = self.sigmoid(self.fully_connected_2(x))
		x = F.dropout(x, p=0.2, training=is_training, inplace=False)

		x = self.sigmoid(self.fully_connected_3(x))
		x = F.dropout(x, p=0.2, training=is_training, inplace=False)

		x = self.softmax(self.fully_connected_final(x), dim=1)

		return x


class NetworkTrainer():

	def __init__(self, DataManager, target_label, outdir, epochs=250, learning_rate=1e-4, batch_size=512, validation_freq=10):


		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.validation_freq = validation_freq
		self.roc_df = []
		self.target_label = target_label
		self.outdir = outdir

		try: os.makedirs(self.outdir + self.target_label + '/')
		except FileExistsError: pass
		try: os.makedirs(self.outdir + self.target_label + '/roc/')
		except FileExistsError: pass
		try: os.makedirs(self.outdir + self.target_label + '/predictions/')
		except FileExistsError: pass

		self.last_train_epoch_loss = 0.
		self.train_loss_vec = []
		self.val_loss_vec = []
		self.epoch_counter_train = 0
		self.epoch_counter_val = []


		self.model = ClassificationNN(DataManager)
		# self.criterion = nn.CrossEntropyLoss(reduction='none')
		# self.criterion = F.binary_cross_entropy()

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=self.learning_rate, 
			betas=(0.9, 0.999), 
			eps=1e-08, 
			weight_decay=0, 
			amsgrad=False)



		self.training_generator = DataManager.get_train_iterator(self.batch_size, self.target_label)
		self.validation_generator = DataManager.get_test_iterator(self.batch_size, self.target_label)


		self.train()


	def validate(self):

		validation_loss = 0.
		validation_label_cache, validation_prediction_cache = [], []
		dummy = 0

		for local_batch, local_labels in self.validation_generator:


			output = self.model.forward(local_batch, is_training=False)

			output = torch.transpose(output, 0, 1)
			local_labels = torch.transpose(local_labels, 0, 1)

			# print('----> output', output.shape)
			# print('----> local_labels', local_labels.shape)
			# print()

			loss = F.binary_cross_entropy(output, local_labels.type(torch.float))
			# loss = self.criterion(output, local_labels.type(torch.long))
			# loss = torch.mean(loss)
			validation_loss += (loss / len(self.validation_generator))

			if dummy == 0:
				validation_label_cache = np.reshape(np.asarray(local_labels.detach().numpy()), (-1,2))
				validation_prediction_cache = np.reshape(np.asarray(output.detach().numpy()), (-1,2))
				dummy += 1
			else:
				validation_label_cache = np.concatenate((validation_label_cache, np.reshape(np.asarray(local_labels.detach().numpy()), (-1,2))), axis=0)
				validation_prediction_cache = np.concatenate((validation_prediction_cache, np.reshape(np.asarray(output.detach().numpy()), (-1,2))), axis=0)


			# validation_label_cache.append(local_labels.detach().numpy())
			# validation_prediction_cache.append(output.detach().numpy())

		y_true = np.reshape(np.asarray(validation_label_cache), (-1,2))[:,0]
		predictions = np.reshape(np.asarray(validation_prediction_cache), (-1,2))[:,0]

		# print('y_true')
		# print(y_true.shape)
		# print('predictions')
		# print(predictions.shape)


		fp_rate, tp_rate, thresholds = roc_curve(y_true, predictions)

		roc_auc = auc(fp_rate, tp_rate)

		y_true = np.ravel(y_true)
		predictions = np.ravel(predictions)


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


		predictions_binary = predictions.copy()
		predictions_binary[predictions_binary >= recall_threshold] = 1.
		predictions_binary[predictions_binary < recall_threshold] = 0.

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

		self.roc_df.append({
			'epoch': self.epoch_counter_train,
			'train_loss': np.round(self.last_train_epoch_loss, 5),
			'val_loss': np.round(validation_loss.detach().numpy(), 5),
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


		if self.epoch_counter_train % 100 == 0:
			print('\nclassification results on validation set for recall approx.', \
				set_recall, ' target: ', self.target_label, '\n', pd.DataFrame(self.roc_df), '\n')

		plt.figure()
		plt.title('epoch ' + str(self.epoch_counter_train))
		plt.plot(fp_rate, tp_rate)
		plt.grid()
		plt.xlim(0.,1.)
		plt.ylim(0.,1.)
		plt.xlabel('FalsePositives')
		plt.ylabel('TruePositives')
		plt.savefig(self.outdir + self.target_label + '/roc/' + self.target_label + '_epoch_' + str(self.epoch_counter_train) + '.pdf')
		plt.close()

		plt.figure()
		plt.title('epoch ' + str(self.epoch_counter_train))
		plt.scatter(np.arange(len(predictions[y_true == 1.])), predictions[y_true == 1.], label='positives', c='darkgreen', alpha=.6)
		plt.scatter(np.arange(len(predictions[y_true == 0.])), predictions[y_true == 0.], label='negatives', c='orange', alpha=.6)
		plt.grid()
		plt.ylim(0.,1.)
		plt.legend()
		plt.xlabel('samples')
		plt.ylabel('predictions')
		plt.savefig(self.outdir + self.target_label + '/predictions/' + self.target_label + '_epoch_' + str(self.epoch_counter_train) + '.pdf')
		plt.close()

		train_loss_plot = np.asarray(self.train_loss_vec)
		train_loss_axis = np.arange(self.epoch_counter_train)
		val_loss_plot = np.asarray(self.val_loss_vec)
		val_loss_axis = np.asarray(self.epoch_counter_val)

		plt.figure(figsize=(16,16))
		plt.plot(val_loss_axis, val_loss_plot, c='orange', label='validation loss')
		plt.plot(train_loss_axis, train_loss_plot, c='darkgreen', label='training loss')
		plt.yscale('log')
		plt.xlabel('epochs')
		plt.ylabel('crossentropy loss')
		plt.title('training progress')
		plt.grid()
		plt.legend()
		plt.savefig(self.outdir + self.target_label + '/training.pdf')
		plt.close()

		performance_x_vec = np.linspace(0, self.epoch_counter_train, len(pd.DataFrame(self.roc_df)))
		plt.figure(figsize=(16,16))
		plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['auroc'], c='darkgreen', label='auroc', linewidth=4, alpha=.6)
		plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['selectivity'], c='darkred', label='selectivity', linewidth=4, alpha=.6)
		plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['precision'], c='orange', label='precision', linewidth=4, alpha=.6)
		plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['accuracy'], c='darkblue', label='accuracy', linewidth=4, alpha=.6)
		plt.xlabel('epochs')
		plt.ylabel('[%]')
		plt.title('Validation Performance Metrics over Training Epochs')
		plt.grid()
		plt.legend()
		plt.savefig(self.outdir + self.target_label + '/performance.pdf')
		plt.close()

		self.val_loss_vec.append(validation_loss.detach().numpy())
		self.epoch_counter_val.append(self.epoch_counter_train)

		# print('validation score @epoch' + str(self.epoch_counter_train).ljust(25, '.') + str(np.round(validation_loss.detach().numpy(), 5)))


	def train(self):

		# TODO built early stopping criterion

		print('\n\n\nstarting training for ' + str(self.epochs) + ' epochs\n')

		for _ in range(self.epochs):

			epoch_loss = 0.

			for local_batch, local_labels in self.training_generator:

				self.optimizer.zero_grad()

				output = self.model(local_batch.type(torch.float))

				output = torch.transpose(output, 0, 1)
				local_labels = torch.transpose(local_labels, 0, 1)

				# loss = self.criterion(output, local_labels.type(torch.long))
				loss = F.binary_cross_entropy(output, local_labels.type(torch.float))

				# epoch_loss += loss / self.batch_size
				epoch_loss += (loss / len(self.training_generator))


				loss.backward()
				self.optimizer.step()

			if self.epoch_counter_train % self.validation_freq == 0:
				self.validate()
				# self.evaluate()

			self.train_loss_vec.append(epoch_loss.detach().numpy())
			self.epoch_counter_train += 1
			self.last_train_epoch_loss = epoch_loss.detach().numpy()

		print('\n\nfinished training\n\n')





