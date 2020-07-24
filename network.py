import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.optim


class RegressionNN(nn.Module):

	def __init__(self, DataManager):

		super(RegressionNN, self).__init__()

		outdim_layer0 = 2048
		outdim_layer1 = 1024
		outdim_layer2 = 512
		outdim_layer3 = 128

		self.activation = torch.sigmoid
		# self.activation = F.relu
		# self.activation = F.elu
		# self.activation = torch.tanh

		self.dropout = nn.Dropout

		self.fully_connected_0 = nn.Linear(DataManager.num_input_features, outdim_layer0)
		self.fully_connected_1 = nn.Linear(outdim_layer0, outdim_layer1)
		self.fully_connected_2 = nn.Linear(outdim_layer1, outdim_layer2)
		self.fully_connected_3 = nn.Linear(outdim_layer2, outdim_layer3)
		self.fully_connected_final = nn.Linear(outdim_layer3, 1)

		self.bn_00 = nn.BatchNorm1d(DataManager.num_input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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

		x = self.fully_connected_final(x)

		return x

class ClassificationNN(nn.Module):

	def __init__(self, DataManager):

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

		self.fully_connected_0 = nn.Linear(DataManager.num_input_features, outdim_layer0)
		self.fully_connected_1 = nn.Linear(outdim_layer0, outdim_layer1)
		self.fully_connected_2 = nn.Linear(outdim_layer1, outdim_layer2)
		self.fully_connected_3 = nn.Linear(outdim_layer2, outdim_layer3)
		self.fully_connected_final = nn.Linear(outdim_layer3, 2)

		self.bn_00 = nn.BatchNorm1d(DataManager.num_input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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

		x = self.softmax(self.fully_connected_final(x), dim=1)

		return x

class NetworkTrainer():

	def __init__(self, DataManager, target_label, outdir, task, epochs=2001, learning_rate=4e-5, batch_size=512, validation_freq=10):


		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.validation_freq = validation_freq
		self.roc_df = []
		self.roc_df_train = []
		self.target_label = target_label
		self.outdir = outdir
		self.task = task

		try: os.makedirs(self.outdir + self.target_label + '/')
		except FileExistsError: pass
		if self.task == 'classification':
			try: os.makedirs(self.outdir + self.target_label + '/roc/')
			except FileExistsError: pass
			try: os.makedirs(self.outdir + self.target_label + '/roc_train/')
			except FileExistsError: pass
		try: os.makedirs(self.outdir + self.target_label + '/predictions/')
		except FileExistsError: pass
		try: os.makedirs(self.outdir + self.target_label + '/predictions_train/')
		except FileExistsError: pass

		self.last_train_epoch_loss = 0.
		self.train_loss_vec = []
		self.val_loss_vec = []
		self.epoch_counter_train = 0
		self.epoch_counter_val = []

		if self.task == 'classification':
			self.model = ClassificationNN(DataManager)
		if self.task == 'regression':
			self.model = RegressionNN(DataManager)
			self.scaler_lo_icu = DataManager.scaler_lo_icu
			self.scaler_lo_hospital = DataManager.scaler_lo_hospital

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

	def evaluate(self, y_true, predictions, validation_loss):

			if self.task == 'classification':

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


				# if self.epoch_counter_train % 100 == 0:
				# 	print('\nclassification results on validation set for recall approx.', \
				# 		set_recall, ' target: ', self.target_label, '\n', pd.DataFrame(self.roc_df), '\n')

				plt.figure(figsize=(10,10))
				plt.title('epoch ' + str(self.epoch_counter_train) + ' | auroc ' + str(np.round(100*roc_auc, 2)))
				plt.plot(fp_rate, tp_rate, c='darkgreen', linewidth=4, alpha=.6)
				plt.grid()
				plt.xlim(0.,1.)
				plt.ylim(0.,1.)
				plt.xlabel('FalsePositives [%]')
				plt.ylabel('TruePositives [%]')
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

				performance_x_vec = np.linspace(0, self.epoch_counter_train, len(pd.DataFrame(self.roc_df)))
				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['auroc'], c='darkgreen', label='auroc', linewidth=4, alpha=.6)
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['selectivity'], c='darkred', label='selectivity @.9recall', linewidth=4, alpha=.6)
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['precision'], c='orange', label='precision @.9recall', linewidth=4, alpha=.6)
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['accuracy'], c='darkblue', label='accuracy @.9recall', linewidth=4, alpha=.6)
				plt.xlabel('epochs')
				plt.ylabel('[%]')
				plt.title('Validation Performance Metrics for ' + self.target_label)
				plt.grid()
				plt.ylim(50.,100.)
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/performance.pdf')
				plt.close()

			if self.task == 'regression':
				

				# y_true = np.ravel(np.reshape(y_true, (-1,1)))
				# predictions = np.ravel(np.reshape(predictions, (-1,1)))

				y_true = np.nan_to_num(y_true)
				predictions = np.nan_to_num(predictions)

				if self.target_label == 'length_of_icu':
					y_true = np.ravel(
						np.exp(
							self.scaler_lo_icu.inverse_transform(
								np.reshape(y_true, (-1,1)))))
					predictions = np.ravel(
						np.exp(
							self.scaler_lo_icu.inverse_transform(
								np.reshape(predictions, (-1,1)))))

				if self.target_label == 'length_of_stay':
					y_true = np.ravel(
						np.exp(
							self.scaler_lo_hospital.inverse_transform(
								np.reshape(y_true, (-1,1)))))
					predictions = np.ravel(
						np.exp(
							self.scaler_lo_hospital.inverse_transform(
								np.reshape(predictions, (-1,1)))))

				mse = mean_squared_error(y_true, predictions)
				r2 = r2_score(y_true, predictions)
				mae = mean_absolute_error(y_true, predictions)
				error_var = np.var(np.abs(y_true - predictions))
				explained_var = explained_variance_score(y_true, predictions)

				if r2 < 0.: r2 = 0.
				if explained_var < 0.: explained_var = 0.


				self.roc_df.append({
					'epoch': self.epoch_counter_train,
					'train_loss': np.round(self.last_train_epoch_loss, 5),
					'val_loss': np.round(validation_loss.detach().numpy(), 5),
					'mse': np.round(mse, 2),
					'r2': np.round(r2, 2),
					'mae': np.round(mae, 2),
					'error_var': np.round(error_var, 2),
					'explained_var': np.round(explained_var, 2),
					})


				plt.figure(figsize=(12,12))
				plt.title('epoch ' + str(self.epoch_counter_train) + ' | mae ' + str(np.round(mae, 2)) + ' | r2 ' + str(np.round(r2, 2)))
				plt.scatter(y_true, predictions, c='darkgreen', s=16, alpha=.4)
				plt.xscale('log')
				plt.yscale('log')
				if self.target_label == 'length_of_icu':
					plt.xlim(1., 1000.)
					plt.ylim(1., 1000.)
				if self.target_label == 'length_of_stay':
					plt.xlim(1., 2000.)
					plt.ylim(1., 2000.)
				plt.grid(which='both')
				plt.xlabel('Labels [hours spent in ICU]')
				plt.ylabel('Predictions [hours spent in ICU]')
				plt.savefig(self.outdir + self.target_label + '/predictions/' + self.target_label + '_epoch_' + str(self.epoch_counter_train) + '.pdf')
				plt.close()

				performance_x_vec = np.linspace(0, self.epoch_counter_train, len(pd.DataFrame(self.roc_df)))

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['mse'], c='darkgreen', label='mse', linewidth=4, alpha=.6)
				plt.yscale('log')
				plt.xlabel('epochs')
				plt.ylabel('MSE Loss')
				plt.title('Mean Squared Error')
				plt.ylim(1e2,1e5)
				plt.grid(which='both')
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/mse.pdf')
				plt.close()

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['r2'], c='darkgreen', label='r2', linewidth=4, alpha=.6)
				plt.xlabel('epochs')
				plt.ylabel('R Squared')
				plt.title('R Squared')
				plt.grid()
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/r2.pdf')
				plt.close()

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['mae'], c='darkgreen', label='mae', linewidth=4, alpha=.6)
				plt.yscale('log')
				plt.xlabel('epochs')
				plt.ylabel('Mean Absolute Error [hours spent in ICU]')
				plt.title('Mean Absolute Error')
				plt.ylim(10.,100.)
				plt.yscale('log')
				plt.grid(which='both')
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/mae.pdf')
				plt.close()

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df)['explained_var'], c='darkgreen', label='explained_var', linewidth=4, alpha=.6)
				plt.xlabel('epochs')
				plt.ylabel('Explained Variance')
				plt.title('Explained Variance')
				plt.grid()
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/explained_var.pdf')
				plt.close()


			train_loss_plot = np.asarray(self.train_loss_vec)
			train_loss_axis = np.arange(self.epoch_counter_train)
			val_loss_plot = np.asarray(self.val_loss_vec)
			val_loss_axis = np.asarray(self.epoch_counter_val)

			plt.figure()
			plt.plot(val_loss_axis, val_loss_plot, c='orange', label='validation loss', linewidth=4, alpha=.6)
			plt.plot(train_loss_axis, train_loss_plot, c='darkgreen', label='training loss', linewidth=4, alpha=.6)
			plt.yscale('log')
			plt.xlabel('epochs')
			plt.ylabel('crossentropy loss')
			plt.title('training progress')
			plt.grid(which='both')
			plt.legend()
			plt.savefig(self.outdir + self.target_label + '/training.pdf')
			plt.close()


			self.val_loss_vec.append(validation_loss.detach().numpy())
			self.epoch_counter_val.append(self.epoch_counter_train)

	def evaluate_training(self, y_true, predictions, train_loss):

			if self.task == 'classification':

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

				self.roc_df_train.append({
					'epoch': self.epoch_counter_train,
					'train_loss': np.round(self.last_train_epoch_loss, 5),
					'val_loss': np.round(train_loss.detach().numpy(), 5),
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


				# if self.epoch_counter_train % 100 == 0:
				# 	print('\nclassification results on validation set for recall approx.', \
				# 		set_recall, ' target: ', self.target_label, '\n', pd.DataFrame(self.roc_df), '\n')

				plt.figure(figsize=(10,10))
				plt.title('epoch ' + str(self.epoch_counter_train) + ' | auroc ' + str(np.round(100*roc_auc, 2)))
				plt.plot(fp_rate, tp_rate, c='darkgreen', linewidth=4, alpha=.6)
				plt.grid()
				plt.xlim(0.,1.)
				plt.ylim(0.,1.)
				plt.xlabel('FalsePositives [%]')
				plt.ylabel('TruePositives [%]')
				plt.savefig(self.outdir + self.target_label + '/roc_train/' + self.target_label + '_epoch_' + str(self.epoch_counter_train) + '.pdf')
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
				plt.savefig(self.outdir + self.target_label + '/predictions_train/' + self.target_label + '_epoch_' + str(self.epoch_counter_train) + '.pdf')
				plt.close()

				performance_x_vec = np.linspace(0, self.epoch_counter_train, len(pd.DataFrame(self.roc_df_train)))
				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['auroc'], c='darkgreen', label='auroc', linewidth=4, alpha=.6)
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['selectivity'], c='darkred', label='selectivity @.9recall', linewidth=4, alpha=.6)
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['precision'], c='orange', label='precision @.9recall', linewidth=4, alpha=.6)
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['accuracy'], c='darkblue', label='accuracy @.9recall', linewidth=4, alpha=.6)
				plt.xlabel('epochs')
				plt.ylabel('[%]')
				plt.title('Validation Performance Metrics for ' + self.target_label)
				plt.grid()
				plt.ylim(50.,100.)
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/performance_train.pdf')
				plt.close()

			if self.task == 'regression':

				# y_true = np.ravel(np.reshape(y_true, (-1,1)))
				# predictions = np.ravel(np.reshape(predictions, (-1,1)))

				y_true = np.nan_to_num(y_true)
				predictions = np.nan_to_num(predictions)

				if self.target_label == 'length_of_icu':
					y_true = np.ravel(
						np.exp(
							self.scaler_lo_icu.inverse_transform(
								np.reshape(y_true, (-1,1)))))
					predictions = np.ravel(
						np.exp(
							self.scaler_lo_icu.inverse_transform(
								np.reshape(predictions, (-1,1)))))

				if self.target_label == 'length_of_stay':
					y_true = np.ravel(
						np.exp(
							self.scaler_lo_hospital.inverse_transform(
								np.reshape(y_true, (-1,1)))))
					predictions = np.ravel(
						np.exp(
							self.scaler_lo_hospital.inverse_transform(
								np.reshape(predictions, (-1,1)))))

				mse = mean_squared_error(y_true, predictions)
				r2 = r2_score(y_true, predictions)
				mae = mean_absolute_error(y_true, predictions)
				error_var = np.var(np.abs(y_true - predictions))
				explained_var = explained_variance_score(y_true, predictions)

				if r2 < 0.: r2 = 0.
				if explained_var < 0.: explained_var = 0.

				self.roc_df_train.append({
					'epoch': self.epoch_counter_train,
					'train_loss': np.round(self.last_train_epoch_loss, 5),
					'val_loss': np.round(train_loss.detach().numpy(), 5),
					'mse': np.round(mse, 2),
					'r2': np.round(r2, 2),
					'mae': np.round(mae, 2),
					'error_var': np.round(error_var, 2),
					'explained_var': np.round(explained_var, 2),
					})


				plt.figure(figsize=(12,12))
				plt.title('epoch ' + str(self.epoch_counter_train) + ' | mae ' + str(np.round(mae, 2)) + ' | r2 ' + str(np.round(r2, 2)))
				plt.scatter(y_true, predictions, c='darkgreen', s=16, alpha=.4)
				plt.xscale('log')
				plt.yscale('log')
				if self.target_label == 'length_of_icu':
					plt.xlim(1., 1000.)
					plt.ylim(1., 1000.)
				if self.target_label == 'length_of_stay':
					plt.xlim(1., 2000.)
					plt.ylim(1., 2000.)
				plt.grid(which='both')
				plt.xlabel('Labels [hours spent in ICU]')
				plt.ylabel('Predictions [hours spent in ICU]')
				plt.savefig(self.outdir + self.target_label + '/predictions_train/' + self.target_label + '_epoch_' + str(self.epoch_counter_train) + '.pdf')
				plt.close()

				performance_x_vec = np.linspace(0, self.epoch_counter_train, len(pd.DataFrame(self.roc_df_train)))

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['mse'], c='darkgreen', label='mse', linewidth=4, alpha=.6)
				plt.yscale('log')
				plt.xlabel('epochs')
				plt.ylabel('MSE Loss')
				plt.title('Mean Squared Error')
				plt.ylim(1e2,1e5)
				plt.grid(which='both')
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/mse_train.pdf')
				plt.close()

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['r2'], c='darkgreen', label='r2', linewidth=4, alpha=.6)
				plt.xlabel('epochs')
				plt.ylabel('R Squared')
				plt.title('R Squared')
				plt.grid()
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/r2_train.pdf')
				plt.close()

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['mae'], c='darkgreen', label='mae', linewidth=4, alpha=.6)
				plt.yscale('log')
				plt.xlabel('epochs')
				plt.ylabel('Mean Absolute Error [hours spent in ICU]')
				plt.title('Mean Absolute Error')
				plt.ylim(10.,100.)
				plt.yscale('log')
				plt.grid(which='both')
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/mae_train.pdf')
				plt.close()

				plt.figure()
				plt.plot(performance_x_vec, pd.DataFrame(self.roc_df_train)['explained_var'], c='darkgreen', label='explained_var', linewidth=4, alpha=.6)
				plt.xlabel('epochs')
				plt.ylabel('Explained Variance')
				plt.title('Explained Variance')
				plt.grid()
				plt.legend()
				plt.savefig(self.outdir + self.target_label + '/explained_var_train.pdf')
				plt.close()

	def validate(self):

		validation_loss = 0.
		validation_label_cache, validation_prediction_cache = [], []
		dummy = 0

		with torch.no_grad():

			for local_batch, local_labels in self.validation_generator:


				output = self.model.forward(local_batch, is_training=False)
				

				local_labels = torch.transpose(local_labels, 0, 1)

				if self.task == 'classification':
					output = torch.transpose(output, 0, 1)
					loss = F.binary_cross_entropy(output, local_labels.type(torch.float))
				if self.task == 'regression':
					# loss = F.mse_loss(torch.reshape(output, (1,-1)), torch.reshape(local_labels.type(torch.float)[0,:], (1,-1)))
					# output = torch.transpose(output, 0, 1)
					loss = torch.mean(
						torch.sqrt(
							F.mse_loss(
								torch.reshape(output, (1,-1)), 
								torch.reshape(local_labels.type(torch.float)[0,:], (1,-1)),
								reduction = 'none')
							))

				validation_loss += (loss / len(self.validation_generator))

				if dummy == 0:
					if self.task == 'classification':
						validation_label_cache = np.reshape(np.asarray(local_labels.detach().numpy()), (-1,2))
						validation_prediction_cache = np.reshape(np.asarray(output.detach().numpy()), (-1,2))
					if self.task == 'regression':
						validation_label_cache = np.reshape(np.asarray(local_labels[0,:].detach().numpy()), (-1,1))
						validation_prediction_cache = np.reshape(np.asarray(output.detach().numpy()), (-1,1))
					dummy += 1
				else:
					if self.task == 'classification':
						validation_label_cache = np.concatenate((validation_label_cache, np.reshape(np.asarray(local_labels.detach().numpy()), (-1,2))), axis=0)
						validation_prediction_cache = np.concatenate((validation_prediction_cache, np.reshape(np.asarray(output.detach().numpy()), (-1,2))), axis=0)
					if self.task == 'regression':
						validation_label_cache = np.concatenate((validation_label_cache, np.reshape(np.asarray(local_labels[0,:].detach().numpy()), (-1,1))), axis=0)
						validation_prediction_cache = np.concatenate((validation_prediction_cache, np.reshape(np.asarray(output.detach().numpy()), (-1,1))), axis=0)


			if self.task == 'classification':
				y_true = np.reshape(np.asarray(validation_label_cache), (-1,2))[:,0]
				predictions = np.reshape(np.asarray(validation_prediction_cache), (-1,2))[:,0]
			if self.task == 'regression':
				y_true = np.reshape(np.asarray(validation_label_cache), (-1,1))
				predictions = np.reshape(np.asarray(validation_prediction_cache), (-1,1))

			y_true[y_true != y_true] = 0.
			predictions[predictions != predictions] = 0.


			# print('y_true')
			# print(y_true.shape)
			# print('predictions')
			# print(predictions.shape)

			self.evaluate(y_true, predictions, validation_loss)

	def train(self):

		# TODO built early stopping criterion

		print('\n\n\nstarting training for ' + str(self.epochs) + ' epochs on label: ' + self.target_label + '\n')
		pbar = tqdm(total=self.epochs)

		for _ in range(self.epochs):

			epoch_loss = 0.
			dummy_train = 0

			for local_batch, local_labels in self.training_generator:

				self.optimizer.zero_grad()

				output = self.model(local_batch.type(torch.float))
				# output = torch.transpose(output, 0, 1)

				local_labels = torch.transpose(local_labels, 0, 1)

				if self.task == 'classification':
					output = torch.transpose(output, 0, 1)
					loss = F.binary_cross_entropy(output, local_labels.type(torch.float))
				if self.task == 'regression':
					# loss = F.mse_loss(torch.reshape(output, (1,-1)), torch.reshape(local_labels.type(torch.float)[0,:], (1,-1)))
					# output = torch.transpose(output, 0, 1)
					loss = torch.mean(
						torch.sqrt(
							F.mse_loss(
								torch.reshape(output, (1,-1)), 
								torch.reshape(local_labels.type(torch.float)[0,:], (1,-1)),
								reduction = 'none')
							))


				epoch_loss += (loss / len(self.training_generator))

				loss.backward()
				self.optimizer.step()


				if self.epoch_counter_train % (1*self.validation_freq) == 0:
					if dummy_train == 0:
						if self.task == 'classification':
							train_label_cache = np.reshape(np.asarray(local_labels.detach().numpy()), (-1,2))
							train_prediction_cache = np.reshape(np.asarray(output.detach().numpy()), (-1,2))
						if self.task == 'regression':
							train_label_cache = np.reshape(np.asarray(local_labels[0,:].detach().numpy()), (-1,1))
							train_prediction_cache = np.reshape(np.asarray(output.detach().numpy()), (-1,1))
						dummy_train += 1
					else:
						if self.task == 'classification':
							train_label_cache = np.concatenate((train_label_cache, np.reshape(np.asarray(local_labels.detach().numpy()), (-1,2))), axis=0)
							train_prediction_cache = np.concatenate((train_prediction_cache, np.reshape(np.asarray(output.detach().numpy()), (-1,2))), axis=0)
						if self.task == 'regression':
							train_label_cache = np.concatenate((train_label_cache, np.reshape(np.asarray(local_labels[0,:].detach().numpy()), (-1,1))), axis=0)
							train_prediction_cache = np.concatenate((train_prediction_cache, np.reshape(np.asarray(output.detach().numpy()), (-1,1))), axis=0)


			if self.epoch_counter_train % (2*self.validation_freq) == 0:
				if self.task == 'classification':
					y_true_train = np.reshape(np.asarray(train_label_cache), (-1,2))[:,0]
					predictions_train = np.reshape(np.asarray(train_prediction_cache), (-1,2))[:,0]
				if self.task == 'regression':
					y_true_train = np.reshape(np.asarray(train_label_cache), (-1,1))
					predictions_train = np.reshape(np.asarray(train_prediction_cache), (-1,1))


			if self.epoch_counter_train % self.validation_freq == 0: 
				self.validate()
				if self.epoch_counter_train % (2*self.validation_freq) == 0: 
					self.evaluate_training(y_true_train, predictions_train, epoch_loss)


			self.train_loss_vec.append(epoch_loss.detach().numpy())
			self.last_train_epoch_loss = epoch_loss.detach().numpy()
			
			self.epoch_counter_train += 1

			pbar.update(1)
		pbar.close()

		print('\n\nfinished training\n\n')





