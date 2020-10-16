import numpy as np
import pandas as pd
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from fl_implementation_utils import *
from data_management import eICU_DataLoader, DataProcessor, DataSetIterator, DataManager


def create_tf_dataset_for_client_fn(client_id, federated_path):

	# a function which takes a client_id and returns a
	# tf.data.Dataset for that client

	label_cols = [
		# doesnt make sense to include or not properly formatted cols
		'patient_id',
		'health_system_id',
		'corr_id',
		'data_set_ref',
		# 'medication_ids',
		# 'drug_strings_prescribed',
		# 'drug_codes_prescribed',
		# 'diagnosis_string',
		'pasthistory_notetypes',
		'pasthistory_values',
		'hospital_discharge_year_2014',
		'hospital_discharge_year_2015',
		# labels we want to predict or shouldnt be available for our predictions
		'length_of_stay',
		'icu_admission_time',
		'length_of_icu',
		'icu_discharge',
		'diagnosis_offset',
		'diagnosis_activeUponDischarge',
		# 'diagnosis_ids',
		# 'diagnosis_priority',
		'diagnosis_ICD9code',
		'unit_discharge_offset',
		'unit_discharge_status_Alive',
		'unit_discharge_status_Expired',
		'unit_discharge_location_Death',
		'unit_discharge_location_Floor',
		'unit_discharge_location_Home',
		# 'unit_discharge_location_Other',
		'unit_discharge_location_Other External',
		'unit_discharge_location_Other Hospital',
		'unit_discharge_location_Other ICU',
		# 'unit_discharge_location_Rehabilitation',
		'unit_discharge_location_Skilled Nursing Facility',
		'unit_discharge_location_Step-Down Unit (SDU)',
		# 'unit_discharge_location_Telemetry',
		'unit_stay_type_admit',
		'unit_stay_type_readmit',
		'unit_stay_type_stepdown/other',
		'unit_stay_type_transfer',
		'hospital_discharge_offset',
		'hospital_discharge_status_Alive',
		'hospital_discharge_status_Expired',
		'will_return',
		'will_die',
		'will_readmit',
		'will_stay_long',
		'visits_current_stay',
		'unit_readmission',
		'survive_current_icu',
		'lab_type_ids',
		'lab_names',
		]

	client_data = pd.read_csv(federated_path + '/hospital_' + str(client_id) + '.csv')

	# print(client_data)
	num_cols = feature_map.shape[1]
	feature_map = client_data.drop(columns = label_cols).fillna(0.)
	feature_map = np.nan_to_num(feature_map)


	print('*******', feature_map.shape)

	labels = client_data[self.target_label]

	# client_data = df[df[client_id_colname] == client_id]

	dataset = tf.data.Dataset.from_tensor_slices((feature_map, labels))
	# dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict('list'))
	dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).repeat(NUM_EPOCHS)

	print('dataset:\n', dataset)

	return dataset

class MLP_classifier:
	@staticmethod
	def build(shape, classes, args):
		model = Sequential()

		# model.add(BatchNormalization())
		model.add(Dense(args.layer_width_0, input_shape=(shape,)))
		model.add(Dropout(rate=.5))
		model.add(Activation("sigmoid"))

		# model.add(BatchNormalization())
		model.add(Dense(args.layer_width_1))
		model.add(Dropout(rate=.5))
		model.add(Activation("sigmoid"))

		# model.add(BatchNormalization())
		model.add(Dense(args.layer_width_2))
		model.add(Dropout(rate=.5))
		model.add(Activation("sigmoid"))

		# model.add(BatchNormalization())
		model.add(Dense(args.layer_width_3))
		model.add(Dropout(rate=.5))
		model.add(Activation("sigmoid"))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model
	

class FederatedLearner():

	def __init__(self, args):

		self.args = args
		self.datapath = self.args.mydata_path_processed
		self.datapath_fed = self.args.datapath_federated
		self.figure_path = self.args.outdir
		self.target_label = self.args.target_label

		filelist = os.listdir(self.datapath_fed)
		if '.DS_Store' in filelist: 
			filelist.remove('.DS_Store')
		if '' in filelist: 
			filelist.remove('')

		# self.available_dataset_ids = [int(float(dataid[9:-4])) for dataid in filelist]
		# self.available_dataset_ids_train = self.available_dataset_ids[:-5]
		# self.available_dataset_ids_test = self.available_dataset_ids[-5:]
		
		self.available_dataset_ids = [73]
		self.available_dataset_ids_train = [176]
		self.available_dataset_ids_test = [73, 122, 188, 176, 165]

		self.train_epoch = 0
		self.roc_df = []
		self.result_df_test = []
		self.result_df_train = []

		# self.eICU_data = DataManager(self.datapath, 
			# ['length_of_stay',
			#  'length_of_icu',
			#  'will_return',
			#  'will_die',
			#  'will_readmit',
			#  'will_stay_long',
			#  'unit_readmission',
			#  'survive_current_icu'])
		# self.X_train, self.y_train = self.eICU_data.get_train_data(target_label)
		# self.X_test, self.y_test = self.eICU_data.get_test_data(target_label)
		# self.feature_dim = self.X_train.shape[1]
		# clients = self.get_hospital_clients(self.X_train, self.y_train, num_clients=10, initial='client')
		


		clients = self.get_hospital_clients(self.available_dataset_ids_train)
		#process and batch the training data for each client
		self.clients_batched = dict()
		for (client_name, data) in clients.items():
			self.clients_batched[client_name] = batch_data(data)
		
		clients_test = self.get_hospital_clients(self.available_dataset_ids_test)
		self.clients_batched_test = dict()
		for (client_name, data) in clients_test.items():
			self.clients_batched_test[client_name] = batch_data(data)
		
		self.feature_dim = self.get_hospital_data(self.available_dataset_ids[0], return_feature_dim=True)
		# print('\nfeature map dimension: ', self.feature_dim)

		#initialize global model
		if self.args.network_kind == 'classification':
			self.global_model = MLP_classifier().build(self.feature_dim, 2, self.args)
		if self.args.network_kind == 'regression':
			self.global_model = MLP_classifier().build(self.feature_dim, 1, self.args)
	

	def get_hospital_clients(self, id_list):

		#create a list of client names
		client_names = ['{}_{}'.format('hospital', i) for i in id_list]

		shards = [self.get_hospital_data(i) for i in id_list]

		#number of clients must equal number of shards
		assert(len(shards) == len(client_names))

		return {client_names[i] : shards[i] for i in range(len(client_names))}


	def get_hospital_data(self, client_id, return_feature_dim=False):

		label_cols = [
			# doesnt make sense to include or not properly formatted cols
			'patient_id',
			'health_system_id',
			'corr_id',
			'data_set_ref',
			# 'medication_ids',
			# 'drug_strings_prescribed',
			# 'drug_codes_prescribed',
			# 'diagnosis_string',
			'pasthistory_notetypes',
			'pasthistory_values',
			'hospital_discharge_year_2014',
			'hospital_discharge_year_2015',
			# labels we want to predict or shouldnt be available for our predictions
			'length_of_stay',
			'icu_admission_time',
			'length_of_icu',
			'icu_discharge',
			'diagnosis_offset',
			'diagnosis_activeUponDischarge',
			# 'diagnosis_ids',
			# 'diagnosis_priority',
			'diagnosis_ICD9code',
			'unit_discharge_offset',
			'unit_discharge_status_Alive',
			'unit_discharge_status_Expired',
			'unit_discharge_location_Death',
			'unit_discharge_location_Floor',
			'unit_discharge_location_Home',
			# 'unit_discharge_location_Other',
			'unit_discharge_location_Other External',
			'unit_discharge_location_Other Hospital',
			'unit_discharge_location_Other ICU',
			# 'unit_discharge_location_Rehabilitation',
			'unit_discharge_location_Skilled Nursing Facility',
			'unit_discharge_location_Step-Down Unit (SDU)',
			# 'unit_discharge_location_Telemetry',
			'unit_stay_type_admit',
			'unit_stay_type_readmit',
			'unit_stay_type_stepdown/other',
			'unit_stay_type_transfer',
			'hospital_discharge_offset',
			'hospital_discharge_status_Alive',
			'hospital_discharge_status_Expired',
			'will_return',
			'will_die',
			'will_readmit',
			'will_stay_long',
			'visits_current_stay',
			'unit_readmission',
			'survive_current_icu',
			'lab_type_ids',
			'lab_names',
			]

		client_data = pd.read_csv(self.datapath_fed + '/hospital_' + str(client_id) + '.csv')

		feature_map = client_data.drop(columns = label_cols).fillna(0.)
		feature_map = np.nan_to_num(feature_map)

		labels = pd.concat(
			[client_data[self.target_label], 1. - client_data[self.target_label]],
			axis = 1)
		labels = np.asarray(labels.values)

		# print('\nfeature_maps: ', feature_map.shape, '\n')

		dataset = list(zip(feature_map, labels))
		random.shuffle(dataset)

		if return_feature_dim:
			return feature_map.shape[1]
		else:
			return dataset


	def train(self):
					
		#create optimizer
		if self.args.network_kind == 'classification':
			loss=self.args.loss
			metrics = ['accuracy']
		optimizer = SGD(
			lr=self.args.learning_rate, 
			decay=lr / self.traing_rounds, 
			momentum=0.9)
		# optimizer = Adam(learning_rate = self.args.learning_rate)
										 
		#commence global training loop
		for comm_round in range(self.args.num_gobal_epochs):

			print('training federated round number ' + str(comm_round))
							
			# get the global model's weights - will serve as the initial weights for all local models
			global_weights = self.global_model.get_weights()
			
			#initial list to collect local model weights after scalling
			scaled_local_weight_list = list()

			#randomize client data - using keys
			client_names= list(self.clients_batched.keys())
			random.shuffle(client_names)
			
			#loop through each client and create new local model
			for client in client_names:

				smlp_local = MLP_classifier()

				if self.args.network_kind == 'classification':
					local_model = smlp_local.build(self.feature_dim, 2, self.args)
				if self.args.network_kind == 'regression':
					local_model = smlp_local.build(self.feature_dim, 1, self.args)

				local_model.compile(
					loss=loss, 
					optimizer=optimizer,
					metrics=metrics)
				
				#set local model weight to the weight of the global model
				local_model.set_weights(global_weights)

				#fit local model with client's data
				for local_epoch in range(4):

					for features, true_labels in self.clients_batched[client]:
					
						local_model.fit(features, true_labels, verbose=0)
				

				#scale the model weights and add to list
				scaling_factor = weight_scalling_factor(self.clients_batched, client)
				scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
				scaled_local_weight_list.append(scaled_weights)
				

			if self.train_epoch % self.args.validation_freq == 0:

				# print('-----------------')
				# print(local_model.evaluate(self.clients_batched, return_dict=True))
				# print('-----------------')


				result_df_test_dummy = pd.DataFrame(self.validate('test'))
				print('\nclassification results on validation set for target: ', self.target_label, '\n', pd.DataFrame(self.roc_df).round(2), '\n')

				result_df_train_dummy = pd.DataFrame(self.validate('train'))
				print('\nclassification results on training set for target: ', self.target_label, '\n', pd.DataFrame(self.roc_df).round(2), '\n')

				if self.train_epoch == 0:

					self.result_df_test = result_df_test_dummy['ce_loss']
					self.result_df_train = result_df_train_dummy['ce_loss']

				else:

					self.result_df_test =  pd.concat(
						[self.result_df_test, result_df_test_dummy['ce_loss']],
						axis = 1)

					self.result_df_train =  pd.concat(
						[self.result_df_train, result_df_train_dummy['ce_loss']],
						axis = 1)

				# print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
				# print(self.result_df_test)
				# print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

				if self.train_epoch > 0:

					result_array_train = np.asarray(self.result_df_train.values)
					plt.figure()
					for i in range(result_array_train.shape[0]-1):
						plt.plot(result_array_train[i,:])
					plt.plot(result_array_train[-1,:], label='total')
					plt.yscale('log')
					plt.grid()
					plt.savefig(self.figure_path + 'loss_train_epoch_' + str(self.train_epoch) + '.pdf')
					plt.close()

					result_array_test = np.asarray(self.result_df_test.values)
					plt.figure()
					for i in range(result_array_test.shape[0]-1):
						plt.plot(result_array_test[i,:])
					plt.plot(result_array_test[-1,:], label='total')
					plt.yscale('log')
					plt.grid()
					plt.savefig(self.figure_path + 'loss_test_epoch_' + str(self.train_epoch) + '.pdf')
					plt.close()

					
			#to get the average over all the local model, we simply take the sum of the scaled weights
			average_weights = sum_scaled_weights(scaled_local_weight_list)
			
			#update global model 
			self.global_model.set_weights(average_weights)

			self.train_epoch += 1

			#clear session to free memory after each communication round
			K.clear_session()


	def validate(self, mode):

		if mode == 'test':
			batched_client_data = self.clients_batched_test
		if mode == 'train':
			batched_client_data = self.clients_batched


	

		# get the global model's weights - will serve as the initial weights for all local models
		global_weights = self.global_model.get_weights()
		
		#initial list to collect local model weights after scalling
		scaled_local_weight_list = list()

		#randomize client data - using keys
		client_names = list(batched_client_data.keys())



		plt.figure(figsize=(16,16))
		plt.title('Classifier ROC ' + self.target_label)
		plt.plot([0,1],[0,1],'r--')

		self.roc_df = []
		total_result_dummy = 0
		#loop through each client and create new local model
		for client in client_names:
			
			# result_dummy, label_dummy = [0], [0]
			client_result_dummy = 0

			for features, true_labels in batched_client_data[client]:

				if client_result_dummy == 0:

					result_dummy = np.reshape(np.asarray(self.global_model.predict(features)), (-1,2))

					label_dummy = np.reshape(np.asarray(true_labels), (-1,2))

					client_result_dummy += 1

				else:

					result_dummy0 = np.reshape(np.asarray(self.global_model.predict(features)), (-1,2))
					result_dummy = np.concatenate((result_dummy, result_dummy0), axis=0)

					label_dummy0 = np.reshape(np.asarray(true_labels), (-1,2))
					label_dummy = np.concatenate((label_dummy, label_dummy0), axis=0)


			result_dummy = np.reshape(np.ravel(np.asarray(result_dummy)), (-1,2))
			label_dummy = np.reshape(np.ravel(np.asarray(label_dummy)), (-1,2))

			print('\nresult dimensions:\n', result_dummy.shape, label_dummy.shape)

			if total_result_dummy == 0:
				result_cache = result_dummy
				label_cache = label_dummy
				total_result_dummy += 1
			else:
				result_cache = np.concatenate((result_cache, result_dummy), axis=0)
				label_cache = np.concatenate((label_cache, label_dummy), axis=0)


			self.roc_analysis(result_dummy, label_dummy, client)

		self.roc_analysis(result_cache, label_cache, 'total')

		plt.legend(loc='lower right')
		plt.xlim([0., 1.])
		plt.ylim([0., 1.])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.grid()
		plt.savefig(self.figure_path + 'classification_' + mode + '/' + self.target_label + '_epoch_' + str(self.train_epoch) + '.pdf')
		plt.close()

		return self.roc_df


	def test_the_model(self, X_test, Y_test,  model, comm_round):

		cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

		#logits = model.predict(X_test, batch_size=100)
		logits = model.predict(X_test)

		loss = cce(Y_test, logits)
		acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))

		model_predictions = tf.argmax(logits, axis=1)
		model_labels = tf.argmax(Y_test, axis=1)

		print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))

		return acc, loss


	def roc_analysis(self, predictions, y_true, client_tag, set_recall = .9):

		def make_ints(input):
			return int(input)
		int_maker = np.vectorize(make_ints)

		y_true = y_true[:,0]
		y_true = np.reshape(y_true, (-1,1))
		y_true = int_maker(y_true)

		predictions = predictions[:,0]
		predictions = np.reshape(predictions, (-1,1))

		try:
			cross_entropy_loss = log_loss(y_true, predictions)
		except ValueError:
			cross_entropy_loss = 1.

		# cross_entropy_loss = log_loss(y_true, predictions)

		fp_rate, tp_rate, thresholds = roc_curve(y_true, predictions)
		roc_auc = auc(fp_rate, tp_rate)

		roc_dummy = pd.DataFrame({
			'fp_rate': fp_rate,
			'tp_rate': tp_rate,
			'threshold': thresholds,
			'tp_dummy': tp_rate,
			})

		roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] > set_recall] = 0.
		roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] < roc_dummy['tp_dummy'].max()] = 0.
		roc_dummy['tp_dummy'] /= roc_dummy['tp_dummy'].max()
		roc_dummy = np.asarray(roc_dummy['threshold'].loc[roc_dummy['tp_dummy'] > .5].values)
		
		if roc_dummy.shape[0] > 1: roc_dummy = roc_dummy[0]

		recall_threshold = roc_dummy

		predictions_binary = predictions
		predictions_binary[predictions_binary >= recall_threshold] = 1.
		predictions_binary[predictions_binary < recall_threshold] = 0.

		num_true_positives = np.sum(np.abs(predictions_binary) * np.abs(y_true))
		num_false_positives = np.sum(np.abs(predictions_binary) * np.abs(1. - y_true))
		num_true_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(1. - y_true))
		num_false_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(y_true))

		num_total_positives = num_true_positives + num_false_negatives + 1e-5
		num_total_negatives = num_true_negatives + num_false_positives + 1e-5

		num_total_positives_predicted = np.sum(np.abs(predictions_binary))

		recall = num_true_positives / num_total_positives
		selectivity = num_true_negatives / num_total_negatives
		precision = num_true_positives / (num_true_positives + num_false_positives)
		accuracy = (num_true_positives + num_true_negatives) / (num_total_positives + num_total_negatives)
		f1score = (2 * num_true_positives) / (2 * num_true_positives + num_false_positives + num_false_negatives)
		informedness = recall + selectivity - 1.

		self.roc_df.append({
			'client': client_tag,
			'ce_loss': cross_entropy_loss,
			'epoch': self.train_epoch,
			'auroc': roc_auc,
			'recall': recall,
			'selectivity': selectivity,
			'precision': precision,
			'accuracy': accuracy,
			'f1score': f1score,
			'informedness': informedness,
			'#TP': num_true_positives,
			'#FP': num_false_positives,
			'#TN': num_true_negatives,
			'#FN': num_false_negatives,
			'#Total': num_total_positives + num_total_negatives,
			})


		try: 
			os.makedirs(self.figure_path + 'classification_train/')
			os.makedirs(self.figure_path + 'classification_test/')
		except FileExistsError: pass

		# plt.figure()
		# plt.title('Classifier ROC ' + self.target_label)

		if client_tag == 'total':
			lineW = 5
		else:
			lineW = 2
		
		plt.plot(fp_rate, tp_rate, label = 'client ' + str(client_tag) + '   auroc: ' + str(np.round(roc_auc, 2)), linewidth=lineW)

		# plt.plot([0,1],[0,1],'r--')
		# plt.legend(loc='lower right')
		# plt.xlim([0., 1.])
		# plt.ylim([0., 1.])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.grid()
		# plt.savefig(self.figure_path + 'classification_roc/' + self.target_label + '_client_' + str(client) + '_epoch_' + str(self.train_epoch) + '.pdf')
		# plt.close()







