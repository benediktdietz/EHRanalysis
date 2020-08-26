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
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from fl_mnist_implementation_tutorial_utils import *
from data_management import eICU_DataLoader, DataProcessor, DataSetIterator, DataManager

datapath_processed = '../mydata/nomeds_20k_processed.csv'
federated_path = '../mydata/federated'

target_label = 'will_return'


available_datasets = os.listdir(federated_path)
available_dataset_ids = [int(dataid[9:-4]) for dataid in available_datasets]

SHUFFLE_BUFFER = 4
NUM_EPOCHS = 10
BATCH_SIZE = 10
PREFETCH_BUFFER = 2

target_features = 'length_of_icu'


def create_tf_dataset_for_client_fn(client_id):

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
	def build(shape, classes):
		model = Sequential()
		model.add(Dense(200, input_shape=(shape,)))
		model.add(Activation("relu"))
		model.add(Dense(200))
		model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model
	

class FederatedLearner():

	def __init__(self, target_label, datapath, datapath_fed):

		self.datapath = datapath
		self.datapath_fed = datapath_fed
		self.target_label = target_label

		self.available_dataset_ids = [int(dataid[9:-4]) for dataid in os.listdir(self.datapath_fed)]
		self.available_dataset_ids_train = self.available_dataset_ids[:-10]
		self.available_dataset_ids_test = self.available_dataset_ids[-10:]

		self.traing_rounds = 100
		self.feature_dim = self.get_hospital_data(self.available_dataset_ids[0], return_feature_dim=True)


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

		#initialize global model
		self.global_model = MLP_classifier().build(self.feature_dim, 2)
	
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

		dataset = list(zip(feature_map, labels))
		random.shuffle(dataset)

		if return_feature_dim:
			return feature_map.shape[1]
		else:
			return dataset

	def train(self):
		
		self.traing_rounds = 100
				
		#create optimizer
		lr = 0.0001 
		loss='categorical_crossentropy'
		metrics = ['accuracy']
		optimizer = SGD(
			lr=lr, 
			decay=lr / self.traing_rounds, 
			momentum=0.9)
										 
		#commence global training loop
		for comm_round in range(self.traing_rounds):

			print('training federated round number' + str(comm_round))
							
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
					local_model = smlp_local.build(self.feature_dim, 2)
					local_model.compile(
						loss=loss, 
						optimizer=optimizer,
						metrics=metrics)
					
					#set local model weight to the weight of the global model
					local_model.set_weights(global_weights)

					# print('\nself.clients_batched[client]\n', self.clients_batched[client], '\n')

					#fit local model with client's data
					local_model.fit(self.clients_batched[client], epochs=1, verbose=0)
					
					#scale the model weights and add to list
					scaling_factor = weight_scalling_factor(self.clients_batched, client)
					scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
					scaled_local_weight_list.append(scaled_weights)
					
					#clear session to free memory after each communication round
					K.clear_session()
					
			#to get the average over all the local model, we simply take the sum of the scaled weights
			average_weights = sum_scaled_weights(scaled_local_weight_list)
			
			#update global model 
			self.global_model.set_weights(average_weights)



	# 		#test global model and print out metrics after each communications round
	# 		for(self.X_test, self.Y_test) in self.test_batched:
	# 				global_acc, global_loss = test_model(self.X_test, self.Y_test, self.global_model, comm_round)
	# 				SGD_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(len(self.y_train)).batch(320)

	
		# self.SGD_model.compile(
		# 	loss=loss, 
		# 	optimizer=optimizer, 
		# 	metrics=metrics)

		# # fit the SGD training data to model
		# _ = self.SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

		# #test the SGD global model and print out metrics
		# for(self.X_test, self.Y_test) in self.test_batched:
		# 				SGD_acc, SGD_loss = test_model(self.X_test, self.Y_test, self.SGD_model, 1)



	def validate(self):

					
		# get the global model's weights - will serve as the initial weights for all local models
		global_weights = self.global_model.get_weights()
		
		#initial list to collect local model weights after scalling
		scaled_local_weight_list = list()

		#randomize client data - using keys
		client_names= list(self.clients_batched.keys())

		
		#loop through each client and create new local model
		for client in client_names:
				smlp_local = MLP_classifier()
				local_model = smlp_local.build(self.feature_dim, 2)
				local_model.compile(
					loss=loss, 
					optimizer=optimizer,
					metrics=metrics)
				
				#set local model weight to the weight of the global model
				local_model.set_weights(global_weights)

				# print('\nself.clients_batched[client]\n', self.clients_batched[client], '\n')

				#fit local model with client's data
				local_model.fit(self.clients_batched[client], epochs=1, verbose=0)
				
				#scale the model weights and add to list
				scaling_factor = weight_scalling_factor(self.clients_batched, client)
				scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
				scaled_local_weight_list.append(scaled_weights)
				
				#clear session to free memory after each communication round
				K.clear_session()
				
		#to get the average over all the local model, we simply take the sum of the scaled weights
		average_weights = sum_scaled_weights(scaled_local_weight_list)
		
		#update global model 
		self.global_model.set_weights(average_weights)


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

FL_network = FederatedLearner(target_label, datapath_processed, federated_path)

FL_network.train()








