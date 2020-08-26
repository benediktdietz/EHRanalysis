import os
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
import pandas as pd
import numpy as np
nest_asyncio.apply()
tf.keras.backend.set_floatx('float32')

federated_path = '../mydata/federated'
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

	labels = client_data[target_features]

	# client_data = df[df[client_id_colname] == client_id]

	dataset = tf.data.Dataset.from_tensor_slices((feature_map, labels))
	# dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict('list'))
	dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).repeat(NUM_EPOCHS)

	print('dataset:\n', dataset)

	return dataset


for i in range(2):
	dummy = create_tf_dataset_for_client_fn(available_dataset_ids[i])
	# print(dummy)
# for _ in range(10): print('****************************')



# Pick a subset of client devices to participate in training.
train_data = [create_tf_dataset_for_client_fn(n) for n in available_dataset_ids[:5]]

# Grab a single batch of data so that TFF knows what data looks like.
sample_batch = tf.nest.map_structure(
lambda x: x.numpy(), iter(train_data[0]).next())

def preprocess(dataset):

	# def batch_format_fn(element):
	# 	return collections.OrderedDict(
	# 		x=tf.reshape(element['features'], [-1, 808]),
	# 		y=tf.reshape(element['label'], [-1, 1]))

	# return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

	
	return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

train_data = preprocess(train_data[0])

print('##############')
print(train_data)
print('##############')



class MyModel(tf.keras.Model):

	def __init__(self):
		super(MyModel, self).__init__()
		# self.input = tf.keras.layers.Input(shape=(None,808)),
		self.dense1 = tf.keras.layers.Dense(512, input_shape=(None,808), activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
		self.dense3 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
		# self.dropout = tf.keras.layers.Dropout(0.5)

	def call(self, inputs, training=False):

		# print('----------', inputs[0])
		# print('----------', inputs[1])

		x = self.dense1(inputs[0])
		# if training: x = self.dropout(x, training=training)

		x = self.dense2(x)
		# if training: x = self.dropout(x, training=training)

		x = self.dense3(x)
		# if training: x = self.dropout(x, training=training)

		return x

def model_fn2():

	mymodel = MyModel()
	# mymodel.call(train_data[0])
	mymodel.build(input_shape=(None, 808))

	print('***************\n', mymodel.layers)
	print('***************\n', mymodel.summary())

	return tff.learning.from_keras_model(
		mymodel,
		loss=tf.keras.losses.MeanAbsolutePercentageError(),
		input_spec=dummy.element_spec,
		metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Wrap a Keras model for use with TFF.
def model_fn():

	model = tf.keras.models.Sequential(
	[
	tf.keras.layers.Dense(
		2048, 
		tf.nn.elu, 
		kernel_initializer='zeros'),
	tf.keras.layers.Dense(
		2048, 
		tf.nn.elu, 
		kernel_initializer='zeros'),
	tf.keras.layers.Dense(
		2048, 
		tf.nn.elu, 
		kernel_initializer='zeros'),
	])

	return tff.learning.from_keras_model(
		model,
		# dummy_batch=sample_batch,
		loss=tf.keras.losses.MeanAbsolutePercentageError(),
		input_spec=dummy.element_spec,
		metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
	model_fn2,
	client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))

state = trainer.initialize()
for _ in range(5):
	state, metrics = trainer.next(state, train_data)
	print (metrics.loss)






