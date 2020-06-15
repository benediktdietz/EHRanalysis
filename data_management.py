import math, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, IterableDataset, DataLoader


class eICU_DataLoader():

	def __init__(self, read_path, write_path):

		self.read_path = read_path
		self.write_path = write_path
		self.build_patient_matrix()

	def build_patient_matrix(self):

		remove_nans_from_codes = True

		patient_table = pd.read_csv(self.read_path + 'patient.csv')
		patient_table = patient_table.loc[patient_table['age'] != 'NaN']
		patient_table = patient_table.loc[patient_table['age'] != 'nan']
		print('\n\n\n**************\npatient_table loaded successfully:\n**************\n', patient_table.nunique())
		medication_table = pd.read_csv(self.read_path + 'medication.csv')
		medication_table = medication_table.loc[medication_table['drugordercancelled'] == 'No']
		print('\n\n\n**************\nmedication_table loaded successfully:\n**************\n', medication_table.nunique())
		diagnosis_table = pd.read_csv(self.read_path + 'diagnosis.csv')
		print('\n\n\n**************\ndiagnosis_table loaded successfully:\n**************\n', diagnosis_table.nunique())

		patientIDs = patient_table['uniquepid'].unique()

		data_df = []
		corr_id_df = []

		# for i in range(len(patientIDs)):
		for i in range(10000):

			if i % 1000 == 0:
				print('\nrunning patient_id ' + str(i))

			patient = patientIDs[i]

			correlated_unitstay_ids = np.asarray(patient_table['patientunitstayid'].loc[patient_table['uniquepid'] == patient].values)

			# corr_id_df = []
			for j in range(len(correlated_unitstay_ids)):


				if patient_table['age'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() == '> 89':
					age_dummy = 90
				else:
					try:
						age_dummy = int(patient_table['age'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item())
					except ValueError:
						continue

				if str(patient_table['ethnicity'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()) == 'nan':
					ethnicity_dummy = 'Unknown'
				else:
					ethnicity_dummy = str(patient_table['ethnicity'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item())


				medication_ids = np.asarray(medication_table['medicationid'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)

				diagnosis_ids = np.asarray(diagnosis_table['diagnosisid'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				
				drug_strings_prescribed = medication_table['drugname'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values

				if remove_nans_from_codes:
					drug_codes_prescribed0 = medication_table['drughiclseqno'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
					drug_codes_prescribed = []
					for h in range(len(drug_codes_prescribed0)):
						if str(drug_codes_prescribed0[h]) != 'nan':
							drug_codes_prescribed.append(int(drug_codes_prescribed0[h]))
					icd9codes0 = diagnosis_table['icd9code'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
					icd9codes = []
					for h in range(len(icd9codes0)):
						if str(icd9codes0[h]) != 'nan':
							icd9codes.append(str(icd9codes0[h]))

				else:
					drug_codes_prescribed = medication_table['drughiclseqno'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
					icd9codes = diagnosis_table['icd9code'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values


				# print(len(diagnosis_table['diagnosisstring'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values))

				corr_id_df.append(
					{
					'patient_id': patient,
					'health_system_id': int(patient_table['patienthealthsystemstayid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values),
					'corr_id': correlated_unitstay_ids[j],
					'gender': patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item(),
					'age': age_dummy,
					'ethnicity': ethnicity_dummy,
					'visit_number': patient_table['unitvisitnumber'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_discharge_status': patient_table['hospitaldischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_admit_offset': patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_discharge_offset': patient_table['hospitaldischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_discharge_year': patient_table['hospitaldischargeyear'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_admit_source': patient_table['unitadmitsource'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_type': patient_table['unittype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_discharge_offset': patient_table['unitdischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_discharge_location': patient_table['unitdischargelocation'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_stay_type': patient_table['unitstaytype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_id': patient_table['hospitalid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'medication_ids': medication_ids,
					'diagnosis_ids': diagnosis_ids,
					'drug_strings_prescribed': np.asarray(drug_strings_prescribed),
					'drug_codes_prescribed': np.asarray(drug_codes_prescribed),
					'diagnosis_activeUponDischarge': diagnosis_table['activeupondischarge'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'diagnosis_offset': diagnosis_table['diagnosisoffset'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'diagnosis_priority': diagnosis_table['diagnosispriority'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'diagnosis_ICD9code': np.asarray(icd9codes),
					'diagnosis_string': diagnosis_table['diagnosisstring'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					})



			if len(correlated_unitstay_ids) == 1:

				if patient_table['age'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item() == '> 89':
					age_dummy = 90
				else:
					try:
						age_dummy = int(patient_table['age'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item())
					except ValueError:
						continue

				if str(patient_table['ethnicity'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item()) == 'nan':
					ethnicity_dummy = 'Unknown'
				else:
					ethnicity_dummy = str(patient_table['ethnicity'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item())

				data_df.append(
					{
					'patientID': patient,
					'health_system_id': patient_table['patienthealthsystemstayid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'CorrUnitIDs': correlated_unitstay_ids,
					'gender': patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item(),
					'age': age_dummy,
					'ethnicity': ethnicity_dummy,
					'hospitalid': patient_table['hospitalid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item(),
					})

		data_df = pd.DataFrame(data_df)
		
		corr_id_df = pd.DataFrame(corr_id_df)

		corr_id_df.to_csv(self.write_path)

		print('\n\n\n')
		print(corr_id_df)
		# print('\n\n\n**************\ncorr_id_df loaded successfully:\n**************\n', corr_id_df.nunique())


		# self.all_files = []
		# for dummyfile in os.listdir(read_path):
		# 	if dummyfile.endswith(".csv"):
		# 		self.all_files.append(dummyfile)

		patient_table = []
		medication_table = []
		diagnosis_table = []

class DataProcessor():

	def __init__(self, read_path, write_path):

		self.read_path = read_path
		self.write_path = write_path

		self.dataframe = pd.read_csv(self.read_path).drop(columns='Unnamed: 0')

		categorical_feature_names = [
			'gender',
			'ethnicity',
			'visit_number',
			'hospital_discharge_status',
			'hospital_discharge_year',
			'unit_admit_source',
			'unit_type',
			'unit_discharge_location',
			'unit_stay_type',
			]

		array_features = [
			'drug_strings_prescribed',
			'diagnosis_string',
			]

		array_features_unused = [
			'medication_ids',
			'diagnosis_ids',
			'drug_codes_prescribed',
			'diagnosis_ICD9code',
			]


		self.df_onehot = pd.get_dummies(self.dataframe, columns = categorical_feature_names)

		# print('*********************')
		# print(self.dataframe.keys())
		# print(self.df_onehot.keys())
		# print('*********************')

		self.process_array_cols(array_features)
		
		self.df_onehot.drop(columns=array_features_unused)

		self.df_onehot.to_csv(self.write_path)


		print('\n\n\n')
		print(self.df_onehot)
		print(self.df_onehot.dtypes)
		print(self.df_onehot.nunique())

	def process_array_cols(self, col_names):

		print_out = False

		# dummy_df = []

		for col_name in col_names:

			for row in range(len(self.dataframe)):
			# for row in range(50):


				if col_name in ['drug_strings_prescribed', 'diagnosis_ICD9code', 'diagnosis_string']:

					nice_dummy_list = list(re.split("'", self.dataframe[col_name].iloc[row]))

					if '[' in nice_dummy_list: nice_dummy_list.remove('[')
					if ']' in nice_dummy_list: nice_dummy_list.remove(']')
					if '[]' in nice_dummy_list: nice_dummy_list.remove('[]')
					if '[nan ' in nice_dummy_list: nice_dummy_list.remove('[nan ')
					if ' nan nan]' in nice_dummy_list: nice_dummy_list.remove(' nan nan]')
					if ' nan nan\n ' in nice_dummy_list: nice_dummy_list.remove(' nan nan\n ')
					
					nice_dummy_list2 = []
					for dummy_i in range(len(nice_dummy_list)):
						# if nice_dummy_list[dummy_i] != '\n ':
						if ' nan' not in nice_dummy_list[dummy_i] and len(nice_dummy_list[dummy_i]) > 1:
							if '\n' not in nice_dummy_list[dummy_i]:
								if nice_dummy_list[dummy_i][0] == '[':
									nice_dummy_list2.append(nice_dummy_list[dummy_i][1:])
								elif nice_dummy_list[dummy_i][-2:] == '\n':
									nice_dummy_list2.append(nice_dummy_list[dummy_i][:-2])
								else:
									nice_dummy_list2.append(nice_dummy_list[dummy_i])

					self.dataframe[col_name].iloc[row] = nice_dummy_list2

					if print_out:
						print()
						print('******************************** ' + col_name)
						print(nice_dummy_list)
						print('---', len(nice_dummy_list))
						print(nice_dummy_list2)
						print('---', len(nice_dummy_list2))
						print()

				if col_name in ['medication_ids', 'diagnosis_ids', 'drug_codes_prescribed']:

					nice_dummy_list = list(re.split(" ", self.dataframe[col_name].iloc[row]))

					if '[' in nice_dummy_list: nice_dummy_list.remove('[')
					if ']' in nice_dummy_list: nice_dummy_list.remove(']')
					if '[]' in nice_dummy_list: nice_dummy_list.remove('[]')
					if '[nan ' in nice_dummy_list: nice_dummy_list.remove('[nan ')
					if ' nan nan]' in nice_dummy_list: nice_dummy_list.remove(' nan nan]')
					if ' nan nan\n ' in nice_dummy_list: nice_dummy_list.remove(' nan nan\n ')
					
					nice_dummy_list2 = []
					for dummy_i in range(len(nice_dummy_list)):
						if ' nan' not in nice_dummy_list[dummy_i] and nice_dummy_list[dummy_i] != '':
							try: nice_dummy_list2.append(int(nice_dummy_list[dummy_i]))
							except ValueError:
								dummy_string = str(nice_dummy_list[dummy_i])
								if dummy_string[0] == '[':
									dummy_string = dummy_string[1:]
								if dummy_string[-1] == ']':
									dummy_string = dummy_string[:-1]
								if dummy_string[-2] == '\n':
									dummy_string = dummy_string[:-2]
								nice_dummy_list2.append(int(dummy_string))

					self.dataframe[col_name].iloc[row] = nice_dummy_list2

					if print_out:
						print()
						print('******************************** ' + col_name)
						print(nice_dummy_list)
						print('---', len(nice_dummy_list))
						print(nice_dummy_list2)
						print('---', len(nice_dummy_list2))
						print()

				# onehot_keys = list(pd.get_dummies(self.dataframe[col_name].iloc[row]).sum(0).keys().values)
				# onehot_values = list(pd.get_dummies(self.dataframe[col_name].iloc[row]).sum(0).values)

				onehot_keys = np.concatenate([['corr_id'], np.asarray(pd.get_dummies(self.dataframe[col_name].iloc[row]).sum(0).keys().values)])
				onehot_values = np.reshape(
					np.concatenate(
						[[self.dataframe['corr_id'].iloc[row]], 
						np.asarray(pd.get_dummies(self.dataframe[col_name].iloc[row]).sum(0).values, dtype=np.int)
						]), (1,-1))
				
				# print('\n###########################')
				# print(pd.DataFrame(onehot_values, index = [row], columns = onehot_keys))



				if row == 0:
					dummy_df = pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)
				else:
					dummy_df = pd.concat([dummy_df, pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)], axis=0, sort=False).fillna(.0)

				if row % 1000 == 0: print('\nrunning id ' + str(row) + '\ndummy DataFrame:\n', dummy_df, '\n************************************\n\n\n')




			self.df_onehot.drop(columns=col_name)
			self.df_onehot = pd.merge(self.df_onehot, dummy_df, on='corr_id')

			print('\n\n************************************\ndf_onehot DataFrame:\n', self.df_onehot, '\n************************************\n\n\n')

class DataSetIterator(Dataset):

  def __init__(self, features, labels):

        self.labels = labels
        self.features = features

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):

        x = self.features[index, :]
        y = self.labels[index]

        return x, y

class DataManager():

	def __init__(self, process_data_path, target_features):

		self.data_df = pd.read_csv(process_data_path).drop(columns='Unnamed: 0')
		self.data_df = self.data_df.loc[self.data_df['unit_discharge_offset'] != 0.]

		self.target_features = target_features


		self.label_cols = [
			'patient_id',
			'health_system_id',
			'corr_id',
			'medication_ids',
			'drug_strings_prescribed',
			'drug_codes_prescribed',
			'diagnosis_offset',
			'diagnosis_activeUponDischarge',
			'diagnosis_ids',
			'diagnosis_priority',
			'diagnosis_ICD9code',
			'diagnosis_string',
			'unit_discharge_offset',
			'unit_discharge_location_Death',
			'hospital_discharge_offset',
			'hospital_discharge_status_Alive',
			'hospital_discharge_status_Expired',
			]


		self.features = self.data_df.drop(columns = self.label_cols).astype(int)

		# for i in range(len(self.features.keys())):
			# print(self.features['diagnosis_offset'])
			# print(str(self.features.keys()[i]).ljust(20, '.'), self.features[self.features.keys()[i]].dtypes)



		self.labels = self.data_df[self.label_cols]


		self.training_data = self.split_data()


	def split_data(self):

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.3, random_state=123)


		for train_index, test_index in stratified_splitter.split(
			np.zeros(self.features.shape[0]), 
			pd.cut(np.reshape(self.labels[self.target_features].values, (-1)), bins=2)
			):
			x_training, x_validation = self.features.iloc[train_index,:], self.features.iloc[test_index,:]
			y_training, y_validation = self.labels[self.target_features].iloc[train_index,:], self.labels[self.target_features].iloc[test_index,:]

		data_container = {
				'x_full': self.features,
				'x_train': x_training,
				'x_test': x_validation,
				'y_full': self.labels,
				'y_train': y_training,
				'y_test': y_validation}

		return data_container

	def get_train_iterator(self, batch_size):
		
		return DataLoader(
			DataSetIterator(
				self.training_data['x_train'], 
				self.training_data['y_train']), 
			batch_size=batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)

	def get_test_iterator(self, batch_size):
		
		return DataLoader(
			DataSetIterator(
				self.training_data['x_test'], 
				self.training_data['y_test']), 
			batch_size=batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)
