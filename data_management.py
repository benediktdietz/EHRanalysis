import math, re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

class eICU_DataLoader():

	def __init__(self, read_path, write_path, num_patients = -1):

		self.read_path = read_path
		self.write_path = write_path

		self.num_patients = num_patients

		self.build_patient_matrix()

	def build_patient_matrix(self):

		remove_nans_from_codes = True

		patient_table = pd.read_csv(self.read_path + 'patient.csv')
		patient_table = patient_table.loc[patient_table['age'] != 'NaN']
		patient_table = patient_table.loc[patient_table['age'] != 'nan']
		print('patient_table loaded successfully'.ljust(50) + str(np.round(len(patient_table)/1000000., 1)) + ' Mio rows | ' + str(int(patient_table.shape[1])) + ' cols')
		medication_table = pd.read_csv(self.read_path + 'medication.csv', low_memory=False)
		medication_table = medication_table.loc[medication_table['drugordercancelled'] == 'No']
		print('medication_table loaded successfully'.ljust(50) + str(np.round(len(medication_table)/1000000., 1)) + ' Mio rows | ' + str(int(medication_table.shape[1])) + ' cols')
		diagnosis_table = pd.read_csv(self.read_path + 'diagnosis.csv')
		print('diagnosis_table loaded successfully'.ljust(50) + str(np.round(len(diagnosis_table)/1000000., 1)) + ' Mio rows | ' + str(int(diagnosis_table.shape[1])) + ' cols')
		pasthistory_table = pd.read_csv(self.read_path + 'pastHistory.csv')
		print('pasthistory_table loaded successfully'.ljust(50) + str(np.round(len(pasthistory_table)/1000000., 1)) + ' Mio rows | ' + str(int(pasthistory_table.shape[1])) + ' cols')
		print('\n\n')

		if self.num_patients < 0:
			num_patients_to_load = len(patientIDs)
		else:
			num_patients_to_load = self.num_patients

		patientIDs = patient_table['uniquepid'].unique()[:num_patients_to_load]

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.7, random_state=123)

		for train_index, test_index in stratified_splitter.split(
			np.zeros(len(patientIDs)), 
			np.zeros(len(patientIDs))):
				train_patient_ids = patientIDs[train_index]
				test_patient_ids = patientIDs[test_index]
		print('dataset split for training (' + str(len(train_patient_ids)) + ' patients) and validation (' + str(len(test_patient_ids)) + ' patients)\n\n')

		data_df = []
		corr_id_df = []

		print('looping through patient IDs in loaded tables to build a consolidated matrix...')
		pbarfreq = 10
		# pbar = tqdm(total=int(np.floor(num_patients_to_load/pbarfreq)))
		pbar = tqdm(total=num_patients_to_load)
		for i in range(num_patients_to_load):

			if i % pbarfreq == 0: pbar.update(pbarfreq)

			patient = patientIDs[i]

			correlated_unitstay_ids = np.asarray(patient_table['patientunitstayid'].loc[patient_table['uniquepid'] == patient].values)

			if patient in train_patient_ids:
				data_set_ref = 'training'
			elif patient in test_patient_ids:
				data_set_ref = 'validation'

			# corr_id_df = []
			for j in range(len(correlated_unitstay_ids)):


				correlated_pasthistory_ids = np.asarray(pasthistory_table['pasthistoryid'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				pasthistory_notetypes = np.asarray(pasthistory_table['pasthistorynotetype'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].unique())
				pasthistory_values = np.asarray(pasthistory_table['pasthistoryvalue'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].unique())


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



				current_visit_number = patient_table['unitvisitnumber'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
				current_health_sys_id = patient_table['patienthealthsystemstayid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
				
				max_visits_for_current_stay = patient_table['unitvisitnumber'].loc[patient_table['patienthealthsystemstayid'] == current_health_sys_id].max()
				if current_visit_number < max_visits_for_current_stay:
					will_return = 1.
				else:
					will_return = 0.

				unit_readmission_dummy = patient_table['unitstaytype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
				if unit_readmission_dummy == 'readmit':
					unit_readmission = 1.
				else:
					unit_readmission = 0.

				will_die_dummy_unit_discharge_status = patient_table['unitdischargestatus'].loc[patient_table['patienthealthsystemstayid'] == current_health_sys_id].values
				will_die_dummy_unit_discharge_location = patient_table['unitdischargelocation'].loc[patient_table['patienthealthsystemstayid'] == current_health_sys_id].values
				if 'Expired' in will_die_dummy_unit_discharge_status or 'Death' in will_die_dummy_unit_discharge_location:
					will_die = 1.
				else:
					will_die = 0.

				survive_current_icu_dummy = patient_table['unitdischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
				if survive_current_icu_dummy == 'Alive':
					survive_current_icu = 1.
				else:
					survive_current_icu = 0.

				will_readmit_dummy = patient_table[['unitstaytype', 'unitvisitnumber']].loc[patient_table['patienthealthsystemstayid'] == current_health_sys_id]
				will_readmit_dummy = will_readmit_dummy['unitstaytype'].loc[will_readmit_dummy['unitvisitnumber'] > current_visit_number].values
				# print('____', will_readmit_dummy)
				if 'readmit' in will_readmit_dummy:
					will_readmit = 1.
				else:
					will_readmit = 0.


				medication_ids = np.asarray(medication_table['medicationid'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				diagnosis_ids = np.asarray(diagnosis_table['diagnosisid'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				drug_strings_prescribed = medication_table['drugname'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values


				lengthofstay = patient_table['hospitaldischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() - patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
				lengthofstay = np.round(lengthofstay/60., 1)

				icu_discharge = patient_table['unitdischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() - patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
				icu_discharge = np.round(icu_discharge/60., 1)




				corr_id_df.append(
					{
					'patient_id': patient,
					'health_system_id': current_health_sys_id,
					'corr_id': correlated_unitstay_ids[j],
					'gender': patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item(),
					'data_set_ref': data_set_ref,
					'age': age_dummy,
					'ethnicity': ethnicity_dummy,
					'visit_number': current_visit_number,
					'icu_admission_time': np.round(np.abs(patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.), 2),
					'length_of_stay': lengthofstay,
					'length_of_icu': np.round(icu_discharge - np.abs(patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.), 1),
					'icu_discharge': icu_discharge,
					'will_return': will_return,
					'will_die': will_die,
					'will_readmit': will_readmit,
					'survive_current_icu': survive_current_icu,
					'unit_readmission': unit_readmission,
					'visits_current_stay': max_visits_for_current_stay,
					'hospital_discharge_status': patient_table['hospitaldischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_admit_offset': patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_discharge_offset': patient_table['hospitaldischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_discharge_year': patient_table['hospitaldischargeyear'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_admit_source': patient_table['unitadmitsource'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_type': patient_table['unittype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_discharge_status': patient_table['unitdischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
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
					'pasthistory_notetypes': pasthistory_notetypes,
					'pasthistory_values': pasthistory_values,
					})

		pbar.close()
		print('\n')


		pd.DataFrame(corr_id_df).to_csv(self.write_path)

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
			'hospital_discharge_status',
			'hospital_discharge_year',
			'unit_discharge_status',
			'unit_admit_source',
			'unit_type',
			'unit_discharge_location',
			'unit_stay_type',
			]

		array_features = [
			'drug_codes_prescribed',
			'diagnosis_ICD9code',
			'pasthistory_notetypes',
			'pasthistory_values',
			]

		array_features_unused = [
			'drug_strings_prescribed',
			'diagnosis_string',
			'medication_ids',
			'diagnosis_ids',
			]


		self.df_onehot = pd.get_dummies(self.dataframe, columns = categorical_feature_names, prefix = categorical_feature_names)
		
		self.process_array_cols(array_features)
		
		self.df_onehot.drop(columns=array_features_unused)

		self.df_onehot.to_csv(self.write_path)


	def process_array_cols(self, col_names):

		print_out = False
		progbar = True
		pbarfreq = 10
		pbarcounter = 0

		# dummy_df = []

		for col_name in col_names:

			print('\nlooping through ' + col_name + ' column to build encoded feature map...')
			# if progbar: pbar = tqdm(total=int(np.floor(len(self.dataframe)/pbarfreq)))
			if progbar: pbar = tqdm(total=len(self.dataframe))


			for row in range(len(self.dataframe)):

				if col_name in ['pasthistory_notetypes', 'pasthistory_values']:

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
						# print(nice_dummy_list)
						# print('---', len(nice_dummy_list))
						print(nice_dummy_list2)
						print('---', len(nice_dummy_list2))
						print()

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
						# print(nice_dummy_list)
						# print('---', len(nice_dummy_list))
						print(nice_dummy_list2)
						print('---', len(nice_dummy_list2))
						print()

				if col_name in ['medication_ids', 'diagnosis_ids', 'drug_codes_prescribed']:

					nice_dummy_list = list(re.split(" ", self.dataframe[col_name].iloc[row]))

					if '[' in nice_dummy_list: nice_dummy_list.remove('[')
					if ']' in nice_dummy_list: nice_dummy_list.remove(']')
					if '[]' in nice_dummy_list: nice_dummy_list.remove('[]')
					if '\n' in nice_dummy_list: nice_dummy_list.remove('\n')
					if '[nan ' in nice_dummy_list: nice_dummy_list.remove('[nan ')
					if ' nan nan]' in nice_dummy_list: nice_dummy_list.remove(' nan nan]')
					if ' nan nan\n ' in nice_dummy_list: nice_dummy_list.remove(' nan nan\n ')
					
					nice_dummy_list2 = []
					for dummy_i in range(len(nice_dummy_list)):
						if ' nan' not in nice_dummy_list[dummy_i] and nice_dummy_list[dummy_i] != '':
							try: nice_dummy_list2.append(int(nice_dummy_list[dummy_i]))
							except ValueError:
								dummy_string = str(nice_dummy_list[dummy_i])
								if len(dummy_string) < 2:
									continue
								if dummy_string[0] == '[':
									dummy_string = dummy_string[1:]
								if dummy_string[-1] == ']':
									dummy_string = dummy_string[:-1]
								# if dummy_string[-2] == '\n' and len(dummy_string) > 2:
								# 	dummy_string = dummy_string[:-2]
								try: 
									nice_dummy_list2.append(int(dummy_string))
								except ValueError:
									continue

									

							

					self.dataframe[col_name].iloc[row] = nice_dummy_list2

					if print_out:
						print()
						print('******************************** ' + col_name)
						# print(nice_dummy_list)
						# print('---', len(nice_dummy_list))
						print(nice_dummy_list2)
						print('---', len(nice_dummy_list2))
						print()



				onehot_keys = np.concatenate([['corr_id'], np.asarray(pd.get_dummies(
					self.dataframe[col_name].iloc[row],
					prefix = col_name
					).sum(0).keys().values)])
				onehot_values = np.reshape(
					np.concatenate(
						[[self.dataframe['corr_id'].iloc[row]], 
						np.asarray(
							pd.get_dummies(
								self.dataframe[col_name].iloc[row], 
								prefix = col_name
								).sum(0).values, 
							dtype=np.int)
						]), (1,-1))
				

				if progbar:	
					pbarcounter += 1
					if pbarcounter % pbarfreq == 0:
						pbar.update(pbarfreq)


				if row == 0:
					dummy_df = pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)
				else:
					dummy_df = pd.concat([dummy_df, pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)], axis=0, sort=False).fillna(.0)


			if progbar: 
				pbar.close()
				print('\n')


			self.df_onehot.drop(columns=col_name)
			self.df_onehot = pd.merge(self.df_onehot, dummy_df, on='corr_id')

			if print_out: 
				print(
					'\n\n************************************\ndf_onehot DataFrame:\n', 
					self.df_onehot, 
					'\n************************************\n\n\n')

class DataSetIterator(Dataset):

  def __init__(self, features, labels):

        self.labels = labels
        self.features = features

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):

        x = self.features.loc[index, :]
        y = self.labels.loc[index]

        return x, y

class DataManager():

	def __init__(self, process_data_path, target_features):

		self.data_df = pd.read_csv(process_data_path).drop(columns='Unnamed: 0')
		self.data_df = self.data_df.loc[self.data_df['unit_discharge_offset'] != 0]


		self.target_features = target_features

		# for k in range(len(self.data_df.keys())):
		# 	print('.........', str(self.data_df.keys()[k]).ljust(70, '.'), self.data_df[self.data_df.keys()[k]].dtype)


		self.label_cols = [
			# doesnt make sense to include or not properly formatted cols
			'patient_id',
			'health_system_id',
			'corr_id',
			'data_set_ref',
			'medication_ids',
			'drug_strings_prescribed',
			'drug_codes_prescribed',
			'diagnosis_string',
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
			'diagnosis_ids',
			'diagnosis_priority',
			'diagnosis_ICD9code',
			'unit_discharge_offset',
			'unit_discharge_status_Alive',
			'unit_discharge_status_Expired',
			'unit_discharge_location_Death',
			'unit_discharge_location_Floor',
			'unit_discharge_location_Home',
			'unit_discharge_location_Other',
			'unit_discharge_location_Other External',
			'unit_discharge_location_Other Hospital',
			'unit_discharge_location_Other ICU',
			'unit_discharge_location_Rehabilitation',
			'unit_discharge_location_Skilled Nursing Facility',
			'unit_discharge_location_Step-Down Unit (SDU)',
			'unit_discharge_location_Telemetry',
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
			'visits_current_stay',
			'unit_readmission',
			'survive_current_icu',
			]

		# self.remove_some_outliers()

		self.consolidate_previous_ICUs()
		self.check_data()


		self.training_data = self.split_data()

	def consolidate_previous_ICUs(self):

		consolidation_keys = self.data_df.keys.values
		consolidation_keys.drop(columns=self.label_cols)
		consolidation_keys.drop(columns=['gender', 'age', 'ethnicity'])

		corr_ids = self.data_df['corr_id'].unique()

		for correlated_id in corr_ids:

			corr_health_id = self.data_df['health_system_id'].loc[self.data_df['corr_id'] == correlated_id].values.item()
			corr_visit_number = self.data_df['visit_number'].loc[self.data_df['corr_id'] == correlated_id].values.item()

			if corr_visit_number > 1:

				print('HELLO!!!')

				self.data_df[consolidation_keys].loc[self.data_df['corr_id'] == correlated_id] = self.data_df[consolidation_keys].loc[self.data_df['health_system_id'] == corr_health_id & self.data_df['visit_number'] < corr_visit_number].sum(0)







	def remove_some_outliers(self):

		num_rows_before = self.data_df.shape[0]
		num_cols_before = len(self.data_df.keys())

		# dummyset = np.asarray(self.data_df.drop(columns = self.label_cols).astype(int).values)
		# isolation_forest = IsolationForest(
		# 	n_estimators = 246,
		# 	max_samples = .8,
		# 	max_features = .5,
		# 	n_jobs = -1,
		# 	verbose = 0,
		# 	).fit(dummyset)
		# outlier_index = np.asarray(isolation_forest.predict(dummyset).astype(int))

		# if np.sum(outlier_index) < num_rows_before:
		# 	print('isolation_forest removing ' + str(np.sum(outlier_index) - num_rows_before) + ' samples\n\n')
		# 	self.data_df = self.data_df.iloc[outlier_index == 1, :]

		print('\nremoving outliers...\n')
		keys_to_drop = []
		pbar = tqdm(total=len(self.data_df.keys().values) - len(self.label_cols))
		for key in self.data_df.keys().values:
			if key not in self.label_cols and self.data_df[key].nunique() < 20:
				for unique_val in self.data_df[key].unique():
					if (self.data_df[key] == unique_val).astype(float).mean() < .0001:
						self.data_df = self.data_df[self.data_df[key] != unique_val]
				if self.data_df[key].nunique() < 2:
					keys_to_drop.append(key)
			pbar.update(1)
		pbar.close()
		self.data_df.drop(columns = keys_to_drop)

	
		# no_value_cols = np.asarray(np.where(self.data_df.apply(pd.Series.value_counts) <= 1))
		# no_value_col_names = list(self.data_df.keys().values[no_value_cols])
		# self.data_df = self.data_df.drop(columns = no_value_col_names)



		num_removed_rows = num_rows_before - self.data_df.shape[0]
		num_removed_cols = num_cols_before - len(self.data_df.keys())
		frac_removed_rows = num_removed_rows / num_rows_before
		frac_removed_cols = num_removed_cols / num_cols_before

		print('\n\noriginal rows:'.ljust(25, '.') + str(int(num_rows_before)) + ' | removed', int(num_removed_rows), '/',  np.round(100 * frac_removed_rows, 1), '% outlier samples')
		print('original cols:'.ljust(25, '.') + str(int(num_cols_before)) + ' | removed', int(num_removed_cols), '/',  np.round(100 * frac_removed_cols, 1), '% outlier features\n\n')

	def check_data(self):

		# unit_stay_type_admit
		# unit_stay_type_readmit
		# unit_stay_type_stepdown/other
		# unit_stay_type_transfer
		
		dummy_patient_info = []
		earlystopper = 0
		previous_patient_id = 0
		for corr_id in self.data_df['corr_id'].values:

			if earlystopper < 50:

				new_patient_id = self.data_df['patient_id'].loc[self.data_df['corr_id'] == corr_id].values.item()

				if new_patient_id != previous_patient_id:
					print('\npatient_id:' + str(self.data_df['patient_id'].loc[self.data_df['corr_id'] == corr_id].values.item()) + '\n', pd.DataFrame(dummy_patient_info))
					dummy_patient_info = []
					earlystopper += 1


				dummy_patient_info.append({
					'patient_id': self.data_df['patient_id'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'system_id': self.data_df['health_system_id'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'case_id': corr_id,
					'L hospital': self.data_df['length_of_stay'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'L ICU': self.data_df['length_of_icu'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'ICU end': self.data_df['icu_discharge'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'return': self.data_df['will_return'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'survive_icu': self.data_df['survive_current_icu'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'will_die': self.data_df['will_die'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'visit': self.data_df['visit_number'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'readmit': self.data_df['unit_readmission'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					'will_readmit': self.data_df['will_readmit'].loc[self.data_df['corr_id'] == corr_id].values.item(),
					})


				previous_patient_id = new_patient_id

	def split_data(self):

		unique_patient_ids = self.data_df['patient_id'].unique()

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.7, random_state=123)

		for train_index, test_index in stratified_splitter.split(
			np.zeros(len(unique_patient_ids)), 
			np.zeros(len(unique_patient_ids))):
				train_patient_ids = unique_patient_ids[train_index]
				test_patient_ids = unique_patient_ids[test_index]

		x_training = self.data_df.loc[self.data_df['data_set_ref'] == 'training'].drop(columns = self.label_cols)
		x_validation = self.data_df.loc[self.data_df['data_set_ref'] == 'validation'].drop(columns = self.label_cols)
		y_training = self.data_df[self.target_features].loc[self.data_df['data_set_ref'] == 'training']
		y_validation = self.data_df[self.target_features].loc[self.data_df['data_set_ref'] == 'validation']


		data_container = {
				'x_full': self.data_df.drop(columns = self.label_cols),
				'x_train': x_training,
				'x_test': x_validation,
				'y_full': self.data_df[self.target_features],
				'y_train': y_training,
				'y_test': y_validation}

		return data_container

	def split_data_old(self):

		unique_patient_ids = self.data_df['patient_id'].unique()

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.7, random_state=123)


		for train_index, test_index in stratified_splitter.split(
			np.zeros(len(self.labels)), 
			np.zeros(len(self.labels)) 
			# pd.cut(np.reshape(self.labels[self.target_features].values, (-1)), bins=2)
			):
			x_training = self.features.iloc[train_index,:]
			y_training = self.labels[self.target_features].iloc[train_index,:]
			x_validation = self.features.iloc[test_index,:]
			y_validation = self.labels[self.target_features].iloc[test_index,:]


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
				self.training_data['y_train'][self.target_features]), 
			batch_size=batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)

	def get_test_iterator(self, batch_size):
		
		return DataLoader(
			DataSetIterator(
				self.training_data['x_test'], 
				self.training_data['y_test'][self.target_features]), 
			batch_size=batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)
