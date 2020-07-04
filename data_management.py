import math, re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
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
		patient_table = patient_table.loc[patient_table['gender'] != 'Unknown']
		print('\n\n')
		print('patient_table loaded successfully'.ljust(50) + str(np.round(len(patient_table)/1000000., 1)) + ' Mio rows | ' + str(int(patient_table.shape[1])) + ' cols')
		# medication_table = pd.read_csv(self.read_path + 'medication.csv', low_memory=False)
		# medication_table = medication_table.loc[medication_table['drugordercancelled'] == 'No']
		# print('medication_table loaded successfully'.ljust(50) + str(np.round(len(medication_table)/1000000., 1)) + ' Mio rows | ' + str(int(medication_table.shape[1])) + ' cols')
		diagnosis_table = pd.read_csv(self.read_path + 'diagnosis.csv')
		print('diagnosis_table loaded successfully'.ljust(50) + str(np.round(len(diagnosis_table)/1000000., 1)) + ' Mio rows | ' + str(int(diagnosis_table.shape[1])) + ' cols')
		pasthistory_table = pd.read_csv(self.read_path + 'pastHistory.csv')
		print('pasthistory_table loaded successfully'.ljust(50) + str(np.round(len(pasthistory_table)/1000000., 1)) + ' Mio rows | ' + str(int(pasthistory_table.shape[1])) + ' cols')
		lab_table = pd.read_csv(self.read_path + 'lab.csv')
		print('lab_table loaded successfully'.ljust(50) + str(np.round(len(lab_table)/1000000., 1)) + ' Mio rows | ' + str(int(lab_table.shape[1])) + ' cols')
		apacheApsVar_table = pd.read_csv(self.read_path + 'apacheApsVar.csv')
		print('apacheApsVar_table loaded successfully'.ljust(50) + str(np.round(len(apacheApsVar_table)/1000000., 1)) + ' Mio rows | ' + str(int(apacheApsVar_table.shape[1])) + ' cols')
		print('\n\n')

		def get_aps_values(str_label, corrunitstayid):
			
			dummy = apacheApsVar_table[str_label].loc[apacheApsVar_table['patientunitstayid'] == corrunitstayid].values
			dummy = np.asarray(dummy)
			dummy = np.ravel(dummy)
			if dummy.shape[0] > 0:
				dummy = dummy[0]
			try:
				dummy = np.float(dummy)
			except TypeError:
				# print('Hello!')
				# print(dummy)
				dummy = 0.
			return dummy


		if self.num_patients < 0:
			num_patients_to_load = len(patientIDs)
		else:
			num_patients_to_load = self.num_patients

		patientIDs = patient_table['uniquepid'].unique()[:num_patients_to_load]

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=123)

		for train_index, test_index in stratified_splitter.split(
			np.zeros(len(patientIDs)), 
			np.zeros(len(patientIDs))):
				train_patient_ids = patientIDs[train_index]
				test_patient_ids = patientIDs[test_index]
		print('dataset split for training (' + str(len(train_patient_ids)) + ' patients) and validation (' + str(len(test_patient_ids)) + ' patients)\n\n')

		# data_df = []
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


				# correlated_pasthistory_ids = np.asarray(pasthistory_table['pasthistoryid'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				pasthistory_notetypes = np.asarray(pasthistory_table['pasthistorynotetype'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].unique())
				pasthistory_values = np.asarray(pasthistory_table['pasthistoryvalue'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].unique())

				# correlated_lab_ids = np.asarray(lab_table['labid'].loc[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				correlated_lab_type_ids = np.asarray(lab_table['labtypeid'].loc[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				correlated_lab_names = np.asarray(lab_table['labname'].loc[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				# correlated_lab_results = np.asarray(lab_table['labresult'].loc[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)


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

					# drug_codes_prescribed0 = medication_table['drughiclseqno'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
					# drug_codes_prescribed = []
					# for h in range(len(drug_codes_prescribed0)):
					# 	if str(drug_codes_prescribed0[h]) != 'nan':
					# 		drug_codes_prescribed.append(int(drug_codes_prescribed0[h]))

					icd9codes0 = diagnosis_table['icd9code'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
					icd9codes = []
					for h in range(len(icd9codes0)):
						if str(icd9codes0[h]) != 'nan':
							icd9codes.append(str(icd9codes0[h]))
				else:
					# drug_codes_prescribed = medication_table['drughiclseqno'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
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


				# medication_ids = np.asarray(medication_table['medicationid'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				# diagnosis_ids = np.asarray(diagnosis_table['diagnosisid'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				# drug_strings_prescribed = medication_table['drugname'].loc[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]].values

				icu_admission_time = np.abs(patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.)
				hospital_discharge_time = np.abs(patient_table['hospitaldischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.)
				icu_discharge_time = np.abs(patient_table['unitdischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.)

				lengthofstay = hospital_discharge_time
				lengthofICU = icu_discharge_time


				if lengthofstay > 24*5.:
					will_stay_long = 1.
				else:
					will_stay_long = 0.


				hospital_id = patient_table['hospitalid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()



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
					'length_of_icu': lengthofICU,
					'icu_discharge': icu_discharge_time,
					'will_return': will_return,
					'will_die': will_die,
					'will_readmit': will_readmit,
					'will_stay_long': will_stay_long,
					'survive_current_icu': survive_current_icu,
					'unit_readmission': unit_readmission,
					'visits_current_stay': max_visits_for_current_stay,
					'hospital_discharge_status': patient_table['hospitaldischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_admit_offset': icu_admission_time,
					'hospital_discharge_offset': hospital_discharge_time,
					'hospital_discharge_year': patient_table['hospitaldischargeyear'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'hospital_id': hospital_id,
					'unit_admit_source': patient_table['unitadmitsource'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_type': patient_table['unittype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_discharge_status': patient_table['unitdischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_discharge_offset': icu_discharge_time,
					'unit_discharge_location': patient_table['unitdischargelocation'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					'unit_stay_type': patient_table['unitstaytype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
					# 'lab_ids': correlated_lab_ids,
					'lab_type_ids': correlated_lab_type_ids,
					'lab_names': correlated_lab_names,
					# 'lab_results': correlated_lab_results,
					# 'medication_ids': medication_ids,
					# 'diagnosis_ids': diagnosis_ids,
					# 'drug_strings_prescribed': np.asarray(drug_strings_prescribed),
					# 'drug_codes_prescribed': np.asarray(drug_codes_prescribed),
					'diagnosis_activeUponDischarge': diagnosis_table['activeupondischarge'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'diagnosis_offset': diagnosis_table['diagnosisoffset'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					# 'diagnosis_priority': diagnosis_table['diagnosispriority'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'diagnosis_ICD9code': np.asarray(icd9codes),
					# 'diagnosis_string': diagnosis_table['diagnosisstring'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values,
					'pasthistory_notetypes': pasthistory_notetypes,
					'pasthistory_values': pasthistory_values,
					'intubated': get_aps_values('intubated', correlated_unitstay_ids[j]),
					'vent': get_aps_values('vent', correlated_unitstay_ids[j]),
					'dialysis': get_aps_values('dialysis', correlated_unitstay_ids[j]),
					'eyes': get_aps_values('eyes', correlated_unitstay_ids[j]),
					'motor': get_aps_values('motor', correlated_unitstay_ids[j]),
					'verbal': get_aps_values('verbal', correlated_unitstay_ids[j]),
					'meds': get_aps_values('meds', correlated_unitstay_ids[j]),
					'urine': get_aps_values('urine', correlated_unitstay_ids[j]),
					'wbc': get_aps_values('wbc', correlated_unitstay_ids[j]),
					'temperature': get_aps_values('temperature', correlated_unitstay_ids[j]),
					'respiratoryRate': get_aps_values('respiratoryrate', correlated_unitstay_ids[j]),
					'sodium': get_aps_values('sodium', correlated_unitstay_ids[j]),
					'meanBp': get_aps_values('meanbp', correlated_unitstay_ids[j]),
					'ph': get_aps_values('ph', correlated_unitstay_ids[j]),
					'hematocrit': get_aps_values('hematocrit', correlated_unitstay_ids[j]),
					'creatinine': get_aps_values('creatinine', correlated_unitstay_ids[j]),
					'albumin': get_aps_values('albumin', correlated_unitstay_ids[j]),
					'pao2': get_aps_values('pao2', correlated_unitstay_ids[j]),
					'pco2': get_aps_values('pco2', correlated_unitstay_ids[j]),
					'bun': get_aps_values('bun', correlated_unitstay_ids[j]),
					'glucose': get_aps_values('glucose', correlated_unitstay_ids[j]),
					'bilirubin': get_aps_values('bilirubin', correlated_unitstay_ids[j]),
					'fio2': get_aps_values('fio2', correlated_unitstay_ids[j]),
					})


		pbar.close()
		print('\n')


		pd.DataFrame(corr_id_df).to_csv(self.write_path)

		patient_table = []
		medication_table = []
		diagnosis_table = []
		pasthistory_table = []
		lab_table = []
		apacheApsVar_table = []

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
			# 'pasthistory_notetypes'
			# 'diagnosis_priority',
			# 'lab_type_ids',
			]

		array_features = [
			# 'drug_codes_prescribed',
			'diagnosis_ICD9code',
			'pasthistory_notetypes',
			'pasthistory_values',
			# 'diagnosis_priority',
			'lab_names',
			'lab_type_ids'
			]

		# array_features_unused = [
		# 	'drug_strings_prescribed',
		# 	'diagnosis_string',
		# 	'medication_ids',
		# 	'diagnosis_ids',
		# 	'lab_ids',
		# 	'lab_results',
		# 	]



		self.df_onehot = pd.get_dummies(self.dataframe, columns = categorical_feature_names, prefix = categorical_feature_names)
		
		self.process_array_cols(array_features)

		self.add_hospital_stats()
		
		# self.df_onehot.drop(columns=array_features_unused)

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

				if col_name in ['drug_strings_prescribed', 'diagnosis_ICD9code', 'diagnosis_string', 'lab_names', 'diagnosis_priority']:

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

				if col_name in ['medication_ids', 'diagnosis_ids', 'drug_codes_prescribed', 'lab_type_ids']:

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

	def add_hospital_stats(self):

		clinic_stats_df = []

		print('\nbuilding hospital stats dataframe...')
		pbar = tqdm(total=len(self.dataframe['hospital_id'].unique()))

		for clinic_id in self.dataframe['hospital_id'].unique():

			clinic_stats_df.append({
				'hospital_id': clinic_id,
				'will_die_mean': self.dataframe['will_die'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_readmit_mean': self.dataframe['will_readmit'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_return_mean': self.dataframe['will_return'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_stay_long_mean': self.dataframe['will_stay_long'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'survive_current_icu_mean': self.dataframe['survive_current_icu'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'unit_readmission_mean': self.dataframe['unit_readmission'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'length_of_stay_mean': self.dataframe['length_of_stay'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'length_of_stay_var': self.dataframe['length_of_stay'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'length_of_icu_mean': self.dataframe['length_of_icu'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'length_of_icu_var': self.dataframe['length_of_icu'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				})


			pbar.update(1)
		pbar.close()

		clinic_stats_df = pd.DataFrame(clinic_stats_df).reset_index(drop=True)


		print('\nattaching hospital stats dataframe to feature map...')
		pbar = tqdm(total=len(self.df_onehot['corr_id'].unique()))

		self.df_onehot['will_die_mean'] = 0.
		self.df_onehot['will_readmit_mean'] = 0.
		self.df_onehot['will_return_mean'] = 0.
		self.df_onehot['will_stay_long_mean'] = 0.
		self.df_onehot['survive_current_icu_mean'] = 0.
		self.df_onehot['unit_readmission_mean'] = 0.
		self.df_onehot['length_of_stay_mean'] = 0.
		self.df_onehot['length_of_stay_var'] = 0.
		self.df_onehot['length_of_icu_mean'] = 0.
		self.df_onehot['length_of_icu_var'] = 0.

		self.df_onehot.reset_index(drop=True)

		for stay_id in self.df_onehot['corr_id'].unique():

			hospital_id_dummy = self.df_onehot['hospital_id'].loc[self.df_onehot['corr_id'] == stay_id].values.item()

			dummy = clinic_stats_df['will_die_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['will_die_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['will_readmit_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['will_readmit_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['will_return_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['will_return_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['will_stay_long_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['will_stay_long_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['survive_current_icu_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['survive_current_icu_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['unit_readmission_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['unit_readmission_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['length_of_stay_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['length_of_stay_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['length_of_stay_var'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['length_of_stay_var'].loc[self.df_onehot['corr_id'] == stay_id] = dummy

			dummy = clinic_stats_df['length_of_icu_mean'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['length_of_icu_mean'].loc[self.df_onehot['corr_id'] == stay_id] = dummy
			
			dummy = clinic_stats_df['length_of_icu_var'].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
			self.df_onehot['length_of_icu_var'].loc[self.df_onehot['corr_id'] == stay_id] = dummy
			
			pbar.update(1)
		pbar.close()

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

		self.process_data_path = process_data_path
		self.target_features = target_features

		self.scaler_lo_icu = RobustScaler()
		self.scaler_lo_hospital = RobustScaler()
		self.scaler_features = RobustScaler()

		self.data_df = pd.read_csv(self.process_data_path).drop(columns='Unnamed: 0')
		self.data_df = self.data_df.loc[self.data_df['unit_discharge_offset'] != 0].fillna(0.)

		self.data_df['length_of_icu'].loc[self.data_df['length_of_icu'] > 1000.] = 1000.
		self.data_df['length_of_icu'].loc[self.data_df['length_of_icu'] < 1.] = 1.
		self.data_df['length_of_stay'].loc[self.data_df['length_of_stay'] > 5000.] = 5000.
		self.data_df['length_of_stay'].loc[self.data_df['length_of_stay'] < 1.] = 1.


		self.label_cols = [
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

		# self.remove_some_outliers()
		# self.consolidate_previous_ICUs()
		
		self.data_df_originial = self.data_df


		self.data_df[['length_of_icu', 'length_of_stay']] = np.log(self.data_df[['length_of_icu', 'length_of_stay']])
		self.data_df[['length_of_icu', 'length_of_stay']] = self.data_df[['length_of_icu', 'length_of_stay']].astype(float)

		self.scaler_lo_icu.fit(self.data_df['length_of_icu'].values.reshape(-1,1))
		self.data_df['length_of_icu'] = self.scaler_lo_icu.transform(self.data_df['length_of_icu'].values.reshape(-1,1))

		self.scaler_lo_hospital.fit(self.data_df['length_of_stay'].values.reshape(-1,1))
		self.data_df['length_of_stay'] = self.scaler_lo_hospital.transform(self.data_df['length_of_stay'].values.reshape(-1,1))

		self.data_df.fillna(0.)



		self.check_data()

		# self.check_data()

		self.training_data, self.num_input_features, self.num_output_features = self.split_data()


	def consolidate_previous_ICUs(self):

		consolidation_keys = self.data_df.keys()
		consolidation_keys.drop(self.label_cols)
		consolidation_keys.drop([
			'age',
			'gender_Female',
			'gender_Male',
			# 'gender_Unknown',
			'ethnicity_African American',
			'ethnicity_Asian',
			'ethnicity_Hispanic',
			'ethnicity_Native American',
			# 'ethnicity_Other/Unknown',
			# 'ethnicity_Unknown',
			'will_die_mean',
			'will_readmit_mean',
			'will_return_mean',
			'will_stay_long_mean',
			'survive_current_icu_mean',
			'unit_readmission_mean',
			'length_of_stay_mean',
			'length_of_stay_var',
			'length_of_icu_mean',
			'length_of_icu_var',
			])


		corr_ids = self.data_df['corr_id'].unique()

		print('\nconsolidating previous ICU features...')
		pbar = tqdm(total=len(corr_ids))
		for correlated_id in corr_ids:
			pbar.update(1)
			corr_health_id = self.data_df['health_system_id'].loc[self.data_df['corr_id'] == correlated_id].values.item()
			corr_visit_number = self.data_df['visit_number'].loc[self.data_df['corr_id'] == correlated_id].values.item()

			other_stays = self.data_df['health_system_id'] == corr_health_id
			previous_stays = self.data_df['visit_number'] < corr_visit_number
			hist_index = np.logical_and(other_stays, previous_stays)

			if corr_visit_number > 1:

				dummy = self.data_df.loc[self.data_df['health_system_id'] == corr_health_id]
				dummy = dummy[consolidation_keys].loc[dummy['visit_number'] < corr_visit_number].sum(0)

				try:
					self.data_df[consolidation_keys].loc[self.data_df['corr_id'] == correlated_id] += dummy
				except TypeError:
					continue

		pbar.close()

		self.data_df.to_csv(self.process_data_path[:-4] + '_consolidated.csv')


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

		print('\n\nDataFrame:')
		print(self.data_df_originial.drop(columns=self.label_cols).shape)
		print('*******************************')
		print(self.data_df_originial.drop(columns=self.label_cols).sum())
		print('*******************************')
		print(self.data_df_originial.drop(columns=self.label_cols))
		print('*******************************')
		for keydummy in self.data_df_originial.drop(columns=self.label_cols).keys():
			print(keydummy)
			print(self.data_df_originial[keydummy].sum())
			print(self.data_df_originial[keydummy].dtype)
			print('----')
		print('*******************************')

		dummy_patient_info = []
		earlystopper = 0
		previous_patient_id = 0
		for corr_id in self.data_df_originial['corr_id'].values:

			if earlystopper < 50:

				new_patient_id = self.data_df_originial['patient_id'].loc[self.data_df_originial['corr_id'] == corr_id].values.item()

				if new_patient_id != previous_patient_id:
					print('\npatient_id:' + str(self.data_df_originial['patient_id'].loc[self.data_df_originial['corr_id'] == corr_id].values.item()) + '\n', pd.DataFrame(dummy_patient_info))
					dummy_patient_info = []
					earlystopper += 1

		
				dummy_patient_info.append({
					'patient_id': self.data_df_originial['patient_id'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'system_id': self.data_df_originial['health_system_id'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'case_id': corr_id,
					'L hospital': self.data_df_originial['length_of_stay'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'L ICU': self.data_df_originial['length_of_icu'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'ICU end': self.data_df_originial['icu_discharge'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'return': self.data_df_originial['will_return'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'survive_icu': self.data_df_originial['survive_current_icu'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'will_die': self.data_df_originial['will_die'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'visit': self.data_df_originial['visit_number'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'readmit': self.data_df_originial['unit_readmission'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'will_readmit': self.data_df_originial['will_readmit'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
					'will_stay_long': self.data_df_originial['will_stay_long'].loc[self.data_df_originial['corr_id'] == corr_id].values.item(),
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

		feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.).values
		feature_map = self.scaler_features.fit_transform(feature_map)
		feature_map = np.nan_to_num(feature_map)

		x_training = feature_map[self.data_df['data_set_ref'] == 'training']
		x_validation = feature_map[self.data_df['data_set_ref'] == 'validation']
		y_training = self.data_df[self.target_features].loc[self.data_df['data_set_ref'] == 'training']
		y_validation = self.data_df[self.target_features].loc[self.data_df['data_set_ref'] == 'validation']


		data_container = {
				'x_full': self.data_df.drop(columns = self.label_cols).values,
				'x_train': x_training,
				'x_test': x_validation,
				'y_full': self.data_df[self.target_features].values,
				'y_train': y_training,
				'y_test': y_validation}

		return data_container, x_training.shape[1], y_training.shape[1]

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

	def get_train_iterator(self, batch_size, target_label):

		dummy = pd.concat([
			self.training_data['y_train'][target_label],
			1-self.training_data['y_train'][target_label],
			], axis=1).values
		
		# dummy = self.training_data['y_train'][target_label].values
		
		return DataLoader(
			DataSetIterator(
				self.training_data['x_train'], 
				dummy), 
			batch_size=batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)

	def get_test_iterator(self, batch_size, target_label):

		dummy = pd.concat([
			self.training_data['y_test'][target_label],
			1-self.training_data['y_test'][target_label],
			], axis=1).values

		# dummy = self.training_data['y_test'][target_label].values

		return DataLoader(
			DataSetIterator(
				self.training_data['x_test'], 
				dummy), 
			batch_size=batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)


