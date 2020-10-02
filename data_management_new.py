import math, re, os
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

class eICU_DataLoader():

	def __init__(self, args):

		self.args = args
		self.read_path = self.args.eICU_path
		self.write_path = self.args.mydata_path
		self.num_patients = self.args.num_patients_to_load

		self.build_patient_matrix()

	def build_patient_matrix(self):

		remove_nans_from_codes = True

		patient_table = pd.read_csv(self.read_path + 'patient.csv')
		patient_table = patient_table.loc[patient_table['age'] != 'NaN']
		patient_table = patient_table.loc[patient_table['age'] != 'nan']
		patient_table = patient_table.loc[patient_table['gender'] != 'Unknown']

		patient_table.sort_values('hospitalid', ascending=True)

		print('\n\n')
		print('patient_table loaded successfully'.ljust(50) + str(np.round(len(patient_table)/1000000., 1)) + ' Mio rows | ' + str(int(patient_table.shape[1])) + ' cols')
		# medication_table = pd.read_csv(self.read_path + 'medication.csv', low_memory=False)
		# medication_table = medication_table.loc[medication_table['drugordercancelled'] == 'No']
		# print('medication_table loaded successfully'.ljust(50) + str(np.round(len(medication_table)/1000000., 1)) + ' Mio rows | ' + str(int(medication_table.shape[1])) + ' cols')
		diagnosis_table = pd.read_csv(self.read_path + 'diagnosis.csv')
		print('diagnosis_table loaded successfully'.ljust(50) + str(np.round(len(diagnosis_table)/1000000., 1)) + ' Mio rows | ' + str(int(diagnosis_table.shape[1])) + ' cols')
		# pasthistory_table = pd.read_csv(self.read_path + 'pastHistory.csv')
		# print('pasthistory_table loaded successfully'.ljust(50) + str(np.round(len(pasthistory_table)/1000000., 1)) + ' Mio rows | ' + str(int(pasthistory_table.shape[1])) + ' cols')
		# lab_table = pd.read_csv(self.read_path + 'lab.csv')
		# print('lab_table loaded successfully'.ljust(50) + str(np.round(len(lab_table)/1000000., 1)) + ' Mio rows | ' + str(int(lab_table.shape[1])) + ' cols')
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
				dummy = 0.
			return dummy


		if self.num_patients < 0:
			num_patients_to_load = len(patient_table['uniquepid'].unique())
		else:
			num_patients_to_load = self.num_patients

		patientIDs = patient_table['uniquepid'].unique()[:num_patients_to_load]

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=123)


		corr_id_df = []
		print('looping through patient IDs in loaded tables to build a consolidated matrix...')
		pbarfreq = 10
		pbar = tqdm(total=num_patients_to_load)
		for i in range(num_patients_to_load):

			if i % pbarfreq == 0: pbar.update(pbarfreq)

			patient = patientIDs[i]

			correlated_unitstay_ids = np.asarray(patient_table['patientunitstayid'].loc[patient_table['uniquepid'] == patient].values)

			for j in range(len(correlated_unitstay_ids)):

				# pasthistory_notetypes = np.asarray(pasthistory_table['pasthistorynotetype'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].unique())
				# pasthistory_values = np.asarray(pasthistory_table['pasthistoryvalue'].loc[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]].unique())

				# correlated_lab_type_ids = np.asarray(lab_table['labtypeid'].loc[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)
				# correlated_lab_names = np.asarray(lab_table['labname'].loc[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]].values)

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

					icd9codes0 = diagnosis_table['icd9code'].loc[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]].values
					icd9codes = []
					for h in range(len(icd9codes0)):
						if str(icd9codes0[h]) != 'nan':
							icd9codes.append(str(icd9codes0[h]))
				else:
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

				icu_admission_time = np.abs(patient_table['hospitaladmitoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.)
				hospital_discharge_time = np.abs(patient_table['hospitaldischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.)
				icu_discharge_time = np.abs(patient_table['unitdischargeoffset'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item() / 60.)

				weight_dummy = np.float(patient_table['admissionweight'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item())
				height_dummy = np.float(patient_table['admissionheight'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()) / 100.
				bmi_dummy = weight_dummy / ((height_dummy * height_dummy) + 1e-6)
				if bmi_dummy > 200:
					bmi_dummy = 0.


				lengthofstay = hospital_discharge_time
				lengthofICU = icu_discharge_time


				if lengthofstay > 24*5.:
					will_stay_long = 1.
				else:
					will_stay_long = 0.


				hospital_id = patient_table['hospitalid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()


				if patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item() == 'Female':
					gender_dummy = 0.
				else:
					gender_dummy = 1.


				corr_id_df.append(
					{
					'patient_id': patient,
					'health_system_id': current_health_sys_id,
					'corr_id': correlated_unitstay_ids[j],
					'gender': gender_dummy,
					'age': age_dummy,
					'weight': weight_dummy,
					'height': height_dummy,
					'bmi': bmi_dummy,
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
					# 'lab_type_ids': correlated_lab_type_ids,
					# 'lab_names': correlated_lab_names,
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
					# 'pasthistory_notetypes': pasthistory_notetypes,
					# 'pasthistory_values': pasthistory_values,
					'aps_intubated': get_aps_values('intubated', correlated_unitstay_ids[j]),
					'aps_vent': get_aps_values('vent', correlated_unitstay_ids[j]),
					'aps_dialysis': get_aps_values('dialysis', correlated_unitstay_ids[j]),
					'aps_eyes': get_aps_values('eyes', correlated_unitstay_ids[j]),
					'aps_motor': get_aps_values('motor', correlated_unitstay_ids[j]),
					'aps_verbal': get_aps_values('verbal', correlated_unitstay_ids[j]),
					'aps_meds': get_aps_values('meds', correlated_unitstay_ids[j]),
					'aps_urine': get_aps_values('urine', correlated_unitstay_ids[j]),
					'aps_wbc': get_aps_values('wbc', correlated_unitstay_ids[j]),
					'aps_temperature': get_aps_values('temperature', correlated_unitstay_ids[j]),
					'aps_respiratoryRate': get_aps_values('respiratoryrate', correlated_unitstay_ids[j]),
					'aps_sodium': get_aps_values('sodium', correlated_unitstay_ids[j]),
					'aps_heartrate': get_aps_values('heartrate', correlated_unitstay_ids[j]),
					'aps_meanBp': get_aps_values('meanbp', correlated_unitstay_ids[j]),
					'aps_ph': get_aps_values('ph', correlated_unitstay_ids[j]),
					'aps_hematocrit': get_aps_values('hematocrit', correlated_unitstay_ids[j]),
					'aps_creatinine': get_aps_values('creatinine', correlated_unitstay_ids[j]),
					'aps_albumin': get_aps_values('albumin', correlated_unitstay_ids[j]),
					'aps_pao2': get_aps_values('pao2', correlated_unitstay_ids[j]),
					'aps_pco2': get_aps_values('pco2', correlated_unitstay_ids[j]),
					'aps_bun': get_aps_values('bun', correlated_unitstay_ids[j]),
					'aps_glucose': get_aps_values('glucose', correlated_unitstay_ids[j]),
					'aps_bilirubin': get_aps_values('bilirubin', correlated_unitstay_ids[j]),
					'aps_fio2': get_aps_values('fio2', correlated_unitstay_ids[j]),
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

	def __init__(self, args):

		self.args = args
		self.read_path = self.args.mydata_path
		self.write_path = self.args.mydata_path_processed

		self.min_patients_per_client = self.args.min_patients_per_hospital

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
			# 'pasthistory_notetypes',
			# 'pasthistory_values',
			# 'diagnosis_priority',
			# 'lab_names',
			# 'lab_type_ids'
			]


		self.feature_df = pd.get_dummies(self.dataframe, columns = categorical_feature_names, prefix = categorical_feature_names)
		

		self.process_array_cols(array_features)
		
		if self.args.integrate_past_cases:
			self.consolidate_previous_ICUs()

		self.add_hospital_stats()

		self.feature_df = self.feature_df.loc[self.feature_df['num_patients'] >= self.min_patients_per_client]

		# self.make_federated_sets()

		self.feature_df.to_csv(self.write_path)

	def process_array_cols(self, col_names):

		print_out = False
		progbar = True
		pbarfreq = 10
		pbarcounter = 0

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


			self.feature_df.drop(columns=col_name)
			self.feature_df = pd.merge(self.feature_df, dummy_df, on='corr_id')

			if print_out: 
				print(
					'\n\n************************************\ndf_onehot DataFrame:\n', 
					self.feature_df, 
					'\n************************************\n\n\n')

	def consolidate_previous_ICUs(self):
		
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

		consolidation_keys = self.feature_df.keys()
		consolidation_keys.drop(label_cols)
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
			# 'will_die_mean',
			# 'will_readmit_mean',
			# 'will_return_mean',
			# 'will_stay_long_mean',
			# 'survive_current_icu_mean',
			# 'unit_readmission_mean',
			# 'length_of_stay_mean',
			# 'length_of_stay_var',
			# 'length_of_icu_mean',
			# 'length_of_icu_var',
			])


		corr_ids = self.feature_df['corr_id'].unique()

		print('\nconsolidating previous ICU features...')
		pbar = tqdm(total=len(corr_ids))
		for correlated_id in corr_ids:
			pbar.update(1)
			corr_health_id = self.feature_df['health_system_id'].loc[self.feature_df['corr_id'] == correlated_id].values.item()
			corr_visit_number = self.feature_df['visit_number'].loc[self.feature_df['corr_id'] == correlated_id].values.item()

			other_stays = self.feature_df['health_system_id'] == corr_health_id
			previous_stays = self.feature_df['visit_number'] < corr_visit_number
			hist_index = np.logical_and(other_stays, previous_stays)

			if corr_visit_number > 1:

				dummy = self.feature_df.loc[self.feature_df['health_system_id'] == corr_health_id]
				dummy = dummy[consolidation_keys].loc[dummy['visit_number'] < corr_visit_number].sum(0)

				try:
					self.feature_df[consolidation_keys].loc[self.feature_df['corr_id'] == correlated_id] += dummy
				except TypeError:
					continue

		pbar.close()

		self.feature_df.to_csv(self.write_path[:-4] + '_consolidated.csv')

	def add_hospital_stats(self):

		clinic_stats_df = []

		print('\nbuilding hospital stats dataframe...')
		pbar = tqdm(total=len(self.dataframe['hospital_id'].unique()))

		for clinic_id in self.dataframe['hospital_id'].unique():

			ethnicity_dummy_df = pd.get_dummies(self.dataframe['ethnicity'][self.dataframe['hospital_id'] == clinic_id], dummy_na=False)
			ethnicity_dummy_df = pd.DataFrame(ethnicity_dummy_df)

			if 'Unknown' in ethnicity_dummy_df.keys():
				if 'Other/Unknown' in ethnicity_dummy_df.keys():
					ethnicity_dummy_df['Unknown'] += ethnicity_dummy_df['Other/Unknown']
					ethnicity_dummy_df.drop(columns='Other/Unknown')
			else: ethnicity_dummy_df['Unknown'] = 0

			if 'African American' not in ethnicity_dummy_df.keys(): ethnicity_dummy_df['African American'] = 0
			if 'Caucasian' not in ethnicity_dummy_df.keys(): ethnicity_dummy_df['Caucasian'] = 0
			if 'Hispanic' not in ethnicity_dummy_df.keys(): ethnicity_dummy_df['Hispanic'] = 0
			if 'Native American' not in ethnicity_dummy_df.keys(): ethnicity_dummy_df['Native American'] = 0
			if 'Asian' not in ethnicity_dummy_df.keys(): ethnicity_dummy_df['Asian'] = 0

			num_patients_dummy = len(self.dataframe[self.dataframe['hospital_id'] == clinic_id])
			clinic_stats_df.append({
				'hospital_id': clinic_id,
				'num_patients': num_patients_dummy,
				'gender_mean': self.dataframe['gender'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'ethnicity_caucasian_mean': ethnicity_dummy_df['Caucasian'].mean(),
				'ethnicity_africanamerican_mean': ethnicity_dummy_df['African American'].mean(),
				'ethnicity_nativeamerican_mean': ethnicity_dummy_df['Native American'].mean(),
				'ethnicity_asian_mean': ethnicity_dummy_df['Asian'].mean(),
				'ethnicity_hispanic_mean': ethnicity_dummy_df['Hispanic'].mean(),
				'ethnicity_unknown_mean': ethnicity_dummy_df['Unknown'].mean(),
				'age_mean': self.dataframe['age'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'age_var': self.dataframe['age'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'age_std': self.dataframe['age'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'bmi_mean': self.dataframe['bmi'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'bmi_var': self.dataframe['bmi'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'bmi_std': self.dataframe['bmi'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'weight_mean': self.dataframe['weight'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'weight_var': self.dataframe['weight'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'weight_std': self.dataframe['weight'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'height_mean': self.dataframe['height'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'height_var': self.dataframe['height'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'height_std': self.dataframe['height'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'will_die_mean': self.dataframe['will_die'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_die_var': self.dataframe['will_die'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'will_die_std': self.dataframe['will_die'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'will_readmit_mean': self.dataframe['will_readmit'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_readmit_var': self.dataframe['will_readmit'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'will_readmit_std': self.dataframe['will_readmit'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'will_return_mean': self.dataframe['will_return'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_return_var': self.dataframe['will_return'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'will_return_std': self.dataframe['will_return'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'will_stay_long_mean': self.dataframe['will_stay_long'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'will_stay_long_var': self.dataframe['will_stay_long'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'will_stay_long_std': self.dataframe['will_stay_long'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'survive_current_icu_mean': self.dataframe['survive_current_icu'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'survive_current_icu_var': self.dataframe['survive_current_icu'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'survive_current_icu_std': self.dataframe['survive_current_icu'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'unit_readmission_mean': self.dataframe['unit_readmission'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'unit_readmission_var': self.dataframe['unit_readmission'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'unit_readmission_std': self.dataframe['unit_readmission'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'length_of_stay_mean': self.dataframe['length_of_stay'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'length_of_stay_var': self.dataframe['length_of_stay'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'length_of_stay_std': self.dataframe['length_of_stay'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'length_of_icu_mean': self.dataframe['length_of_icu'].loc[self.dataframe['hospital_id'] == clinic_id].mean(),
				'length_of_icu_var': self.dataframe['length_of_icu'].loc[self.dataframe['hospital_id'] == clinic_id].var(),
				'length_of_icu_std': self.dataframe['length_of_icu'].loc[self.dataframe['hospital_id'] == clinic_id].std(),
				'aps_intubated_mean': self.dataframe['aps_intubated'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_vent_mean': self.dataframe['aps_vent'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_dialysis_mean': self.dataframe['aps_dialysis'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_eyes_mean': self.dataframe['aps_eyes'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_motor_mean': self.dataframe['aps_motor'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_verbal_mean': self.dataframe['aps_verbal'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_meds_mean': self.dataframe['aps_meds'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_urine_mean': self.dataframe['aps_urine'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_wbc_mean': self.dataframe['aps_wbc'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_temperature_mean': self.dataframe['aps_temperature'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'aps_respiratoryRate_mean': self.dataframe['aps_respiratoryRate'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'aps_sodium_mean': self.dataframe['aps_sodium'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_heartrate_mean': self.dataframe['aps_heartrate'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_meanBp_mean': self.dataframe['aps_meanBp'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_ph_mean': self.dataframe['aps_ph'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_hematocrit_mean': self.dataframe['aps_hematocrit'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_creatinine_mean': self.dataframe['aps_creatinine'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_albumin_mean': self.dataframe['aps_albumin'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_pao2_mean': self.dataframe['aps_pao2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_pco2_mean': self.dataframe['aps_pco2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_bun_mean': self.dataframe['aps_bun'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_glucose_mean': self.dataframe['aps_glucose'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_bilirubin_mean': self.dataframe['aps_bilirubin'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_fio2_mean': self.dataframe['aps_fio2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),				
				'aps_intubated_std': self.dataframe['aps_intubated'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_vent_std': self.dataframe['aps_vent'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_dialysis_std': self.dataframe['aps_dialysis'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_eyes_std': self.dataframe['aps_eyes'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_motor_std': self.dataframe['aps_motor'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_verbal_std': self.dataframe['aps_verbal'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_meds_std': self.dataframe['aps_meds'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_urine_std': self.dataframe['aps_urine'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_wbc_std': self.dataframe['aps_wbc'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_temperature_std': self.dataframe['aps_temperature'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'aps_respiratoryRate_std': self.dataframe['aps_respiratoryRate'].loc[self.dataframe['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'aps_sodium_std': self.dataframe['aps_sodium'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_heartrate_std': self.dataframe['aps_heartrate'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_meanBp_std': self.dataframe['aps_meanBp'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_ph_mean': self.dataframe['aps_ph'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_hematocrit_std': self.dataframe['aps_hematocrit'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_creatinine_std': self.dataframe['aps_creatinine'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_albumin_std': self.dataframe['aps_albumin'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_pao2_std': self.dataframe['aps_pao2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_pco2_std': self.dataframe['aps_pco2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_bun_std': self.dataframe['aps_bun'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_glucose_std': self.dataframe['aps_glucose'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_bilirubin_std': self.dataframe['aps_bilirubin'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_fio2_std': self.dataframe['aps_fio2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_intubated_missing': self.dataframe['aps_intubated'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_vent_missing': self.dataframe['aps_vent'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_dialysis_missing': self.dataframe['aps_dialysis'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_eyes_missing': self.dataframe['aps_eyes'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_motor_missing': self.dataframe['aps_motor'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_verbal_missing': self.dataframe['aps_verbal'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_meds_missing': self.dataframe['aps_meds'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_urine_missing': self.dataframe['aps_urine'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_wbc_missing': self.dataframe['aps_wbc'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_temperature_missing': self.dataframe['aps_temperature'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_respiratoryRate_missing': self.dataframe['aps_respiratoryRate'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_sodium_missing': self.dataframe['aps_sodium'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_heartrate_missing': self.dataframe['aps_heartrate'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_meanBp_missing': self.dataframe['aps_meanBp'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_ph_mean': self.dataframe['aps_ph'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_hematocrit_missing': self.dataframe['aps_hematocrit'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_creatinine_missing': self.dataframe['aps_creatinine'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_albumin_missing': self.dataframe['aps_albumin'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_pao2_missing': self.dataframe['aps_pao2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_pco2_missing': self.dataframe['aps_pco2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_bun_missing': self.dataframe['aps_bun'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_glucose_missing': self.dataframe['aps_glucose'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_bilirubin_missing': self.dataframe['aps_bilirubin'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_fio2_missing': self.dataframe['aps_fio2'].loc[self.dataframe['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				})



			pbar.update(1)
		pbar.close()

		clinic_stats_df = pd.DataFrame(clinic_stats_df)
		clinic_stats_df.to_csv('clinic_stats.csv')
		clinic_stats_df.reset_index(drop=True)


		print('\nattaching hospital stats dataframe to feature map...')
		pbar = tqdm(total=len(self.feature_df['corr_id'].unique()))

		extra_hospital_keys = clinic_stats_df.keys().values
		# self.feature_df[clinic_stats_df.keys().values] = 0.
		for extra_hospital_key in extra_hospital_keys:
			if extra_hospital_key != 'hospital_id':
				self.feature_df[extra_hospital_key] = 0.



		self.feature_df.reset_index(drop=True)

		for stay_id in self.feature_df['corr_id'].unique():

			hospital_id_dummy = self.feature_df['hospital_id'].loc[self.feature_df['corr_id'] == stay_id].values.item()

			for clinic_key in extra_hospital_keys:

				if clinic_key != 'hospital_id':

					dummy = clinic_stats_df[clinic_key].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
					self.feature_df[clinic_key].loc[self.feature_df['corr_id'] == stay_id] = dummy

			pbar.update(1)
		pbar.close()

	def make_federated_sets(self):

		print('\ncomputing federated datasets...')
		pbar = tqdm(total=len(self.feature_df['hospital_id'].unique()))

		try: os.makedirs('../mydata/federated/')
		except FileExistsError: pass

		for hospital_number in self.feature_df['hospital_id'].unique():

			print('saving data for hospital ' + str(hospital_number))
			print('shape: ', np.asarray(self.feature_df[self.feature_df['hospital_id'] == hospital_number]).shape, '\n----------------------')

			self.feature_df[self.feature_df['hospital_id'] == hospital_number].to_csv('../mydata/federated/hospital_' + str(hospital_number) + '.csv')

			pbar.update(1)
		pbar.close()

class ICD10code_transformer():

	def __init__(self, args):

		self.args = args
		self.feature_df = pd.read_csv(self.args.mydata_path_processed)

		self.diagnosis_table = self.transfom_ICD10_codes()
		self.diagnosis_table.to_csv(self.args.diag_table_path)

	def transfom_ICD10_codes(self):

		diagnosis_df = self.feature_df[[
			'patient_id',
			'health_system_id',
			'corr_id',
			'hospital_id',
			'diagnosis_ICD9code',
			]]

		icd10table = []

		for i in range(len(diagnosis_df)):

			missing_icd10_diagnosis = 0
			if len(diagnosis_df['diagnosis_ICD9code'].iloc[i]) <= 2:
				missing_icd10_diagnosis += 1

			else:

				splitted_entry = re.split("'", diagnosis_df['diagnosis_ICD9code'].iloc[i])


				for dummy in splitted_entry:

					if len(dummy) < 4:

						splitted_entry.remove(dummy)

				# print(re.split(',', str(splitted_entry)))

				splitted_entry = re.split(', ', str(splitted_entry).translate({ord(c): None for c in "'!@#$[]"}))


				for dummy in splitted_entry:

					if len(dummy) < 3:
						splitted_entry.remove(dummy)

					if not str(dummy).upper().isupper():
						splitted_entry.remove(dummy)

				for entry in splitted_entry:

					if entry[0].isalpha() and entry[1:3].isnumeric():
						icd10code_letter = entry[0]
						icd10code_number = entry[1:3]
					else: continue

					if entry[-2:].isnumeric():
						if int(entry[-2]) == 0:
							icd10code_decimal = entry[-1:]
						else:
							icd10code_decimal = entry[-2:]
					elif entry[-1:].isnumeric():
						icd10code_decimal = entry[-1:]
					else: continue


					icd10table.append({
						'patient_id': diagnosis_df['patient_id'].iloc[i],
						'health_system_id': diagnosis_df['health_system_id'].iloc[i],
						'corr_id': diagnosis_df['corr_id'].iloc[i],
						'hospital_id': diagnosis_df['hospital_id'].iloc[i],
						'icd10code_letter': icd10code_letter,
						'icd10code_number': icd10code_number,
						'icd10code_decimal': icd10code_decimal,
						})

		return pd.DataFrame(icd10table)

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

	def __init__(self, args):

		self.args = args
		try:
			self.process_data_path = self.args.mydata_path_processed
			self.target_features = self.args.target_label
			self.train_split = self.args.train_split
			self.split_strategy = self.args.split_strategy
		except AttributeError:
			self.process_data_path = self.args['mydata_path_processed']
			self.target_features = self.args['target_label']
			self.train_split = self.args['train_split']
			self.split_strategy = self.args['split_strategy']


		self.scaler_lo_icu = RobustScaler()
		self.scaler_lo_hospital = RobustScaler()
		self.scaler_features = RobustScaler()

		self.data_df = pd.read_csv(self.process_data_path).drop(columns='Unnamed: 0')
		self.data_df = self.data_df.loc[self.data_df['unit_discharge_offset'] != 0].fillna(0.)

		self.data_df['length_of_icu'].loc[self.data_df['length_of_icu'] > 1000.] = 1000.
		self.data_df['length_of_icu'].loc[self.data_df['length_of_icu'] < 1.] = 1.
		self.data_df['length_of_stay'].loc[self.data_df['length_of_stay'] > 5000.] = 5000.
		self.data_df['length_of_stay'].loc[self.data_df['length_of_stay'] < 1.] = 1.

		# self.data_df = self.data_df.loc[self.data_df['num_patients'] >= self.args.min_patients_per_hospital]

		self.label_cols = [
			# doesnt make sense to include or not properly formatted cols
			'patient_id',
			'health_system_id',
			'corr_id',
			# 'data_set_ref',
			# 'medication_ids',
			# 'drug_strings_prescribed',
			# 'drug_codes_prescribed',
			# 'diagnosis_string',
			# 'pasthistory_notetypes',
			# 'pasthistory_values',
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
			# 'lab_type_ids',
			# 'lab_names',
			]

		self.data_container, self.sampling_df = self.get_data()

	def split_data(self):

		unique_hospital_ids = self.data_df['hospital_id'].unique()

		# feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.).values
		# feature_map = np.nan_to_num(feature_map)
		feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.)

		# y_full = np.nan_to_num(self.data_df[self.target_features].values)
		y_full = self.data_df[self.target_features]

		train_ids, val_ids, test_ids = [], [], []
		train_idx, val_idx, test_idx = [], [], []

		dummyrunner = 0

		sampling_df = []

		for hosp_id in unique_hospital_ids:

			hospital_dummy_df = self.data_df[['hospital_id', 'patient_id', 'corr_id']].loc[self.data_df['hospital_id'] == hosp_id]
			
			train_frac, val_frac, test_frac = np.split(
				hospital_dummy_df['corr_id'].sample(frac=1.), 
				[
				int(self.train_split*np.asarray(hospital_dummy_df).shape[0]), 
				int((1-(.5*(1-self.train_split)))*np.asarray(hospital_dummy_df).shape[0])
				])

			sampling_df.append({
				'hospital_id': hosp_id,
				'train_ids': train_frac,
				'val_ids': val_frac,
				'test_ids': test_frac,
				})

			if dummyrunner == 0:
				train_ids = np.reshape(train_frac.values, (-1,1))
				val_ids = np.reshape(val_frac.values, (-1,1))
				test_ids = np.reshape(test_frac.values, (-1,1))
				dummyrunner += 1
			else:
				train_ids = np.concatenate((train_ids, np.reshape(train_frac.values, (-1,1))), axis=0)
				val_ids = np.concatenate((val_ids, np.reshape(val_frac.values, (-1,1))), axis=0)
				test_ids = np.concatenate((test_ids, np.reshape(test_frac.values, (-1,1))), axis=0)

		for i in range(self.data_df.shape[0]):

			if self.data_df['corr_id'].iloc[i] in train_ids:
				train_idx.append(i)
			if self.data_df['corr_id'].iloc[i] in val_ids:
				val_idx.append(i)
			if self.data_df['corr_id'].iloc[i] in test_ids:
				test_idx.append(i)

		train_idx = np.reshape(np.asarray(train_idx), (-1))
		val_idx = np.reshape(np.asarray(val_idx), (-1))
		test_idx = np.reshape(np.asarray(test_idx), (-1))

		x_train = feature_map.iloc[train_idx,:]
		x_val = feature_map.iloc[val_idx,:]
		x_test = feature_map.iloc[test_idx,:]

		y_train = y_full.iloc[train_idx]
		y_val = y_full.iloc[val_idx]
		y_test = y_full.iloc[test_idx]

		data_container = {
			'x_full': feature_map,
			'x_train': x_train,
			'x_val': x_val,
			'x_test': x_test,
			'y_full': y_full,
			'y_train': y_train,
			'y_val': y_val,
			'y_test': y_test,
		}

		return data_container, pd.DataFrame(sampling_df)

	def second_split(self):

		unique_hospital_ids = self.data_df['hospital_id'].unique()

		# feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.).values
		# feature_map = np.nan_to_num(feature_map)
		feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.)

		# y_full = np.nan_to_num(self.data_df[self.target_features].values)
		y_full = self.data_df[self.target_features]

		train_ids, val_ids, test_ids = [], [], []
		train_idx, val_idx, test_idx = [], [], []

		dummyrunner = 0

		sampling_df = []

		for hosp_id in unique_hospital_ids:

			hospital_dummy_df = self.data_df[['hospital_id', 'patient_id', 'corr_id']].loc[self.data_df['hospital_id'] == hosp_id]

			positives_dummy = self.data_df[self.target_features].loc[self.data_df['hospital_id'] == hosp_id]
			
			try:
				stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.5, random_state=87)
				for train_index, test_index in stratified_splitter.split(
					np.zeros(len(positives_dummy)), 
					positives_dummy):
						val_frac = hospital_dummy_df['corr_id'].values[train_index]
						test_frac = hospital_dummy_df['corr_id'].values[test_index]
			except ValueError:
				stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.5, random_state=87)
				for train_index, test_index in stratified_splitter.split(
					np.zeros(len(positives_dummy)), 
					np.zeros(len(positives_dummy))):
						val_frac = hospital_dummy_df['corr_id'].values[train_index]
						test_frac = hospital_dummy_df['corr_id'].values[test_index]

			sampling_df.append({
				'hospital_id': hosp_id,
				# 'train_ids': train_frac,
				'val_ids': val_frac,
				'test_ids': test_frac,
				})

			if dummyrunner == 0:
				val_ids = np.reshape(val_frac, (-1,1))
				test_ids = np.reshape(test_frac, (-1,1))
				dummyrunner += 1
			else:
				val_ids = np.concatenate((val_ids, np.reshape(val_frac, (-1,1))), axis=0)
				test_ids = np.concatenate((test_ids, np.reshape(test_frac, (-1,1))), axis=0)

		for i in range(self.data_df.shape[0]):

			if self.data_df['corr_id'].iloc[i] in val_ids:
				val_idx.append(i)
			if self.data_df['corr_id'].iloc[i] in test_ids:
				test_idx.append(i)

		val_idx = np.reshape(np.asarray(val_idx), (-1))
		test_idx = np.reshape(np.asarray(test_idx), (-1))

		x_val = feature_map.iloc[val_idx,:]
		x_test = feature_map.iloc[test_idx,:]

		y_val = y_full.iloc[val_idx]
		y_test = y_full.iloc[test_idx]

		data_container = {
			'x_full': feature_map,
			'x_train': feature_map,
			'x_val': x_val,
			'x_test': x_test,
			'y_full': y_full,
			'y_train': y_full,
			'y_val': y_val,
			'y_test': y_test,
		}

		return data_container, pd.DataFrame(sampling_df)
	
	def get_data(self):

		if self.split_strategy == 'trainN_testN':
			data_container, sampling_df = self.split_data()
		elif self.split_strategy == 'trainNminus1_test1':
			data_container, sampling_df = self.second_split()

		return data_container, sampling_df

	def get_full_train_data(self):

		return self.data_container['x_train'].values, np.reshape(self.data_container['y_train'].values, (-1,1))

	def get_full_val_data(self):

		return self.data_container['x_val'].values, np.reshape(self.data_container['y_val'].values, (-1,1))

	def get_full_test_data(self):

		return self.data_container['x_test'].values, np.reshape(self.data_container['y_test'].values, (-1,1))

	def get_full_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_full'][self.data_container['x_full']['hospital_id'] == hospital_id].values
		y_dummy = self.data_container['y_full'][self.data_container['x_full']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))

	def get_train_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_train'][self.data_container['x_train']['hospital_id'] == hospital_id].values
		y_dummy = self.data_container['y_train'][self.data_container['x_train']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))

	def get_test_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_test'][self.data_container['x_test']['hospital_id'] == hospital_id].values
		y_dummy = self.data_container['y_test'][self.data_container['x_test']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))

	def get_val_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_val'][self.data_container['x_val']['hospital_id'] == hospital_id].values
		y_dummy = self.data_container['y_val'][self.data_container['x_val']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))


