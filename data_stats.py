import math, re, os
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

eICU_path = '../medical_data/eicu/physionet.org/files/eicu-crd/2.0/'
result_path = '../data_stats/'

class eICU_DataLoader():

	def __init__(self, read_path, write_path, num_patients=-1):

		self.read_path = read_path
		self.write_path = write_path

		self.num_patients = num_patients

		patient_data_filename = 'eICU_patient_table'
		try:
			self.dataframe_patients = pd.read_csv(self.write_path + patient_data_filename + '.csv')
		except FileNotFoundError:
			self.dataframe_patients = self.get_processed_dataframe(filename=patient_data_filename)
		
		self.dataframe_hospitals = self.add_hospital_stats()

	def get_processed_dataframe(self, filename='eICU_patient_table'):

		try: 
			os.makedirs(self.write_path)
		except FileExistsError: pass

		remove_nans_from_codes = True

		patient_table = pd.read_csv(self.read_path + 'patient.csv')
		patient_table = patient_table.loc[patient_table['age'] != 'NaN']
		patient_table = patient_table.loc[patient_table['age'] != 'nan']
		patient_table = patient_table.loc[patient_table['gender'] != 'Unknown']

		patient_table.sort_values('hospitalid', ascending=True)

		if self.num_patients < 0:
			num_patients_to_load = len(patient_table['uniquepid'].unique())
		else:
			num_patients_to_load = self.num_patients

		patientIDs = patient_table['uniquepid'].unique()[:num_patients_to_load]


		print('looping through patient IDs in loaded tables to build a consolidated matrix...')
		pbarfreq = 10
		# pbar = tqdm(total=int(np.floor(num_patients_to_load/pbarfreq)))
		pbar = tqdm(total=num_patients_to_load + 1)

		patient_dataframe = []

		for i in range(num_patients_to_load):

			if i % pbarfreq == 0: pbar.update(pbarfreq)

			patient = patientIDs[i]
			correlated_unitstay_ids = np.asarray(patient_table['patientunitstayid'].loc[patient_table['uniquepid'] == patient].values)

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



				patient_dataframe.append(
					{
					'patient_id': patient,
					'health_system_id': current_health_sys_id,
					'corr_id': correlated_unitstay_ids[j],
					'gender': patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item(),
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
					})

		pbar.close()
		print('\n')

		pd.DataFrame(patient_dataframe).to_csv(self.write_path + filename + '.csv')
		
		return pd.DataFrame(patient_dataframe)
		
	def add_hospital_stats(self, filename='eICU_hospital_table'):

		clinic_stats_df = []

		print('\nbuilding hospital stats dataframe...')
		pbar = tqdm(total=len(self.dataframe_patients['hospital_id'].unique())+1)

		for clinic_id in self.dataframe_patients['hospital_id'].unique():

			clinic_stats_df.append({
				'hospital_id': clinic_id,
				'num_patients': len(self.dataframe_patients[self.dataframe_patients['hospital_id'] == clinic_id]),
				'age_mean': self.dataframe_patients['age'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'age_var': self.dataframe_patients['age'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_die_mean': self.dataframe_patients['will_die'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_die_var': self.dataframe_patients['will_die'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_readmit_mean': self.dataframe_patients['will_readmit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_readmit_var': self.dataframe_patients['will_readmit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_return_mean': self.dataframe_patients['will_return'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_return_var': self.dataframe_patients['will_return'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_stay_long_mean': self.dataframe_patients['will_stay_long'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_stay_long_var': self.dataframe_patients['will_stay_long'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'survive_current_icu_mean': self.dataframe_patients['survive_current_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'survive_current_icu_var': self.dataframe_patients['survive_current_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'unit_readmission_mean': self.dataframe_patients['unit_readmission'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'unit_readmission_var': self.dataframe_patients['unit_readmission'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'length_of_stay_mean': self.dataframe_patients['length_of_stay'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'length_of_stay_var': self.dataframe_patients['length_of_stay'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'length_of_icu_mean': self.dataframe_patients['length_of_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'length_of_icu_var': self.dataframe_patients['length_of_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				})


			pbar.update(1)
		pbar.close()
		print('\n')

		clinic_stats_df.append({
				'hospital_id': 0,
				'num_patients': len(self.dataframe_patients),
				'age_mean': self.dataframe_patients['age'].mean(),
				'age_var': self.dataframe_patients['age'].var(),
				'will_die_mean': self.dataframe_patients['will_die'].mean(),
				'will_die_var': self.dataframe_patients['will_die'].var(),
				'will_readmit_mean': self.dataframe_patients['will_readmit'].mean(),
				'will_readmit_var': self.dataframe_patients['will_readmit'].var(),
				'will_return_mean': self.dataframe_patients['will_return'].mean(),
				'will_return_var': self.dataframe_patients['will_return'].var(),
				'will_stay_long_mean': self.dataframe_patients['will_stay_long'].mean(),
				'will_stay_long_var': self.dataframe_patients['will_stay_long'].var(),
				'survive_current_icu_mean': self.dataframe_patients['survive_current_icu'].mean(),
				'survive_current_icu_var': self.dataframe_patients['survive_current_icu'].var(),
				'unit_readmission_mean': self.dataframe_patients['unit_readmission'].mean(),
				'unit_readmission_var': self.dataframe_patients['unit_readmission'].var(),
				'length_of_stay_mean': self.dataframe_patients['length_of_stay'].mean(),
				'length_of_stay_var': self.dataframe_patients['length_of_stay'].var(),
				'length_of_icu_mean': self.dataframe_patients['length_of_icu'].mean(),
				'length_of_icu_var': self.dataframe_patients['length_of_icu'].var(),
				})


		clinic_stats_df = pd.DataFrame(clinic_stats_df)
		clinic_stats_df.to_csv(self.write_path + filename + '.csv')
		# clinic_stats_df.reset_index(drop=True)

		return clinic_stats_df
	
eICUdata = eICU_DataLoader(eICU_path, result_path, num_patients=-1)

print('\n\nfetched data patients:\n', eICU.dataframe_patients, '\n\n******************\n')
print('\n\nfetched data hospitals:\n', eICU.dataframe_hospitals, '\n\n******************\n')