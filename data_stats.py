import math, re, os
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 16})

eICU_path = '../medical_data/eicu/physionet.org/files/eicu-crd/2.0/'
result_path = '../data_stats_29_9/'
# result_path = '../data_stats_22_9_2/'

class eICU_DataLoader():

	def __init__(self, read_path, write_path, num_patients=-1):

		self.read_path = read_path
		self.write_path = write_path

		self.annotation_size = 8

		try: 
			os.makedirs(self.write_path)
		except FileExistsError: pass

		self.num_patients = num_patients

		patient_data_filename = 'eICU_patient_table'
		hospital_data_filename = 'eICU_hospital_table'

		try:
			self.dataframe_patients = pd.read_csv(self.write_path + patient_data_filename + '.csv')
		except FileNotFoundError:
			self.dataframe_patients = self.get_processed_dataframe(filename=patient_data_filename)
		
		try:
			self.dataframe_hospitals = pd.read_csv(self.write_path + hospital_data_filename + '.csv')
		except FileNotFoundError:
			self.dataframe_hospitals = self.get_hospital_stats(filename=hospital_data_filename)

		self.get_hospital_plots()

		self.patient_visit_df = self.get_patient_visit_plot()
		
		self.get_patient_plot()

	def get_processed_dataframe(self, filename='eICU_patient_table'):

		remove_nans_from_codes = True

		patient_table = pd.read_csv(self.read_path + 'patient.csv')
		patient_table = patient_table.loc[patient_table['age'] != 'NaN']
		patient_table = patient_table.loc[patient_table['age'] != 'nan']
		patient_table = patient_table.loc[patient_table['gender'] != 'Unknown']
		apache_table = pd.read_csv(self.read_path + 'apacheApsVar.csv')



		def get_aps_values(str_label, corrunitstayid):
			
			dummy = apache_table[str_label].loc[apache_table['patientunitstayid'] == corrunitstayid].values
			dummy = np.asarray(dummy)
			dummy = np.ravel(dummy)
			if dummy.shape[0] > 0:
				dummy = dummy[0]
			try:
				dummy = np.float(dummy)
			except TypeError:
				# print('Hello!')
				# print(dummy)
				dummy = np.NaN
			return dummy



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


				weight_dummy = np.float(patient_table['admissionweight'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item())
				height_dummy = np.float(patient_table['admissionheight'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()) / 100.
				bmi_dummy = weight_dummy / ((height_dummy * height_dummy) + 1e-6)
				if bmi_dummy > 200:
					bmi_dummy = 0.

				if patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item() == 'Female':
					gender_dummy = 0.
				else:
					gender_dummy = 1.


				patient_dataframe.append(
					{
					'patient_id': patient,
					'health_system_id': current_health_sys_id,
					'corr_id': correlated_unitstay_ids[j],
					'gender': gender_dummy,
					'age': age_dummy,
					'ethnicity': ethnicity_dummy,
					'weight': weight_dummy,
					'height': height_dummy,
					'bmi': bmi_dummy,
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

		pd.DataFrame(patient_dataframe).to_csv(self.write_path + filename + '.csv')
		
		return pd.DataFrame(patient_dataframe)
		
	def get_hospital_stats(self, filename='eICU_hospital_table'):

		clinic_stats_df = []

		print('\nbuilding hospital stats dataframe...')
		pbar = tqdm(total=len(self.dataframe_patients['hospital_id'].unique())+1)

		for clinic_id in self.dataframe_patients['hospital_id'].unique():

			num_patients_dummy = len(self.dataframe_patients[self.dataframe_patients['hospital_id'] == clinic_id])

			ethnicity_dummy_df = pd.get_dummies(
				self.dataframe_patients['ethnicity'][self.dataframe_patients['hospital_id'] == clinic_id],
				dummy_na=False,
				)
			ethnicity_dummy_df = pd.DataFrame(ethnicity_dummy_df)


			if 'Unknown' in ethnicity_dummy_df.keys():
				if 'Other/Unknown' in ethnicity_dummy_df.keys():
					ethnicity_dummy_df['Unknown'] += ethnicity_dummy_df['Other/Unknown']
					ethnicity_dummy_df.drop(columns='Other/Unknown')
			else:
				ethnicity_dummy_df['Unknown'] = 0

			if 'African American' not in ethnicity_dummy_df.keys():
				ethnicity_dummy_df['African American'] = 0
			if 'Caucasian' not in ethnicity_dummy_df.keys():
				ethnicity_dummy_df['Caucasian'] = 0
			if 'Hispanic' not in ethnicity_dummy_df.keys():
				ethnicity_dummy_df['Hispanic'] = 0
			if 'Native American' not in ethnicity_dummy_df.keys():
				ethnicity_dummy_df['Native American'] = 0
			if 'Asian' not in ethnicity_dummy_df.keys():
				ethnicity_dummy_df['Asian'] = 0

			print('\n\n****************************\n\n')
			print(pd.DataFrame(ethnicity_dummy_df))
			print('\n\n****************************\n\n')

			clinic_stats_df.append({
				'hospital_id': clinic_id,
				'num_patients': num_patients_dummy,
				'gender_mean': self.dataframe_patients['gender'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'ethnicity_caucasian_mean': ethnicity_dummy_df['Caucasian'].mean(),
				'ethnicity_africanamerican_mean': ethnicity_dummy_df['African American'].mean(),
				'ethnicity_nativeamerican_mean': ethnicity_dummy_df['Native American'].mean(),
				'ethnicity_asian_mean': ethnicity_dummy_df['Asian'].mean(),
				'ethnicity_hispanic_mean': ethnicity_dummy_df['Hispanic'].mean(),
				'ethnicity_unknown_mean': ethnicity_dummy_df['Unknown'].mean(),
				'age_mean': self.dataframe_patients['age'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'age_var': self.dataframe_patients['age'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'age_std': self.dataframe_patients['age'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'bmi_mean': self.dataframe_patients['bmi'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'bmi_var': self.dataframe_patients['bmi'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'bmi_std': self.dataframe_patients['bmi'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'weight_mean': self.dataframe_patients['weight'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'weight_var': self.dataframe_patients['weight'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'weight_std': self.dataframe_patients['weight'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'height_mean': self.dataframe_patients['height'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'height_var': self.dataframe_patients['height'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).var(),
				'height_std': self.dataframe_patients['height'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'will_die_mean': self.dataframe_patients['will_die'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_die_var': self.dataframe_patients['will_die'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_die_std': self.dataframe_patients['will_die'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'will_readmit_mean': self.dataframe_patients['will_readmit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_readmit_var': self.dataframe_patients['will_readmit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_readmit_std': self.dataframe_patients['will_readmit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'will_return_mean': self.dataframe_patients['will_return'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_return_var': self.dataframe_patients['will_return'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_return_std': self.dataframe_patients['will_return'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'will_stay_long_mean': self.dataframe_patients['will_stay_long'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'will_stay_long_var': self.dataframe_patients['will_stay_long'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'will_stay_long_std': self.dataframe_patients['will_stay_long'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'survive_current_icu_mean': self.dataframe_patients['survive_current_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'survive_current_icu_var': self.dataframe_patients['survive_current_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'survive_current_icu_std': self.dataframe_patients['survive_current_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'unit_readmission_mean': self.dataframe_patients['unit_readmission'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'unit_readmission_var': self.dataframe_patients['unit_readmission'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'unit_readmission_std': self.dataframe_patients['unit_readmission'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'length_of_stay_mean': self.dataframe_patients['length_of_stay'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'length_of_stay_var': self.dataframe_patients['length_of_stay'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'length_of_stay_std': self.dataframe_patients['length_of_stay'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'length_of_icu_mean': self.dataframe_patients['length_of_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].mean(),
				'length_of_icu_var': self.dataframe_patients['length_of_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].var(),
				'length_of_icu_std': self.dataframe_patients['length_of_icu'].loc[self.dataframe_patients['hospital_id'] == clinic_id].std(),
				'aps_intubated_mean': self.dataframe_patients['aps_intubated'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_vent_mean': self.dataframe_patients['aps_vent'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_dialysis_mean': self.dataframe_patients['aps_dialysis'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_eyes_mean': self.dataframe_patients['aps_eyes'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_motor_mean': self.dataframe_patients['aps_motor'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_verbal_mean': self.dataframe_patients['aps_verbal'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_meds_mean': self.dataframe_patients['aps_meds'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_urine_mean': self.dataframe_patients['aps_urine'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_wbc_mean': self.dataframe_patients['aps_wbc'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_temperature_mean': self.dataframe_patients['aps_temperature'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'aps_respiratoryRate_mean': self.dataframe_patients['aps_respiratoryRate'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).mean(),
				'aps_sodium_mean': self.dataframe_patients['aps_sodium'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_heartrate_mean': self.dataframe_patients['aps_heartrate'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_meanBp_mean': self.dataframe_patients['aps_meanBp'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_ph_mean': self.dataframe_patients['aps_ph'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_hematocrit_mean': self.dataframe_patients['aps_hematocrit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_creatinine_mean': self.dataframe_patients['aps_creatinine'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_albumin_mean': self.dataframe_patients['aps_albumin'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_pao2_mean': self.dataframe_patients['aps_pao2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_pco2_mean': self.dataframe_patients['aps_pco2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_bun_mean': self.dataframe_patients['aps_bun'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_glucose_mean': self.dataframe_patients['aps_glucose'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_bilirubin_mean': self.dataframe_patients['aps_bilirubin'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),
				'aps_fio2_mean': self.dataframe_patients['aps_fio2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).mean(),				
				'aps_intubated_std': self.dataframe_patients['aps_intubated'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_vent_std': self.dataframe_patients['aps_vent'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_dialysis_std': self.dataframe_patients['aps_dialysis'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_eyes_std': self.dataframe_patients['aps_eyes'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_motor_std': self.dataframe_patients['aps_motor'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_verbal_std': self.dataframe_patients['aps_verbal'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_meds_std': self.dataframe_patients['aps_meds'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_urine_std': self.dataframe_patients['aps_urine'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_wbc_std': self.dataframe_patients['aps_wbc'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_temperature_std': self.dataframe_patients['aps_temperature'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'aps_respiratoryRate_std': self.dataframe_patients['aps_respiratoryRate'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(0., np.NaN).std(),
				'aps_sodium_std': self.dataframe_patients['aps_sodium'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_heartrate_std': self.dataframe_patients['aps_heartrate'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_meanBp_std': self.dataframe_patients['aps_meanBp'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_ph_mean': self.dataframe_patients['aps_ph'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_hematocrit_std': self.dataframe_patients['aps_hematocrit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_creatinine_std': self.dataframe_patients['aps_creatinine'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_albumin_std': self.dataframe_patients['aps_albumin'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_pao2_std': self.dataframe_patients['aps_pao2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_pco2_std': self.dataframe_patients['aps_pco2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_bun_std': self.dataframe_patients['aps_bun'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_glucose_std': self.dataframe_patients['aps_glucose'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_bilirubin_std': self.dataframe_patients['aps_bilirubin'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_fio2_std': self.dataframe_patients['aps_fio2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).std(),
				'aps_intubated_missing': self.dataframe_patients['aps_intubated'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_vent_missing': self.dataframe_patients['aps_vent'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_dialysis_missing': self.dataframe_patients['aps_dialysis'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_eyes_missing': self.dataframe_patients['aps_eyes'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_motor_missing': self.dataframe_patients['aps_motor'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_verbal_missing': self.dataframe_patients['aps_verbal'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_meds_missing': self.dataframe_patients['aps_meds'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_urine_missing': self.dataframe_patients['aps_urine'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_wbc_missing': self.dataframe_patients['aps_wbc'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_temperature_missing': self.dataframe_patients['aps_temperature'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_respiratoryRate_missing': self.dataframe_patients['aps_respiratoryRate'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_sodium_missing': self.dataframe_patients['aps_sodium'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_heartrate_missing': self.dataframe_patients['aps_heartrate'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_meanBp_missing': self.dataframe_patients['aps_meanBp'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_ph_mean': self.dataframe_patients['aps_ph'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_hematocrit_missing': self.dataframe_patients['aps_hematocrit'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_creatinine_missing': self.dataframe_patients['aps_creatinine'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_albumin_missing': self.dataframe_patients['aps_albumin'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_pao2_missing': self.dataframe_patients['aps_pao2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_pco2_missing': self.dataframe_patients['aps_pco2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_bun_missing': self.dataframe_patients['aps_bun'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_glucose_missing': self.dataframe_patients['aps_glucose'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_bilirubin_missing': self.dataframe_patients['aps_bilirubin'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				'aps_fio2_missing': self.dataframe_patients['aps_fio2'].loc[self.dataframe_patients['hospital_id'] == clinic_id].replace(-1., np.NaN).isna().sum()/num_patients_dummy,
				})



			pbar.update(1)
		pbar.close()
		print('\n')



		ethnicity_dummy_df = pd.get_dummies(
			self.dataframe_patients['ethnicity'],
			dummy_na=False,
			)
		ethnicity_dummy_df = pd.DataFrame(ethnicity_dummy_df)


		if 'Unknown' in ethnicity_dummy_df.keys():
			if 'Other/Unknown' in ethnicity_dummy_df.keys():
				ethnicity_dummy_df['Unknown'] += ethnicity_dummy_df['Other/Unknown']
				ethnicity_dummy_df.drop(columns='Other/Unknown')
		else:
			ethnicity_dummy_df['Unknown'] = 0

		if 'African American' not in ethnicity_dummy_df.keys():
			ethnicity_dummy_df['African American'] = 0
		if 'Caucasian' not in ethnicity_dummy_df.keys():
			ethnicity_dummy_df['Caucasian'] = 0
		if 'Hispanic' not in ethnicity_dummy_df.keys():
			ethnicity_dummy_df['Hispanic'] = 0
		if 'Native American' not in ethnicity_dummy_df.keys():
			ethnicity_dummy_df['Native American'] = 0
		if 'Asian' not in ethnicity_dummy_df.keys():
			ethnicity_dummy_df['Asian'] = 0


		num_patients_total = len(self.dataframe_patients)
		clinic_stats_df_all = clinic_stats_df
		clinic_stats_df_all.append({
				'hospital_id': 0,
				'num_patients': num_patients_total,
				'gender_mean': self.dataframe_patients['gender'].mean(),
				'ethnicity_caucasian_mean': ethnicity_dummy_df['Caucasian'].mean(),
				'ethnicity_africanamerican_mean': ethnicity_dummy_df['African American'].mean(),
				'ethnicity_nativeamerican_mean': ethnicity_dummy_df['Native American'].mean(),
				'ethnicity_asian_mean': ethnicity_dummy_df['Asian'].mean(),
				'ethnicity_hispanic_mean': ethnicity_dummy_df['Hispanic'].mean(),
				'ethnicity_unknown_mean': ethnicity_dummy_df['Unknown'].mean(),
				'age_mean': self.dataframe_patients['age'].replace(0., np.NaN).mean(),
				'age_var': self.dataframe_patients['age'].replace(0., np.NaN).var(),
				'age_std': self.dataframe_patients['age'].replace(0., np.NaN).std(),
				'bmi_mean': self.dataframe_patients['bmi'].replace(0., np.NaN).mean(),
				'bmi_var': self.dataframe_patients['bmi'].replace(0., np.NaN).var(),
				'bmi_std': self.dataframe_patients['bmi'].replace(0., np.NaN).std(),
				'weight_mean': self.dataframe_patients['weight'].replace(0., np.NaN).mean(),
				'weight_var': self.dataframe_patients['weight'].replace(0., np.NaN).var(),
				'weight_std': self.dataframe_patients['weight'].replace(0., np.NaN).std(),
				'height_mean': self.dataframe_patients['height'].replace(0., np.NaN).mean(),
				'height_var': self.dataframe_patients['height'].replace(0., np.NaN).var(),
				'height_std': self.dataframe_patients['height'].replace(0., np.NaN).std(),
				'will_die_mean': self.dataframe_patients['will_die'].mean(),
				'will_die_var': self.dataframe_patients['will_die'].var(),
				'will_die_std': self.dataframe_patients['will_die'].std(),
				'will_readmit_mean': self.dataframe_patients['will_readmit'].mean(),
				'will_readmit_var': self.dataframe_patients['will_readmit'].var(),
				'will_readmit_std': self.dataframe_patients['will_readmit'].std(),
				'will_return_mean': self.dataframe_patients['will_return'].mean(),
				'will_return_var': self.dataframe_patients['will_return'].var(),
				'will_return_std': self.dataframe_patients['will_return'].std(),
				'will_stay_long_mean': self.dataframe_patients['will_stay_long'].mean(),
				'will_stay_long_var': self.dataframe_patients['will_stay_long'].var(),
				'will_stay_long_std': self.dataframe_patients['will_stay_long'].std(),
				'survive_current_icu_mean': self.dataframe_patients['survive_current_icu'].mean(),
				'survive_current_icu_var': self.dataframe_patients['survive_current_icu'].var(),
				'survive_current_icu_std': self.dataframe_patients['survive_current_icu'].std(),
				'unit_readmission_mean': self.dataframe_patients['unit_readmission'].mean(),
				'unit_readmission_var': self.dataframe_patients['unit_readmission'].var(),
				'unit_readmission_std': self.dataframe_patients['unit_readmission'].std(),
				'length_of_stay_mean': self.dataframe_patients['length_of_stay'].mean(),
				'length_of_stay_var': self.dataframe_patients['length_of_stay'].var(),
				'length_of_stay_std': self.dataframe_patients['length_of_stay'].std(),
				'length_of_icu_mean': self.dataframe_patients['length_of_icu'].mean(),
				'length_of_icu_var': self.dataframe_patients['length_of_icu'].var(),
				'length_of_icu_std': self.dataframe_patients['length_of_icu'].std(),
				'aps_intubated_mean': self.dataframe_patients['aps_intubated'].replace(-1., np.NaN).mean(),
				'aps_vent_mean': self.dataframe_patients['aps_vent'].replace(-1., np.NaN).mean(),
				'aps_dialysis_mean': self.dataframe_patients['aps_dialysis'].replace(-1., np.NaN).mean(),
				'aps_eyes_mean': self.dataframe_patients['aps_eyes'].replace(-1., np.NaN).mean(),
				'aps_motor_mean': self.dataframe_patients['aps_motor'].replace(-1., np.NaN).mean(),
				'aps_verbal_mean': self.dataframe_patients['aps_verbal'].replace(-1., np.NaN).mean(),
				'aps_meds_mean': self.dataframe_patients['aps_meds'].replace(-1., np.NaN).mean(),
				'aps_urine_mean': self.dataframe_patients['aps_urine'].replace(-1., np.NaN).mean(),
				'aps_wbc_mean': self.dataframe_patients['aps_wbc'].replace(-1., np.NaN).mean(),
				'aps_temperature_mean': self.dataframe_patients['aps_temperature'].replace(-1., np.NaN).mean(),
				'aps_respiratoryRate_mean': self.dataframe_patients['aps_respiratoryRate'].replace(-1., np.NaN).mean(),
				'aps_sodium_mean': self.dataframe_patients['aps_sodium'].replace(-1., np.NaN).mean(),
				'aps_heartrate_mean': self.dataframe_patients['aps_heartrate'].replace(-1., np.NaN).mean(),
				'aps_meanBp_mean': self.dataframe_patients['aps_meanBp'].replace(-1., np.NaN).mean(),
				'aps_ph_mean': self.dataframe_patients['aps_ph'].replace(-1., np.NaN).mean(),
				'aps_hematocrit_mean': self.dataframe_patients['aps_hematocrit'].replace(-1., np.NaN).mean(),
				'aps_creatinine_mean': self.dataframe_patients['aps_creatinine'].replace(-1., np.NaN).mean(),
				'aps_albumin_mean': self.dataframe_patients['aps_albumin'].replace(-1., np.NaN).mean(),
				'aps_pao2_mean': self.dataframe_patients['aps_pao2'].replace(-1., np.NaN).mean(),
				'aps_pco2_mean': self.dataframe_patients['aps_pco2'].replace(-1., np.NaN).mean(),
				'aps_bun_mean': self.dataframe_patients['aps_bun'].replace(-1., np.NaN).mean(),
				'aps_glucose_mean': self.dataframe_patients['aps_glucose'].replace(-1., np.NaN).mean(),
				'aps_bilirubin_mean': self.dataframe_patients['aps_bilirubin'].replace(-1., np.NaN).mean(),
				'aps_fio2_mean': self.dataframe_patients['aps_fio2'].replace(-1., np.NaN).mean(),
				'aps_intubated_std': self.dataframe_patients['aps_intubated'].replace(-1., np.NaN).std(),
				'aps_vent_std': self.dataframe_patients['aps_vent'].replace(-1., np.NaN).std(),
				'aps_dialysis_std': self.dataframe_patients['aps_dialysis'].replace(-1., np.NaN).std(),
				'aps_eyes_std': self.dataframe_patients['aps_eyes'].replace(-1., np.NaN).std(),
				'aps_motor_std': self.dataframe_patients['aps_motor'].replace(-1., np.NaN).std(),
				'aps_verbal_std': self.dataframe_patients['aps_verbal'].replace(-1., np.NaN).std(),
				'aps_meds_std': self.dataframe_patients['aps_meds'].replace(-1., np.NaN).std(),
				'aps_urine_std': self.dataframe_patients['aps_urine'].replace(-1., np.NaN).std(),
				'aps_wbc_std': self.dataframe_patients['aps_wbc'].replace(-1., np.NaN).std(),
				'aps_temperature_std': self.dataframe_patients['aps_temperature'].replace(0., np.NaN).std(),
				'aps_respiratoryRate_std': self.dataframe_patients['aps_respiratoryRate'].replace(0., np.NaN).std(),
				'aps_sodium_std': self.dataframe_patients['aps_sodium'].replace(-1., np.NaN).std(),
				'aps_heartrate_std': self.dataframe_patients['aps_heartrate'].replace(-1., np.NaN).std(),
				'aps_meanBp_std': self.dataframe_patients['aps_meanBp'].replace(-1., np.NaN).std(),
				'aps_ph_std': self.dataframe_patients['aps_ph'].replace(-1., np.NaN).std(),
				'aps_hematocrit_std': self.dataframe_patients['aps_hematocrit'].replace(-1., np.NaN).std(),
				'aps_creatinine_std': self.dataframe_patients['aps_creatinine'].replace(-1., np.NaN).std(),
				'aps_albumin_std': self.dataframe_patients['aps_albumin'].replace(-1., np.NaN).std(),
				'aps_pao2_std': self.dataframe_patients['aps_pao2'].replace(-1., np.NaN).std(),
				'aps_pco2_std': self.dataframe_patients['aps_pco2'].replace(-1., np.NaN).std(),
				'aps_bun_std': self.dataframe_patients['aps_bun'].replace(-1., np.NaN).std(),
				'aps_glucose_std': self.dataframe_patients['aps_glucose'].replace(-1., np.NaN).std(),
				'aps_bilirubin_std': self.dataframe_patients['aps_bilirubin'].replace(-1., np.NaN).std(),
				'aps_fio2_std': self.dataframe_patients['aps_fio2'].replace(-1., np.NaN).std(),
				'aps_intubated_missing': self.dataframe_patients['aps_intubated'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_vent_missing': self.dataframe_patients['aps_vent'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_dialysis_missing': self.dataframe_patients['aps_dialysis'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_eyes_missing': self.dataframe_patients['aps_eyes'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_motor_missing': self.dataframe_patients['aps_motor'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_verbal_missing': self.dataframe_patients['aps_verbal'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_meds_missing': self.dataframe_patients['aps_meds'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_urine_missing': self.dataframe_patients['aps_urine'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_wbc_missing': self.dataframe_patients['aps_wbc'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_temperature_missing': self.dataframe_patients['aps_temperature'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_respiratoryRate_missing': self.dataframe_patients['aps_respiratoryRate'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_sodium_missing': self.dataframe_patients['aps_sodium'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_heartrate_missing': self.dataframe_patients['aps_heartrate'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_meanBp_missing': self.dataframe_patients['aps_meanBp'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_ph_missing': self.dataframe_patients['aps_ph'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_hematocrit_missing': self.dataframe_patients['aps_hematocrit'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_creatinine_missing': self.dataframe_patients['aps_creatinine'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_albumin_missing': self.dataframe_patients['aps_albumin'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_pao2_missing': self.dataframe_patients['aps_pao2'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_pco2_missing': self.dataframe_patients['aps_pco2'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_bun_missing': self.dataframe_patients['aps_bun'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_glucose_missing': self.dataframe_patients['aps_glucose'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_bilirubin_missing': self.dataframe_patients['aps_bilirubin'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				'aps_fio2_missing': self.dataframe_patients['aps_fio2'].replace(-1., np.NaN).isna().sum()/num_patients_total,
				})

		clinic_stats_df = pd.DataFrame(clinic_stats_df)
		clinic_stats_df.to_csv(self.write_path + filename + '.csv')

		dummyframe = clinic_stats_df.mean()
		dummyframe.to_csv(self.write_path + filename + '_mean_ofhospitalmeans.csv')
		dummyframe = clinic_stats_df.var()
		dummyframe.to_csv(self.write_path + filename + '_vars_ofhospitalmeans.csv')
		dummyframe = clinic_stats_df.std()
		dummyframe.to_csv(self.write_path + filename + '_stds_ofhospitalmeans.csv')

		clinic_stats_df_all = pd.DataFrame(clinic_stats_df_all)
		clinic_stats_df_all[clinic_stats_df_all['hospital_id'] == 0].to_csv(self.write_path + filename + '_statsoverallcases.csv')

		return clinic_stats_df
	
	def get_patient_visit_plot(self):


		try:
			patient_visit_df = pd.read_csv(self.write_path + 'patient_visit_df.csv.csv')
	
		except FileNotFoundError:

			patient_visit_df = []

			for health_sys_id in self.dataframe_patients['health_system_id'].unique():

				dummydf = self.dataframe_patients[self.dataframe_patients['health_system_id'] == health_sys_id]
				max_visits = dummydf['visits_current_stay'].max()

				dummydf = dummydf[dummydf['visits_current_stay'] == max_visits]

				patient_visit_df.append({
					'patient_id': dummydf['patient_id'].iloc[0],
					'health_sys_id': health_sys_id,
					'max_visits': max_visits,
					'hospital_id': dummydf['hospital_id'].iloc[0],
					})

			patient_visit_df = pd.DataFrame(patient_visit_df)
			patient_visit_df.to_csv(self.write_path + 'patient_visit_df.csv')


		plt.figure(figsize=(6,6))

		for hospital_id in list(patient_visit_df['hospital_id'].unique().values):

			dummydf = patient_visit_df[patient_visit_df['hospital_id'] == hospital_id]

			plt.scatter(dummydf['max_visits'].mean(), dummydf['max_visits'].std(), c='darkgreen', s=100, alpha=.2)

		plt.grid()
		plt.xlabel('Mean number of visits')
		plt.ylabel('Standard deviation of visits')
		# plt.title(x_label + ' vs. ' + y_label)

		plt.savefig(self.write_path + 'visits.pdf')
		plt.close()

		return patient_visit_df

	def get_hospital_plots(self):

		scatter_plot_path = self.write_path + 'scatter_plots/'
		scatter2d_plot_path = self.write_path + 'scatter2d_plots/'
		mean_var_path = self.write_path + 'mean_var/'
		mean_std_path = self.write_path + 'mean_std/'
		histogram_path = self.write_path + 'histograms/'

		# hospital_id, num_patients, age_mean, age_var, will_die_mean, will_die_var, will_readmit_mean, will_readmit_var, will_return_mean, will_return_var, will_stay_long_mean, will_stay_long_var, survive_current_icu_mean, survive_current_icu_var, unit_readmission_mean, unit_readmission_var, length_of_stay_mean, length_of_stay_var, length_of_icu_mean, length_of_icu_var
		try: 
			os.makedirs(scatter_plot_path)
			os.makedirs(scatter2d_plot_path)
			os.makedirs(mean_var_path)
			os.makedirs(mean_std_path)
			os.makedirs(histogram_path)
		except FileExistsError: pass


		def label_transform(label_code):

			if label_code == 'num_patients':
				return 'Number of patients'
			elif label_code == 'length_of_icu':
				return 'Length of ICU [hours]'
			elif label_code == 'length_of_stay':
				return 'Length of stay [hours]'
			elif label_code == 'will_die':
				return 'Will die'
			elif label_code == 'will_return':
				return 'Will return'
			elif label_code == 'survive_current_icu':
				return 'ICU survival'
			elif label_code == 'will_readmit':
				return 'Will readmit'
			elif label_code == 'unit_readmission':
				return 'Unit readmission'
			elif label_code == 'age':
				return 'Age [Years]'
			elif label_code == 'bmi':
				return 'BMI [kg/m^2]'
			elif label_code == 'weight':
				return 'Weight [kg]'
			elif label_code == 'height':
				return 'Height [m]'
			elif label_code == 'gender':
				return 'Gender (Female=0)'
			elif label_code == 'gender_mean':
				return 'Gender (Female=0)'

			elif label_code == 'will_return_mean':
				return 'Mean of "Will return"'
			elif label_code == 'will_return_var':
				return 'Variance of "Will return"'

			elif label_code == 'unit_readmission_mean':
				return 'Mean of "Unit readmission"'
			elif label_code == 'unit_readmission_var':
				return 'Variance of "Unit readmission"'

			elif label_code == 'will_readmit_mean':
				return 'Mean of "Will readmit"'
			elif label_code == 'will_readmit_var':
				return 'Variance of "Will readmit"'

			elif label_code == 'will_die_mean':
				return 'Mean of "Will die"'
			elif label_code == 'will_die_var':
				return 'Variance of "Will die"'

			elif label_code == 'survive_current_icu_mean':
				return 'ICU survival mean'
			elif label_code == 'survive_current_icu_mean':
				return 'ICU survival variance'

			else:
				return label_code


		def scatter3d(dataframe, x_label, y_label, z_label, outpath):

			dataframe = dataframe.sort_values('num_patients', ascending=False)
			dataframe = dataframe[dataframe['num_patients'] > 10]

			plt.figure()

			plt.scatter(dataframe[x_label], dataframe[y_label], c=dataframe[z_label], s=32, alpha=.2, cmap='jet')

			for i in range(len(dataframe)):
				plt.annotate(dataframe['hospital_id'].iloc[i], (dataframe[x_label].iloc[i], dataframe[y_label].iloc[i]), size=self.annotation_size)

			# plt.legend()
			plt.colorbar()
			plt.grid()
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.title('colorbar: ' + z_label)

			plt.savefig(outpath + x_label + '_' + y_label + '_' + z_label + '.pdf')
			plt.close()
 		
		def scatter2d(dataframe, x_label, y_label, outpath, log=False):

			dataframe = dataframe.sort_values('num_patients', ascending=False)
			dataframe = dataframe[dataframe['num_patients'] > 10]

			plt.figure(figsize=(6,6))

			plt.scatter(dataframe[x_label], dataframe[y_label], c='darkgreen', s=100, alpha=.2)

			# for i in range(len(dataframe)):
			# 	plt.annotate(dataframe['hospital_id'].iloc[i], (dataframe[x_label].iloc[i], dataframe[y_label].iloc[i]), size=self.annotation_size)

			# plt.legend()
			# plt.colorbar()
			if y_label == 'num_patients':
				plt.yscale('log')
			if log:
				plt.xscale('log')
				plt.yscale('log') 
			plt.grid()
			plt.xlabel(label_transform(x_label))
			plt.ylabel(label_transform(y_label))
			# plt.title(x_label + ' vs. ' + y_label)

			if log:
				plt.savefig(outpath + x_label + '_' + y_label + '_log.pdf')
			else:
				plt.savefig(outpath + x_label + '_' + y_label + '.pdf')
			plt.close()
 		
		def mean_var_scatter(dataframe, label, outpath):

			dataframe = dataframe.sort_values('num_patients', ascending=False)
			dataframe = dataframe[dataframe['num_patients'] > 10]

			plt.figure()

			plt.scatter(dataframe[label+'_mean'], dataframe[label+'_var'], c='darkgreen', s=32, alpha=.3)

			for i in range(len(dataframe)):
				plt.annotate(dataframe['hospital_id'].iloc[i], (dataframe[label+'_mean'].iloc[i], dataframe[label+'_var'].iloc[i]), size=self.annotation_size)

			# plt.legend()
			plt.grid()
			plt.xlabel('Mean')
			plt.ylabel('Variance')
			plt.title(label)

			plt.savefig(outpath + label + '.pdf')
			plt.close()
 		
		def mean_std_scatter(dataframe, label, outpath, log=False):

			dataframe = dataframe.sort_values('num_patients', ascending=False)
			dataframe = dataframe[dataframe['num_patients'] > 10]

			plt.figure(figsize=(6,6))

			plt.scatter(dataframe[label+'_mean'], dataframe[label+'_std'], c='darkgreen', s=100, alpha=.3)

			# for i in range(len(dataframe)):
			# 	plt.annotate(dataframe['hospital_id'].iloc[i], (dataframe[label+'_mean'].iloc[i], dataframe[label+'_std'].iloc[i]), size=self.annotation_size)

			# plt.legend()
			if log:
				plt.xscale('log')
				plt.yscale('log')
			plt.grid()
			plt.xlabel('Mean')
			plt.ylabel('Standard Deviation')
			plt.title(label_transform(label))
			plt.tight_layout()

			if log:
				plt.savefig(outpath + label + '_log.pdf')
			else:
				plt.savefig(outpath + label + '.pdf')
			plt.close()
 		
		def histogram_plot(dataframe, x_label, outpath):

			plt.figure()

			plt.title(x_label)

			plt.hist(dataframe[x_label])

			plt.grid()
			plt.savefig(outpath + x_label +'.pdf')
			plt.close()

		mean_var_scatter(self.dataframe_hospitals, 'age', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'bmi', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'weight', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'height', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'length_of_icu', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'length_of_stay', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'will_stay_long', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'will_return', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'will_readmit', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'unit_readmission', mean_var_path)
		mean_var_scatter(self.dataframe_hospitals, 'survive_current_icu', mean_var_path)

		mean_std_scatter(self.dataframe_hospitals, 'age', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'bmi', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'bmi', mean_std_path, log=True)
		mean_std_scatter(self.dataframe_hospitals, 'weight', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'height', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'length_of_icu', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'length_of_stay', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'will_stay_long', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'will_return', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'will_readmit', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'unit_readmission', mean_std_path)
		mean_std_scatter(self.dataframe_hospitals, 'survive_current_icu', mean_std_path)

		scatter2d(self.dataframe_hospitals, 'will_readmit_mean', 'unit_readmission_mean', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'will_readmit_var', 'unit_readmission_var', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'will_readmit_mean', 'will_return_mean', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'will_readmit_var', 'will_return_var', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'unit_readmission_mean', 'will_return_mean', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'unit_readmission_var', 'will_return_var', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'gender_mean', 'will_die_mean', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'gender_mean', 'will_die_var', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'gender_mean', 'num_patients', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'gender_mean', 'unit_readmission_mean', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'gender_mean', 'survive_current_icu_mean', scatter2d_plot_path)
		scatter2d(self.dataframe_hospitals, 'num_patients', 'bmi_mean', scatter2d_plot_path, log=True)

		# scatter3d(self.dataframe_hospitals, 'age_mean', 'age_var', 'num_patients', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'age_mean', 'num_patients', 'will_die_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'age_mean', 'will_die_mean', 'will_readmit_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'age_mean', 'length_of_icu_mean', 'will_die_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'age_mean', 'will_readmit_mean', 'num_patients', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'num_patients', 'will_die_mean', 'length_of_icu_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'num_patients', 'length_of_icu_mean', 'will_readmit_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'num_patients', 'unit_readmission_mean', 'age_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'num_patients', 'length_of_stay_mean', 'age_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'num_patients', 'length_of_icu_mean', 'age_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'length_of_stay_mean', 'length_of_icu_mean', 'age_mean', scatter_plot_path)
		# scatter3d(self.dataframe_hospitals, 'length_of_icu_mean', 'length_of_icu_var', 'length_of_stay_mean', scatter_plot_path)

		# histogram_plot(self.dataframe_hospitals, 'num_patients', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'age_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'age_var', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'will_die_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'length_of_icu_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'length_of_stay_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'will_readmit_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'will_stay_long_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'survive_current_icu_mean', histogram_path)
		# histogram_plot(self.dataframe_hospitals, 'unit_readmission_mean', histogram_path)

	def get_patient_plot(self):

		try: 
			os.makedirs(self.write_path + 'patient_scatter/')
		except FileExistsError: pass

		def patient_scatter(xlabel, ylabel):

			plt.figure()
			plt.title(xlabel + ' vs ' + ylabel)

			plt.scatter(patient_mean_df[xlabel], patient_mean_df[ylabel], c='darkgreen', alpha=.4)

			plt.grid()
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)

			plt.savefig(self.write_path + 'patient_scatter/' + xlabel + '_' + ylabel + '.pdf')


		relevant_cols = [
			'length_of_stay',
			'length_of_icu',
			'will_return',
			'will_die',
			'will_readmit',
			'will_stay_long',
			'survive_current_icu',
			'unit_readmission']

		unique_patient_ids = self.dataframe_patients['patient_id'].unique()

		patient_mean_df = []

		for patientID in unique_patient_ids[:2000]:

			dummydf = self.dataframe_patients[relevant_cols][self.dataframe_patients['patient_id'] == patientID]

			patient_mean_df.append({
				'patient_id': patientID,
				'length_of_stay': dummydf['length_of_stay'].mean(),
				'length_of_icu': dummydf['length_of_icu'].mean(),
				'will_return': dummydf['will_return'].mean(),
				'will_die': dummydf['will_die'].mean(),
				'will_readmit': dummydf['will_readmit'].mean(),
				'will_stay_long': dummydf['will_stay_long'].mean(),
				'survive_current_icu': dummydf['survive_current_icu'].mean(),
				'unit_readmission': dummydf['unit_readmission'].mean(),
				})

		patient_mean_df = pd.DataFrame(patient_mean_df)

		patient_scatter('will_return', 'will_readmit')
		patient_scatter('will_return', 'unit_readmission')
		patient_scatter('will_readmit', 'unit_readmission')
		patient_scatter('will_readmit', 'survive_current_icu')
		patient_scatter('unit_readmission', 'survive_current_icu')
		patient_scatter('will_die', 'survive_current_icu')


eICUdata = eICU_DataLoader(eICU_path, result_path, num_patients=-1)

print('\n\nfetched data patients:\n', eICUdata.dataframe_patients, '\n\n******************\n')
print('\n\nfetched data hospitals:\n', eICUdata.dataframe_hospitals, '\n\n******************\n')

