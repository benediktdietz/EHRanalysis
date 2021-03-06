import math, re, os
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 16})


def patch_loaded_hospital_files_together(path_to_dir):

	hospital_csv_files = [f for f in os.listdir(path_to_dir) if f.endswith('_hospital.csv')]
	dummyvar = 0
	for hospital_csv_file in hospital_csv_files:
		print('loading data from ', hospital_csv_file)
			
		new_feature_set = pd.read_csv(path_to_dir + hospital_csv_file)

		if dummyvar == 0:
			full_feature_set = new_feature_set
			dummyvar = 1
		else:
			full_feature_set = pd.concat([full_feature_set, new_feature_set])
		print('dimension of new feature set: ', len(new_feature_set))
		print('dimension of full feature set: ', len(full_feature_set))
		print('--------')

	pd.DataFrame(full_feature_set).to_csv(path_to_dir + 'full_loaded_set.csv')

	return pd.DataFrame(full_feature_set)

def patch_encoded_files_together(path_to_dir, list_of_hositalIDs, feature_df):

	print('\n\npatching files together for hospitals: ', list_of_hositalIDs, '\n\n')
	dummyvar = 0
	for hospitalID in list_of_hositalIDs:

		hospital_csv_file = path_to_dir + 'encoded_diagnosis_ICD9code_hospital_' + str(hospitalID) + '.csv'

		print('loading data from ', hospital_csv_file)
			
		new_feature_set = pd.read_csv(path_to_dir + hospital_csv_file)

		if dummyvar == 0:
			full_feature_set = new_feature_set
			dummyvar = 1
		else:
			full_feature_set = pd.concat([full_feature_set, new_feature_set], axis=0, sort=False).fillna(.0)

		print('loaded set: '.ljust(15, '.'), len(new_feature_set))
		print('full set: '.ljust(15, '.'), len(full_feature_set))
		print('***********************')


	# print('feature_map: ', len(feature_df))

	# feature_df = pd.merge(feature_df, pd.DataFrame(full_feature_set), on='corr_id')
	
	# print('feature_map: ', len(feature_df))

	pd.DataFrame(full_feature_set).to_csv(path_to_dir + 'full_encoded_set.csv')
	# pd.DataFrame(feature_df).to_csv(path_to_dir + 'full_processed_set.csv')

	return pd.DataFrame(full_feature_set)

def patch_encoded_hospital_files_together(path_to_dir):

	hospital_csv_files = [f for f in os.listdir(path_to_dir) if f.startswith('hospital_')]

	loaded_tags = ['pasthistory', 'lab', 'diagnosis', 'drugs']

	for loaded_tag in loaded_tags:
	
		dummyvar = 0

		current_files = [f for f in hospital_csv_files if f.endswith(loaded_tag + '.csv')]

		for hospital_file in current_files:
				
			new_feature_set = pd.read_csv(path_to_dir + hospital_file)

			if dummyvar == 0:
				full_feature_set = new_feature_set
				dummyvar = 1
			else:
				full_feature_set = pd.concat([full_feature_set, new_feature_set], axis=0)
			print('dimension of new feature set: ', len(new_feature_set))
			print('dimension of full feature set: ', len(full_feature_set))
			print('--------')

		pd.DataFrame(full_feature_set).fillna(.0).to_csv(path_to_dir + 'full_set_' + loaded_tag + '.csv')


	patched_files = [f for f in os.listdir(path_to_dir) if f.startswith('full_set_')]

	dummyvar = 0
	for patched_file in patched_files:

		print('patching file: ', patched_file)

		new_feature_set = pd.read_csv(path_to_dir + patched_file)

		if dummyvar == 0:
			full_set = new_feature_set
			dummyvar += 1
		else:
			full_set = pd.merge(full_set, new_feature_set, on='corr_id')

	full_set = pd.DataFrame(full_set)
	full_set.fillna(.0).to_csv(path_to_dir + 'full_encoded_featureset.csv')

	return full_set

def get_most_important_features(x_data, y_data, num_features_to_use, plotpath, targetlabelname):

	try:
		x_data = x_data.drop(columns='hospital_id')
	except KeyError:
		pass

	skl_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_samples=.8)
	# skl_model = LogisticRegressionCV(Cs=10, cv=5, solver='lbfgs', max_iter=10000, n_jobs=-1, multi_class='auto', random_state=42)
	skl_model.fit(x_data, y_data)


	sklmodel_sorted_idx = np.argsort(skl_model.feature_importances_)[-num_features_to_use:]
	# sklmodel_sorted_idx = np.argsort(skl_model.coef_)[-num_features_to_use:]

	plt.figure(figsize=(20,30))
	plt.barh(
		np.arange(0, len(sklmodel_sorted_idx)), 
		skl_model.feature_importances_[sklmodel_sorted_idx], 
		height=0.7)
	plt.yticks(np.arange(len(x_data.keys()[sklmodel_sorted_idx])), x_data.keys()[sklmodel_sorted_idx])
	plt.ylim((0, len(sklmodel_sorted_idx)+.5))
	plt.grid()
	plt.tight_layout()	
	plt.savefig(plotpath + 'feature_importances_randomforest_' + str(targetlabelname) + '.pdf')
	plt.close()


	permutation_analysis = permutation_importance(skl_model, x_data, y_data, n_repeats=10, n_jobs=-1, random_state=42)
	permutation_sorted_idx = permutation_analysis.importances_mean.argsort()[-num_features_to_use:]

	plt.figure(figsize=(20,30))
	plt.boxplot(
		permutation_analysis.importances[permutation_sorted_idx].T, 
		vert=False,
		labels=x_data.keys()[permutation_sorted_idx])
	plt.ylim((0, len(sklmodel_sorted_idx)+.5))
	plt.grid()
	plt.tight_layout()
	plt.savefig(plotpath + 'feature_importances_randomforest_permutation_' + str(targetlabelname) + '.pdf')
	plt.close()

	best_feature_list = []
	for bestfeat in x_data.keys().values[permutation_sorted_idx]:
		best_feature_list.append(bestfeat)
	best_feature_list.append('hospital_id')

	return best_feature_list



class eICU_DataLoader():

	def __init__(self, args):

		self.args = args
		self.read_path = self.args.eICU_path
		self.write_path = self.args.mydata_path
		self.num_patients = self.args.num_patients_to_load
		self.num_hospitals = self.args.num_hospitals_to_load
		print('writing files to ', self.write_path)
		if self.args.big_hospitals_first:
			self.sort_hospitals_ascending = False
		else:
			self.sort_hospitals_ascending = True


		self.build_patient_matrix()

	def build_patient_matrix(self):

		remove_nans_from_codes = True

		patient_table = pd.read_csv(self.read_path + 'patient.csv')
		patient_table = patient_table.loc[patient_table['age'] != 'NaN']
		patient_table = patient_table.loc[patient_table['age'] != 'nan']
		patient_table = patient_table.loc[patient_table['gender'] != 'Unknown']


		print('\n\n')
		print('patient_table loaded successfully'.ljust(50) + str(np.round(len(patient_table)/1000000., 1)) + ' Mio rows | ' + str(int(patient_table.shape[1])) + ' cols')
		medication_table = pd.read_csv(self.read_path + 'medication.csv', low_memory=False)
		medication_table = medication_table.loc[medication_table['drugordercancelled'] == 'No']
		print('medication_table loaded successfully'.ljust(50) + str(np.round(len(medication_table)/1000000., 1)) + ' Mio rows | ' + str(int(medication_table.shape[1])) + ' cols')
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
				dummy = 0.
			return dummy


		if self.num_patients < 0:
			num_patients_to_load = len(patient_table['uniquepid'].unique())
		else:
			num_patients_to_load = self.num_patients
		if self.num_hospitals < 0:
			num_hospitals_to_load = len(patient_table['hospitalid'].unique())
		else:
			num_hospitals_to_load = self.num_hospitals

		patientIDs = patient_table['uniquepid'].unique()[:num_patients_to_load]
		hospitalIDs = patient_table['hospitalid'].unique()

		hospital_population_table = []
		for hospital_id_dummy in hospitalIDs:
			dummytable = patient_table[patient_table['hospitalid'] == hospital_id_dummy]
			hospital_population_table.append({
				'hospital_id': hospital_id_dummy,
				'hospital_population': len(dummytable),
				})
		hospital_population_table = pd.DataFrame(hospital_population_table).sort_values('hospital_population', ascending=self.sort_hospitals_ascending)

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=123)


		corr_id_df = []
		print('looping through patient IDs in loaded tables to build a consolidated matrix...\n\n')

		for hospital_id_dummy in hospital_population_table['hospital_id'].values[:num_hospitals_to_load]:


			print('\nprocessing hospital ', hospital_id_dummy)

			pbardummy, pbarfreq = 0, 10
			pbar = tqdm(total=len(patient_table['uniquepid'].loc[patient_table['hospitalid'] == hospital_id_dummy].unique()))
			for patient in patient_table['uniquepid'].loc[patient_table['hospitalid'] == hospital_id_dummy].unique():


				pbar.update(1)
				pbardummy += 1


				correlated_unitstay_ids = np.asarray(patient_table['patientunitstayid'].loc[patient_table['uniquepid'] == patient].values)

				for j in range(len(correlated_unitstay_ids)):


					#################################
					###		  Patient Table 	  ###
					#################################

					current_visit_number = patient_table['unitvisitnumber'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()
					current_health_sys_id = patient_table['patienthealthsystemstayid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()

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
					
					lengthofstay = hospital_discharge_time
					lengthofICU = icu_discharge_time

					weight_dummy = np.float(patient_table['admissionweight'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item())
					height_dummy = np.float(patient_table['admissionheight'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()) / 100.
					bmi_dummy = weight_dummy / ((height_dummy * height_dummy) + 1e-6)
					if bmi_dummy > 200:
						bmi_dummy = 0.



					if lengthofstay > 24*5.: will_stay_long = 1.
					else: will_stay_long = 0.

					hospital_id = patient_table['hospitalid'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item()

					if patient_table['gender'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[0]].values.item() == 'Female':
						gender_dummy = 0.
					else: gender_dummy = 1.




					#################################
					###	     Diagnosis Table 	  ###
					#################################


					diagnosis_dummy_table = diagnosis_table[diagnosis_table['patientunitstayid'] == correlated_unitstay_ids[j]]
					diagnosis_dummy_table_processed = []

					for diag_id in diagnosis_dummy_table['diagnosisid']:

						diagnosis_dummy_table_processed.append({
							'diagnosisid': diag_id,
							'patientunitstayid': correlated_unitstay_ids[j],
							'icd9code': diagnosis_dummy_table['icd9code'].loc[diagnosis_dummy_table['diagnosisid'] == diag_id].values.item(),
							'diag_offset': diagnosis_dummy_table['diagnosisoffset'].loc[diagnosis_dummy_table['diagnosisid'] == diag_id].values.item(),
							'diag_activeUponDischarge': diagnosis_dummy_table['activeupondischarge'].loc[diagnosis_dummy_table['diagnosisid'] == diag_id].values.item(),
							})

					diagnosis_dummy_table_processed_df = pd.DataFrame(diagnosis_dummy_table_processed).dropna(axis=0)

					if not diagnosis_dummy_table_processed_df.empty:
						icd9codes = diagnosis_dummy_table_processed_df['icd9code'].values
						icd9offsets = diagnosis_dummy_table_processed_df['diag_offset'].values
						icd9active_at_discharge = diagnosis_dummy_table_processed_df['diag_activeUponDischarge'].values
					else:
						icd9codes = [0]
						icd9offsets = [0]
						icd9active_at_discharge = [0]


					#################################
					###	   	    Lab Table 	      ###
					#################################


					lab_dummy_table = lab_table[lab_table['patientunitstayid'] == correlated_unitstay_ids[j]]
					lab_dummy_table_processed = []

					for lab_id in lab_dummy_table['labid']:

						lab_dummy_table_processed.append({
							'lab_id': lab_id,
							'patientunitstayid': correlated_unitstay_ids[j],
							'result_offset': lab_dummy_table['labresultoffset'].loc[lab_dummy_table['labid'] == lab_id].values.item(),
							'labname': lab_dummy_table['labname'].loc[lab_dummy_table['labid'] == lab_id].values.item(),
							'labresulttext': lab_dummy_table['labresulttext'].loc[lab_dummy_table['labid'] == lab_id].values.item(),
							})

					lab_dummy_table_processed_df = pd.DataFrame(lab_dummy_table_processed).dropna(axis=0)

					if not lab_dummy_table_processed_df.empty:
						lab_result_offsets = lab_dummy_table_processed_df['result_offset'].values
						lab_name_offsets = lab_dummy_table_processed_df['labname'].values
						lab_result_texts = lab_dummy_table_processed_df['labresulttext'].values
					else:
						lab_result_offsets = [0]
						lab_name_offsets = [0]
						lab_result_texts = [0]


					#################################
					###	    PastHistory Table 	  ###
					#################################


					past_hist_dummy_table = pasthistory_table[pasthistory_table['patientunitstayid'] == correlated_unitstay_ids[j]]
					past_hist_dummy_table_processed = []

					for past_hist_id in past_hist_dummy_table['pasthistoryid']:

						past_hist_dummy_table_processed.append({
							'past_hist_id': past_hist_id,
							'patientunitstayid': correlated_unitstay_ids[j],
							'pasthistory_offset': past_hist_dummy_table['pasthistoryoffset'].loc[past_hist_dummy_table['pasthistoryid'] == past_hist_id].values.item(),
							'pasthistory_entered_offset': past_hist_dummy_table['pasthistoryenteredoffset'].loc[past_hist_dummy_table['pasthistoryid'] == past_hist_id].values.item(),
							'pasthistory_notetype': past_hist_dummy_table['pasthistorynotetype'].loc[past_hist_dummy_table['pasthistoryid'] == past_hist_id].values.item(),
							'pasthistory_value': past_hist_dummy_table['pasthistoryvalue'].loc[past_hist_dummy_table['pasthistoryid'] == past_hist_id].values.item(),
							})

					past_hist_dummy_table_processed_df = pd.DataFrame(past_hist_dummy_table_processed).dropna(axis=0)

					if not past_hist_dummy_table_processed_df.empty:
						pasthistory_offsets = past_hist_dummy_table_processed_df['pasthistory_offset'].values
						pasthistory_entered_offsets = past_hist_dummy_table_processed_df['pasthistory_entered_offset'].values
						pasthistory_notetypes = past_hist_dummy_table_processed_df['pasthistory_notetype'].values
						pasthistory_values = past_hist_dummy_table_processed_df['pasthistory_value'].values
					else:
						pasthistory_offsets = [0]
						pasthistory_entered_offsets = [0]
						pasthistory_notetypes = [0]
						pasthistory_values = [0]



					#################################
					###	     Medication Table 	  ###
					#################################


					medication_dummy_table = medication_table[medication_table['patientunitstayid'] == correlated_unitstay_ids[j]]

					medication_dummy_table_processed = []
					

					for med_id in medication_dummy_table['medicationid']:

						medication_dummy_table_processed.append({
							'medication_id': med_id,
							'patientunitstayid': correlated_unitstay_ids[j],
							'drugstart_offset': medication_dummy_table['drugstartoffset'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							'drugorder_offset': medication_dummy_table['drugorderoffset'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							'drugstop_offset': medication_dummy_table['drugstopoffset'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							'drug_name': medication_dummy_table['drugname'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							'drug_hiclseqno': medication_dummy_table['drughiclseqno'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							'drug_dosage': medication_dummy_table['dosage'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							'drug_gtc_code': medication_dummy_table['gtc'].loc[medication_dummy_table['medicationid'] == med_id].values.item(),
							})

					medication_dummy_table_processed_df = pd.DataFrame(medication_dummy_table_processed).dropna(axis=0)

					if not medication_dummy_table_processed_df.empty:
						drugstart_offsets = medication_dummy_table_processed_df['drugstart_offset'].values
						drugorder_offsets = medication_dummy_table_processed_df['drugorder_offset'].values
						drugstop_offsets = medication_dummy_table_processed_df['drugstop_offset'].values
						drug_names = medication_dummy_table_processed_df['drug_name'].values
						drug_hiclseqnos = medication_dummy_table_processed_df['drug_hiclseqno'].values
						drug_dosages = medication_dummy_table_processed_df['drug_dosage'].values
						drug_gtc_codes = medication_dummy_table_processed_df['drug_gtc_code'].values
					else:
						drugstart_offsets = [0]
						drugorder_offsets = [0]
						drugstop_offsets = [0]
						drug_names = [0]
						drug_hiclseqnos = [0]
						drug_dosages = [0]
						drug_gtc_codes = [0]




					corr_id_df.append(
						{
						#################################
						###		  Patient Table 	  ###
						#################################
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
						'icu_discharge': icu_discharge_time,
						'length_of_stay': lengthofstay,
						'length_of_icu': lengthofICU,
						'will_return': will_return,
						'will_die': will_die,
						'will_readmit': will_readmit,
						'will_stay_long': will_stay_long,
						'survive_current_icu': survive_current_icu,
						'visits_current_stay': max_visits_for_current_stay,
						'hospital_discharge_status': patient_table['hospitaldischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						'hospital_admit_offset': icu_admission_time,
						'hospital_discharge_offset': hospital_discharge_time,
						'hospital_discharge_year': patient_table['hospitaldischargeyear'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						'hospital_id': hospital_id,
						'unit_readmission': unit_readmission,
						'unit_admit_source': patient_table['unitadmitsource'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						'unit_type': patient_table['unittype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						'unit_discharge_status': patient_table['unitdischargestatus'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						'unit_discharge_offset': icu_discharge_time,
						'unit_discharge_location': patient_table['unitdischargelocation'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						'unit_stay_type': patient_table['unitstaytype'].loc[patient_table['patientunitstayid'] == correlated_unitstay_ids[j]].values.item(),
						#################################
						###	     Diagnosis Table 	  ###
						#################################
						'diagnosis_activeUponDischarge': icd9active_at_discharge,
						'diagnosis_offset': icd9offsets,
						'diagnosis_ICD9code': icd9codes,
						#################################
						###	   	    Lab Table 	      ###
						#################################
						'lab_result_offsets': lab_result_offsets,
						'lab_name_offsets': lab_name_offsets,
						'lab_result_texts': lab_result_texts,
						#################################
						###	    PastHistory Table 	  ###
						#################################
						'pasthistory_offsets': pasthistory_offsets,
						'pasthistory_entered_offsets': pasthistory_entered_offsets,
						'pasthistory_notetypes': pasthistory_notetypes,
						'pasthistory_values': pasthistory_values,
						#################################
						###	     Medication Table 	  ###
						#################################
						'drugstart_offsets': drugstart_offsets,
						'drugorder_offsets': drugorder_offsets,
						'drugstop_offsets': drugstop_offsets,
						'drug_names': drug_names,
						'drug_hiclseqnos': drug_hiclseqnos,
						'drug_dosages': drug_dosages,
						'drug_gtc_codes': drug_gtc_codes,
						#################################
						###	       Apache Table 	  ###
						#################################
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

			pd.DataFrame(corr_id_df).to_csv(self.write_path[:-4] + '_' + str(hospital_id_dummy) + '_hospital.csv')
			corr_id_df = []


		# pd.DataFrame(corr_id_df).to_csv(self.write_path)

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
		self.read_path_files = self.args.mydata_path_files
		print('writing files to ', self.read_path_files)
		self.min_patients_per_client = self.args.min_patients_per_hospital

		self.offset_threshold = 1000

		self.add_hosp_stats_to_features = False

		try:
			self.dataframe = pd.read_csv(self.read_path_files + 'full_loaded_set.csv')
		except FileNotFoundError:
			self.dataframe = patch_loaded_hospital_files_together(self.read_path_files)


		hospital_population_table = []
		for hospital_id_dummy in self.dataframe['hospital_id'].unique():
			dummytable = self.dataframe[self.dataframe['hospital_id'] == hospital_id_dummy]
			hospital_population_table.append({
				'hospital_id': hospital_id_dummy,
				'hospital_population': len(dummytable),
				})
		hospital_population_table = pd.DataFrame(hospital_population_table).sort_values('hospital_population', ascending=False)
		self.loaded_hospital_ids = hospital_population_table['hospital_id'].iloc[:self.args.num_hospitals_to_load].values
		print('\n\n************* loaded_hospital_ids ', self.loaded_hospital_ids, '\n\n')



		categorical_feature_names = [
			'ethnicity',
			# 'hospital_discharge_status',
			'hospital_discharge_year',
			# 'unit_discharge_status',
			'unit_admit_source',
			'unit_type',
			# 'unit_discharge_location',
			'unit_stay_type',
			]

		features_to_drop = [
			'diagnosis_activeUponDischarge',
			'diagnosis_offset',
			'diagnosis_ICD9code',
			'lab_result_offsets',
			'lab_name_offsets',
			'lab_result_texts',
			'pasthistory_offsets',
			'pasthistory_entered_offsets',
			'pasthistory_notetypes',
			'pasthistory_values',
			'drugstart_offsets',
			'drugorder_offsets',
			'drugstop_offsets',
			'drug_names',
			'drug_hiclseqnos',
			'drug_dosages',
			'drug_gtc_codes',
			################
			# 'hospital_discharge_status_Alive',
			# 'hospital_discharge_status_Expired',
			# 'unit_discharge_status_Alive',
			# 'unit_discharge_status_Expired',
			'hospital_discharge_status',
			'unit_discharge_status',
			'unit_discharge_location',
			'icu_admission_time',
			'icu_discharge',
			'hospital_admit_offset',
			'hospital_discharge_offset',
			'unit_discharge_offset',
			################
			'Unnamed: 0_x',
			'Unnamed: 0.1_x',
			'Unnamed: 0',
			'Unnamed: 0_y',
			'Unnamed: 0.1_y',
			]


		self.feature_df = pd.get_dummies(self.dataframe, columns = categorical_feature_names, prefix = categorical_feature_names)
		
		self.add_hospital_stats()
		
		self.process_pasthist_col()
		self.process_drugs_col(self.offset_threshold)
		self.process_lab_col(self.offset_threshold)
		self.process_diag_col(self.offset_threshold)

		self.full_encoded_set = patch_encoded_hospital_files_together(self.read_path_files)

		self.feature_df = pd.merge(self.feature_df, self.full_encoded_set, on='corr_id')

		self.feature_df.to_csv(self.write_path[:-4] + '_with_DiagnosisCodes_' + str(self.offset_threshold) + '.csv')
		for feature_to_drop in features_to_drop:
			try: self.feature_df = self.feature_df.drop(columns=feature_to_drop)
			except KeyError: continue
		self.feature_df.to_csv(self.write_path[:-4] + '_' + str(self.offset_threshold) + '.csv')


	def process_array_cols(self, col_names):

		print_out = False
		progbar = True
		pbarfreq = 10
		pbarcounter = 0

		for col_name in col_names:

			print('\nlooping through ' + col_name + ' column to build encoded feature map...')

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
					pbar.update(1)
					pbarcounter += 1


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

		pd.DataFrame(self.feature_df).to_csv(self.write_path[:-4] + '_diagnosis_loaded.csv')

	def process_diag_col(self, diag_offset_threshold):

		for hospital_id_dummy in self.loaded_hospital_ids:
			print('processing diagnosis data for hospital ', hospital_id_dummy)

			dummy_hospital_df = self.dataframe[self.dataframe['hospital_id'] == hospital_id_dummy]
		
			pbarcounter = 0
			pbar = tqdm(total=len(dummy_hospital_df))

			for row in range(len(dummy_hospital_df)):

				splitted_icd9_codes = re.split("'", dummy_hospital_df['diagnosis_ICD9code'].iloc[row])
				splitted_icd9_offsets = re.split(" ", dummy_hospital_df['diagnosis_offset'].iloc[row])

				if '[' in splitted_icd9_offsets: splitted_icd9_offsets.remove('[')
				if ']' in splitted_icd9_offsets: splitted_icd9_offsets.remove(']')
				if '' in splitted_icd9_offsets: splitted_icd9_offsets.remove('')
				if ' ' in splitted_icd9_offsets: splitted_icd9_offsets.remove(' ')

				if '[' in splitted_icd9_codes: splitted_icd9_codes.remove('[')
				if ']' in splitted_icd9_codes: splitted_icd9_codes.remove(']')
				if '[]' in splitted_icd9_codes: splitted_icd9_codes.remove('[]')
				if '[nan ' in splitted_icd9_codes: splitted_icd9_codes.remove('[nan ')
				if ' nan nan]' in splitted_icd9_codes: splitted_icd9_codes.remove(' nan nan]')
				if ' nan nan\n ' in splitted_icd9_codes: splitted_icd9_codes.remove(' nan nan\n ')

				if len(splitted_icd9_offsets) > 1:
					splitted_icd9_offsets = [f for f in splitted_icd9_offsets if len(f) >= 1]
				if splitted_icd9_offsets[0][0] == '[':
					splitted_icd9_offsets[0] = splitted_icd9_offsets[0][1:]
				if splitted_icd9_offsets[-1][-1] == ']':
					splitted_icd9_offsets[-1] = splitted_icd9_offsets[-1][:-1]
				if len(splitted_icd9_offsets) > 1:
					splitted_icd9_offsets = [int(f) for f in splitted_icd9_offsets]

				for dummy in splitted_icd9_codes:
					if len(dummy) < 4:
						splitted_icd9_codes.remove(dummy)

				splitted_icd9_codes = re.split(', ', str(splitted_icd9_codes).translate({ord(c): None for c in "'!@#$[]"}))

				for dummy in splitted_icd9_codes:

					if len(dummy) < 3:
						splitted_icd9_codes.remove(dummy)

					if not str(dummy).upper().isupper():
						try:
							splitted_icd9_codes.remove(dummy)
						except ValueError:
							continue

				icd10tabledummy = []
				for entry in splitted_icd9_codes:

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


					icd10tabledummy.append({
						'icd10code_letter': icd10code_letter,
						'icd10code_number': icd10code_number,
						'icd10code_decimal': icd10code_decimal,
						'icd10code_full': icd10code_letter + icd10code_number + '.' + icd10code_decimal,
						})

				icd10tabledummy = pd.DataFrame(icd10tabledummy)

				if icd10tabledummy.empty:
					dummy_hospital_df['diagnosis_ICD9code'].iloc[row] = [0]
				if not icd10tabledummy.empty:
					splitted_icd9_codes = icd10tabledummy['icd10code_full'].values

					# print(np.asarray(splitted_icd9_offsets) < diag_offset_threshold)

					if len(splitted_icd9_codes) == len(splitted_icd9_offsets):
						# splitted_icd9_codes = splitted_icd9_codes[np.asarray(splitted_icd9_offsets) < diag_offset_threshold]
						splitted_icd9_codes_dummy = []
						for j in range(len(splitted_icd9_offsets)):
							if float(splitted_icd9_offsets[j]) < diag_offset_threshold:
								splitted_icd9_codes_dummy.append(splitted_icd9_codes[j])
						splitted_icd9_codes = splitted_icd9_codes_dummy
						dummy_hospital_df['diagnosis_ICD9code'].iloc[row] = splitted_icd9_codes				
					else:
						dummy_hospital_df['diagnosis_ICD9code'].iloc[row] = [0]

				onehot_keys = np.concatenate([['corr_id'], np.asarray(pd.get_dummies(
					dummy_hospital_df['diagnosis_ICD9code'].iloc[row],
					prefix = 'diagnosis_ICD9code'
					).sum(0).keys().values)])
				onehot_values = np.reshape(
					np.concatenate(
						[[dummy_hospital_df['corr_id'].iloc[row]], 
						np.asarray(
							pd.get_dummies(
								dummy_hospital_df['diagnosis_ICD9code'].iloc[row], 
								prefix = 'diagnosis_ICD9code'
								).sum(0).values, 
							dtype=np.int)
						]), (1,-1))

				if row == 0: dummy_df = pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)
				else: dummy_df = pd.concat([dummy_df, pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)], axis=0, sort=False).fillna(.0)
				
				pbar.update(1)
				pbarcounter += 1
						
			pbar.close()

			pd.DataFrame(dummy_df).to_csv(self.read_path_files+ 'hospital_' + str(hospital_id_dummy) + '_diagnosis.csv')

	def process_lab_col(self, lab_offset_threshold):

		for hospital_id_dummy in self.loaded_hospital_ids:
			print('processing lab data for hospital ', hospital_id_dummy)

			dummy_hospital_df = self.dataframe[self.dataframe['hospital_id'] == hospital_id_dummy]
		
			pbarcounter = 0
			loop_helper = 0
			pbar = tqdm(total=len(dummy_hospital_df))

			for row in range(len(dummy_hospital_df)):


				splitted_lab_names = re.split("'", dummy_hospital_df['lab_name_offsets'].iloc[row])
				splitted_lab_offsets = re.split(" ", dummy_hospital_df['lab_result_offsets'].iloc[row])
				splitted_lab_values = re.split("'", dummy_hospital_df['lab_result_texts'].iloc[row])


				if '[' in splitted_lab_offsets: splitted_lab_offsets.remove('[')
				if ']' in splitted_lab_offsets: splitted_lab_offsets.remove(']')
				if '' in splitted_lab_offsets: splitted_lab_offsets.remove('')
				if ' ' in splitted_lab_offsets: splitted_lab_offsets.remove(' ')

				if '[' in splitted_lab_values: splitted_lab_values.remove('[')
				if ']' in splitted_lab_values: splitted_lab_values.remove(']')
				if '' in splitted_lab_values: splitted_lab_values.remove('')
				if ' ' in splitted_lab_values: splitted_lab_values.remove(' ')

				if '[' in splitted_lab_names: splitted_lab_names.remove('[')
				if ']' in splitted_lab_names: splitted_lab_names.remove(']')
				if '[]' in splitted_lab_names: splitted_lab_names.remove('[]')
				if '[nan ' in splitted_lab_names: splitted_lab_names.remove('[nan ')
				if ' nan nan]' in splitted_lab_names: splitted_lab_names.remove(' nan nan]')
				if ' nan nan\n ' in splitted_lab_names: splitted_lab_names.remove(' nan nan\n ')


				if len(splitted_lab_offsets) > 1:
					splitted_lab_offsets = [f for f in splitted_lab_offsets if len(f) > 1]
					if splitted_lab_offsets[0][0] == '[':
						splitted_lab_offsets[0] = splitted_lab_offsets[0][1:]
					if splitted_lab_offsets[0][0] == '<':
						splitted_lab_offsets[0] = splitted_lab_offsets[0][1:]
					if splitted_lab_offsets[0][0] == '>':
						splitted_lab_offsets[0] = splitted_lab_offsets[0][1:]
					if splitted_lab_offsets[-1][-1] == ']':
						splitted_lab_offsets[-1] = splitted_lab_offsets[-1][:-1]

					splitted_lab_offsets = [int(f) for f in splitted_lab_offsets if f not in ['\n', '\n ', ' ', '...', ' ... ']]

				if len(splitted_lab_values) > 1:
					splitted_lab_values = [f for f in splitted_lab_values if f not in ['\n', '\n ', ' ', '...', ' ... ']]

				splitoffsetarray = []
				for splitoffset in splitted_lab_offsets:
					if str(splitoffset)[0] == '[':
						splitoffset = str(splitoffset)[1:]
					if str(splitoffset)[0] in ['>', '<']:
						splitoffset = str(splitoffset)[1:]
					if str(splitoffset)[-1] == ']':
						splitoffset = str(splitoffset)[:-1]
					splitoffsetarray.append(int(splitoffset))
				splitted_lab_offsets = np.asarray(splitoffsetarray)

				splitvaluearray = []
				for splitvalue in splitted_lab_values:
					if str(splitvalue)[0] == '[':
						splitvalue = str(splitvalue)[1:]
					if str(splitvalue)[0] in ['>', '<']:
						splitvalue = str(splitvalue)[1:]
					if str(splitvalue)[-1] == ' ':
						splitvalue = str(splitvalue)[:-1]
					if str(splitvalue)[-1] == ']':
						splitvalue = str(splitvalue)[:-1]
					if str(splitvalue)[-1] == '%':
						splitvalue = str(splitvalue)[:-1]
					splitvaluearray.append(float(splitvalue))
				splitted_lab_values = np.asarray(splitvaluearray)


				for dummy in splitted_lab_names:
					if len(dummy) < 3:
						splitted_lab_names.remove(dummy)

				splitted_lab_names = re.split(', ', str(splitted_lab_names).translate({ord(c): None for c in "'!@#$[]"}))


				if len(splitted_lab_names) == len(splitted_lab_values) and len(splitted_lab_names) == len(splitted_lab_offsets):


					labtabledummy = []
					for ii in range(len(splitted_lab_names)):

						labtabledummy.append({
							'lab_name': splitted_lab_names[ii],
							'lab_value': splitted_lab_values[ii],
							'lab_offset': splitted_lab_offsets[ii],
							})

					labtabledummy = pd.DataFrame(labtabledummy)


					if not labtabledummy.empty:

						lab_info_table = []

						lab_name_dummies = labtabledummy['lab_name'].values
						lab_value_dummies = labtabledummy['lab_value'].values
						lab_offset_dummies = labtabledummy['lab_offset'].astype(int).values

						lab_name_dummies = lab_name_dummies[np.asarray(lab_offset_dummies) < lab_offset_threshold]
						lab_value_dummies = lab_value_dummies[np.asarray(lab_offset_dummies) < lab_offset_threshold]
						lab_offset_dummies = lab_offset_dummies[np.asarray(lab_offset_dummies) < lab_offset_threshold]

						for ii in range(len(lab_name_dummies)):

							lab_name_dummy = lab_name_dummies[ii]
							lab_value_dummy = lab_value_dummies[ii]

							lab_info_table.append({
								'corr_id': dummy_hospital_df['corr_id'].iloc[row],
								lab_name_dummy: lab_value_dummy,
								})

						lab_info_table = pd.DataFrame(lab_info_table)

						if loop_helper == 0:
							dummy_df = lab_info_table
							loop_helper += 1
						else: 
							dummy_df = pd.concat([dummy_df, lab_info_table], axis=0, sort=False).fillna(.0)

				pbar.update(1)
				pbarcounter += 1
					


			dummy_df = pd.DataFrame(dummy_df)

			# if '0' in dummy_df.keys():
			# 	dummy_df = dummy_df.drop(columns='0')


			hospital_lab_table_file = []
			dummyloophelper = 0

			for caseid in dummy_df['corr_id'].unique():

				dummy_table = dummy_df[dummy_df['corr_id'] == caseid]
				dummy_table = dummy_table.astype(float)

				for dummykey in dummy_table.keys():

					dummy_table[dummykey] = dummy_table[dummykey].max()

				dummy_table = pd.DataFrame(dummy_table).iloc[0,:]


				if dummyloophelper == 0:
					hospital_lab_table_file = dummy_table
					dummyloophelper += 1
				else:
					hospital_lab_table_file = pd.concat([hospital_lab_table_file, dummy_table], axis=1)


			hospital_lab_table_file = pd.DataFrame(hospital_lab_table_file).transpose()
			hospital_lab_table_file.to_csv(self.read_path_files+ 'hospital_' + str(hospital_id_dummy) + '_lab.csv')

			pbar.close()

	def process_drugs_col(self, lab_offset_threshold):

		for hospital_id_dummy in self.loaded_hospital_ids:
			print('processing drug data for hospital ', hospital_id_dummy)

			dummy_hospital_df = self.dataframe[self.dataframe['hospital_id'] == hospital_id_dummy]
		
			pbarcounter = 0
			loop_helper = 0
			pbar = tqdm(total=len(dummy_hospital_df))

			for row in range(len(dummy_hospital_df)):

				splitted_gtc_codes = re.split("'", dummy_hospital_df['drug_gtc_codes'].iloc[row])
				splitted_drugstart_offsets = re.split(" ", dummy_hospital_df['drugstart_offsets'].iloc[row])


				if '[' in splitted_drugstart_offsets: splitted_drugstart_offsets.remove('[')
				if ']' in splitted_drugstart_offsets: splitted_drugstart_offsets.remove(']')
				if '' in splitted_drugstart_offsets: splitted_drugstart_offsets.remove('')
				if ' ' in splitted_drugstart_offsets: splitted_drugstart_offsets.remove(' ')


				if '' in splitted_gtc_codes: splitted_gtc_codes.remove('')
				if ' ' in splitted_gtc_codes: splitted_gtc_codes.remove(' ')
				if '[' in splitted_gtc_codes: splitted_gtc_codes.remove('[')
				if ']' in splitted_gtc_codes: splitted_gtc_codes.remove(']')
				if '[]' in splitted_gtc_codes: splitted_gtc_codes.remove('[]')
				if '[nan ' in splitted_gtc_codes: splitted_gtc_codes.remove('[nan ')
				if ' nan nan]' in splitted_gtc_codes: splitted_gtc_codes.remove(' nan nan]')
				if ' nan nan\n ' in splitted_gtc_codes: splitted_gtc_codes.remove(' nan nan\n ')


				splitted_drugstart_offsets = [f for f in splitted_drugstart_offsets if len(f) > 1]
				if splitted_drugstart_offsets[0][0] == '[':
					splitted_drugstart_offsets[0] = splitted_drugstart_offsets[0][1:]
				if splitted_drugstart_offsets[0][0] == '<':
					splitted_drugstart_offsets[0] = splitted_drugstart_offsets[0][1:]
				if splitted_drugstart_offsets[0][0] == '>':
					splitted_drugstart_offsets[0] = splitted_drugstart_offsets[0][1:]
				if splitted_drugstart_offsets[-1][-1] == ']':
					splitted_drugstart_offsets[-1] = splitted_drugstart_offsets[-1][:-1]
				splitted_drugstart_offsets = [int(f) for f in splitted_drugstart_offsets]


				splitted_gtc_codes = re.split(', ', str(splitted_gtc_codes).translate({ord(c): None for c in "'!@#$[]"}))
				splitted_gtc_codes = re.split(' ', str(splitted_gtc_codes).translate({ord(c): None for c in "'!@#$[]"}))
				if '' in splitted_gtc_codes: splitted_gtc_codes.remove('')
				splitted_gtc_codes = [f for f in splitted_gtc_codes if f.isnumeric()]


				if len(splitted_gtc_codes) == len(splitted_drugstart_offsets):

					drugtabledummy = []
					for ii in range(len(splitted_gtc_codes)):
						drugtabledummy.append({
							'gtc_code': splitted_gtc_codes[ii],
							'drugstart_offset': splitted_drugstart_offsets[ii],
							})
					drugtabledummy = pd.DataFrame(drugtabledummy)


					if not drugtabledummy.empty:

						druginfotable = []

						gtc_code_dummies = drugtabledummy['gtc_code'].values
						drugstart_offset_dummies = drugtabledummy['drugstart_offset'].values

						gtc_code_dummies = gtc_code_dummies[np.asarray(drugstart_offset_dummies) < lab_offset_threshold]
						drugstart_offset_dummies = drugstart_offset_dummies[np.asarray(drugstart_offset_dummies) < lab_offset_threshold]

						onehot_keys = np.concatenate([['corr_id'], np.asarray(pd.get_dummies(
							gtc_code_dummies,
							prefix = 'gtc_code'
							).sum(0).keys().values)])
						onehot_values = np.reshape(
							np.concatenate(
								[[dummy_hospital_df['corr_id'].iloc[row]], 
								np.asarray(
									pd.get_dummies(
										gtc_code_dummies, 
										prefix = 'gtc_code'
										).sum(0).values, 
									dtype=np.int)
								]), (1,-1))

						if loop_helper == 0: 
							dummy_df = pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)
							loop_helper += 1
						else: 
							dummy_df = pd.concat([dummy_df, pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)], axis=0, sort=False).fillna(.0)
					

				pbar.update(1)
				pbarcounter += 1
			pbar.close()
	

			dummy_df = pd.DataFrame(dummy_df)

			if '0' in dummy_df.keys():
				dummy_df = dummy_df.drop(columns='0')

			dummy_df.to_csv(self.read_path_files+ 'hospital_' + str(hospital_id_dummy) + '_drugs.csv')
	
	def process_pasthist_col(self):

		for hospital_id_dummy in self.loaded_hospital_ids:
			print('processing pasthistory data for hospital ', hospital_id_dummy)

			dummy_hospital_df = self.dataframe[self.dataframe['hospital_id'] == hospital_id_dummy]
		
			pbarcounter = 0
			loop_helper = 0
			pbar = tqdm(total=len(dummy_hospital_df))

			for row in range(len(dummy_hospital_df)):

				splitted_pasthist_items = re.split("'", dummy_hospital_df['pasthistory_notetypes'].iloc[row])

				if '' in splitted_pasthist_items: splitted_pasthist_items.remove('')
				if ' ' in splitted_pasthist_items: splitted_pasthist_items.remove(' ')
				if '\\n ' in splitted_pasthist_items: splitted_pasthist_items.remove('\\n ')
				if '[' in splitted_pasthist_items: splitted_pasthist_items.remove('[')
				if ']' in splitted_pasthist_items: splitted_pasthist_items.remove(']')
				if '[]' in splitted_pasthist_items: splitted_pasthist_items.remove('[]')
				if '[nan ' in splitted_pasthist_items: splitted_pasthist_items.remove('[nan ')
				if ' nan nan]' in splitted_pasthist_items: splitted_pasthist_items.remove(' nan nan]')
				if ' nan nan\n ' in splitted_pasthist_items: splitted_pasthist_items.remove(' nan nan\n ')


				splitted_pasthist_items = re.split(', ', str(splitted_pasthist_items).translate({ord(c): None for c in "'!@#$[]"}))
				if '\\n ' in splitted_pasthist_items: splitted_pasthist_items.remove('\\n ')
				if ' ' in splitted_pasthist_items: splitted_pasthist_items.remove(' ')

				splitted_pasthist_items = [f for f in splitted_pasthist_items if f not in [' ', '', '\\n ']]


				if len(splitted_pasthist_items) >= 1:

					onehot_keys = np.concatenate([['corr_id'], np.asarray(pd.get_dummies(
						splitted_pasthist_items,
						prefix = 'pasthistory'
						).sum(0).keys().values)])
					onehot_values = np.reshape(
						np.concatenate(
							[[dummy_hospital_df['corr_id'].iloc[row]], 
							np.asarray(
								pd.get_dummies(
									splitted_pasthist_items, 
									prefix = 'pasthistory'
									).sum(0).values, 
								dtype=np.int)
							]), (1,-1))

					if loop_helper == 0: 
						dummy_df = pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)
						loop_helper += 1
					else: 
						dummy_df = pd.concat([dummy_df, pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)], axis=0, sort=False).fillna(.0)
				

				pbar.update(1)
				pbarcounter += 1
			pbar.close()
	

			dummy_df = pd.DataFrame(dummy_df)

			if '0' in dummy_df.keys():
				dummy_df = dummy_df.drop(columns='0')

			dummy_df.to_csv(self.read_path_files+ 'hospital_' + str(hospital_id_dummy) + '_pasthistory.csv')

	def process_array_cols_new(self, col_names):

		print_out = False
		progbar = True
		pbarfreq = 10
		pbarcounter = 0

		for hospital_id_dummy in self.loaded_hospital_ids:
			print('\nprocessing columns for hospital ', hospital_id_dummy)
			dummy_hospital_df = self.dataframe[self.dataframe['hospital_id'] == hospital_id_dummy]

			for col_name in col_names:
				# print('\nlooping through ' + col_name + ' column to build encoded feature map...')
				if progbar: pbar = tqdm(total=len(dummy_hospital_df))


				for row in range(len(dummy_hospital_df)):

					if col_name in ['pasthistory_notetypes', 'pasthistory_values']:

						nice_dummy_list = list(re.split("'", dummy_hospital_df[col_name].iloc[row]))

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

						dummy_hospital_df[col_name].iloc[row] = nice_dummy_list2

					if col_name in ['drug_strings_prescribed', 'diagnosis_string', 'lab_names', 'diagnosis_priority']:

						nice_dummy_list = list(re.split("'", dummy_hospital_df[col_name].iloc[row]))

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

						dummy_hospital_df[col_name].iloc[row] = nice_dummy_list2


					if col_name in ['medication_ids', 'diagnosis_ids', 'drug_codes_prescribed', 'lab_type_ids']:

						nice_dummy_list = list(re.split(" ", dummy_hospital_df[col_name].iloc[row]))

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
											

						dummy_hospital_df[col_name].iloc[row] = nice_dummy_list2


					if col_name == 'diagnosis_ICD9code':


						splitted_entry = re.split("'", dummy_hospital_df[col_name].iloc[row])


						for dummy in splitted_entry:

							if len(dummy) < 4:

								splitted_entry.remove(dummy)

						# print(re.split(',', str(splitted_entry)))

						splitted_entry = re.split(', ', str(splitted_entry).translate({ord(c): None for c in "'!@#$[]"}))


						for dummy in splitted_entry:

							if len(dummy) < 3:
								splitted_entry.remove(dummy)

							if not str(dummy).upper().isupper():
								try:
									splitted_entry.remove(dummy)
								except ValueError:
									continue

						icd10tabledummy = []
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


							icd10tabledummy.append({
								'patient_id': diagnosis_df['patient_id'].iloc[i],
								'health_system_id': diagnosis_df['health_system_id'].iloc[i],
								'corr_id': diagnosis_df['corr_id'].iloc[i],
								'hospital_id': diagnosis_df['hospital_id'].iloc[i],
								'icd10code_letter': icd10code_letter,
								'icd10code_number': icd10code_number,
								'icd10code_decimal': icd10code_decimal,
								'icd10code_full': icd10code_letter + icd10code_number + '.' + icd10code_decimal,
								})

						dummy_hospital_df[col_name].iloc[row] = pd.DataFrame(icd10tabledummy)['icd10code_full'].values







					onehot_keys = np.concatenate([['corr_id'], np.asarray(pd.get_dummies(
						dummy_hospital_df[col_name].iloc[row],
						prefix = col_name
						).sum(0).keys().values)])
					onehot_values = np.reshape(
						np.concatenate(
							[[dummy_hospital_df['corr_id'].iloc[row]], 
							np.asarray(
								pd.get_dummies(
									dummy_hospital_df[col_name].iloc[row], 
									prefix = col_name
									).sum(0).values, 
								dtype=np.int)
							]), (1,-1))
					
					if progbar:	
						pbar.update(1)
						pbarcounter += 1
							

					if row == 0: dummy_df = pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)
					else: dummy_df = pd.concat([dummy_df, pd.DataFrame(onehot_values, index = [row], columns = onehot_keys)], axis=0, sort=False).fillna(.0)

				if progbar: pbar.close()

				pd.DataFrame(dummy_df).to_csv(self.read_path_files + 'encoded_' + col_name + '_hospital_' + str(hospital_id_dummy) + '.csv')


			# self.feature_df.drop(columns=col_name)
			# self.feature_df = pd.merge(self.feature_df, dummy_df, on='corr_id')

			# if print_out: 
			# 	print(
			# 		'\n\n************************************\ndf_onehot DataFrame:\n', 
			# 		self.feature_df, 
			# 		'\n************************************\n\n\n')

		# pd.DataFrame(self.feature_df).to_csv(self.write_path[:-4] + '_diagnosis_loaded.csv')

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
		pbar = tqdm(total=len(corr_ids)+1)
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
		clinic_stats_df.sort_values('num_patients', ascending=False).to_csv(self.read_path_files + 'clinic_stats.csv')
		clinic_stats_df.reset_index(drop=True)

		if self.add_hosp_stats_to_features:

			print('\nattaching hospital stats dataframe to feature map...')
			pbar = tqdm(total=len(self.feature_df['corr_id'].unique())+1)

			extra_hospital_keys = clinic_stats_df.keys().values
			# self.feature_df[clinic_stats_df.keys().values] = 0.
			for extra_hospital_key in extra_hospital_keys:
				if extra_hospital_key != 'hospital_id':
					self.feature_df['clinic_stats_' + extra_hospital_key] = 0.



			self.feature_df.reset_index(drop=True)

			for stay_id in self.feature_df['corr_id'].unique():

				hospital_id_dummy = self.feature_df['hospital_id'].loc[self.feature_df['corr_id'] == stay_id].values[0]

				for clinic_key in extra_hospital_keys:

					if clinic_key != 'hospital_id':

						dummy = clinic_stats_df[clinic_key].loc[clinic_stats_df['hospital_id'] == hospital_id_dummy].values.item()
						self.feature_df['clinic_stats_' + clinic_key].loc[self.feature_df['corr_id'] == stay_id] = dummy

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
		self.feature_df = pd.read_csv(self.args.mydata_path_files + 'full_loaded_set.csv')

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
						try:
							splitted_entry.remove(dummy)
						except ValueError:
							continue

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

class DataAnalysis():

	def __init__(self, path_to_processed_feature_map, outpath):

		self.outpath = outpath
		self.data = pd.read_csv(path_to_processed_feature_map)
		try:
			self.data = self.data.drop(columns='Unnamed: 0')
		except KeyError:
			pass

		self.label_cols = [
			# doesnt make sense to include or not properly formatted cols
			'patient_id',
			'health_system_id',
			'corr_id',
			'hospital_discharge_year_2014',
			'hospital_discharge_year_2015',
			# labels we want to predict or shouldnt be available for our predictions
			'icu_admission_time',
			'icu_discharge',
			'diagnosis_offset',
			'diagnosis_activeUponDischarge',
			'diagnosis_ICD9code',
			'unit_discharge_offset',
			'unit_discharge_status_Alive',
			'unit_discharge_status_Expired',
			'unit_discharge_location_Death',
			'unit_discharge_location_Floor',
			'unit_discharge_location_Home',
			'unit_discharge_location_Other External',
			'unit_discharge_location_Other Hospital',
			'unit_discharge_location_Other ICU',
			'unit_discharge_location_Skilled Nursing Facility',
			'unit_discharge_location_Step-Down Unit (SDU)',
			'unit_stay_type_admit',
			'unit_stay_type_readmit',
			'unit_stay_type_stepdown/other',
			'unit_stay_type_transfer',
			'hospital_discharge_offset',
			'hospital_discharge_status_Alive',
			'hospital_discharge_status_Expired',
			'visits_current_stay',
			]

		self.target_features = [
			'length_of_stay',
			'length_of_icu',
			'will_return',
			'will_die',
			'will_readmit',
			'will_stay_long',
			'unit_readmission',
			'survive_current_icu',
			]

		for deletecol in self.label_cols:
			try:
				self.data = self.data.drop(columns = deletecol)
			except KeyError:
				continue
		self.data = self.data.fillna(0.)

		try: os.makedirs(self.outpath)
		except FileExistsError: pass

		self.get_correlations('pearson')
		self.get_correlations('spearman')

	def get_correlations(self, method):

		correlation_matrix = self.data.corr(method=method)

		for target_feat in self.target_features:

			target_correaltions = correlation_matrix[target_feat]

			target_correaltions = pd.DataFrame(target_correaltions).abs().sort_values(target_feat, ascending=False).iloc[1:201]

			print('\ntarget_correaltions\n')
			print(target_correaltions)

			plt.figure(figsize=(40,50))
			plt.title(target_feat)
			plt.barh(target_correaltions.index,target_correaltions[target_feat])
			plt.grid()
			plt.savefig(self.outpath + 'corr_' + method + '_' + target_feat + '.pdf')
			plt.close()

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
			self.mydata_path_files = self.args.mydata_path_files
		except AttributeError:
			self.process_data_path = self.args['mydata_path_processed']
			self.target_features = self.args['target_label']
			self.train_split = self.args['train_split']
			self.split_strategy = self.args['split_strategy']
			self.mydata_path_files = self.args['mydata_path_files']


		self.scaler_lo_icu = RobustScaler()
		self.scaler_lo_hospital = RobustScaler()
		self.scaler_features = RobustScaler()

		data_files = [f for f in os.listdir(self.mydata_path_files) if f.startswith('processed_featureset_')]

		self.data_df = pd.read_csv(self.mydata_path_files + data_files[0])
		# self.data_df = self.data_df.loc[self.data_df['unit_discharge_offset'] != 0].fillna(0.)
		self.data_df = self.data_df.fillna(0.)

		self.data_df['length_of_icu'].loc[self.data_df['length_of_icu'] > 1000.] = 1000.
		self.data_df['length_of_icu'].loc[self.data_df['length_of_icu'] < 1.] = 1.
		self.data_df['length_of_stay'].loc[self.data_df['length_of_stay'] > 5000.] = 5000.
		self.data_df['length_of_stay'].loc[self.data_df['length_of_stay'] < 1.] = 1.

		for uselesscol in ['Unnamed: 0.1', 'Unnamed: 0']:
			try:
				self.data_df = self.data_df.drop(columns=uselesscol)
			except KeyError:
				continue


		self.label_cols = [
			'patient_id',
			'health_system_id',
			'corr_id',
			'length_of_stay',
			'length_of_icu',
			'will_return',
			'will_die',
			'will_readmit',
			'will_stay_long',
			'survive_current_icu',
			'visits_current_stay',
			# 'hospital_id',
			'aps_intubated',
			'aps_vent',
			'aps_dialysis',
			]

		self.data_container, self.sampling_df = self.get_data()

	def split_data(self):

		unique_hospital_ids = self.data_df['hospital_id'].unique()

		# feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.).values
		# feature_map = np.nan_to_num(feature_map)

		# feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.)
		feature_map = self.data_df.copy()

		# feature_map = pd.get_dummies(feature_map, columns='hospital_id')


		for labelcol in self.label_cols:
			try:
				feature_map = feature_map.drop(columns = labelcol)
			except KeyError:
				continue
		feature_map = feature_map.fillna(0.).astype(float)

		features_to_scale = []
		for feat in feature_map.keys().values:
			if feature_map[feat].nunique() > 2 and feat != 'hospital_id':
				features_to_scale.append(feat)
		feature_map[features_to_scale] = self.scaler_features.fit_transform(feature_map[features_to_scale])
		# self.scaler_features.fit(feature_map)

		y_full = self.data_df[self.target_features]


		train_ids, val_ids, test_ids = [], [], []
		train_idx, val_idx, test_idx = [], [], []

		dummyrunner = 0

		sampling_df = []

		for hosp_id in unique_hospital_ids:

			hospital_dummy_df = self.data_df[['hospital_id', 'patient_id', 'corr_id']].loc[self.data_df['hospital_id'] == hosp_id]
			
			train_frac, val_frac, test_frac = np.split(
				hospital_dummy_df['patient_id'].sample(frac=1.), 
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


			if self.data_df['patient_id'].iloc[i] in train_ids:
				train_idx.append(i)
			if self.data_df['patient_id'].iloc[i] in val_ids:
				val_idx.append(i)
			if self.data_df['patient_id'].iloc[i] in test_ids:
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


		best_features = get_most_important_features(x_train, y_train, 100, self.mydata_path_files, self.target_features)
		feature_map = feature_map[best_features]
		x_train = x_train[best_features]
		x_val = x_val[best_features]
		x_test = x_test[best_features]


		print('\n\ntesting out RandomForests:')
		RandomForest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_samples=.8)
		RandomForest.fit(x_train, y_train)
		rf_y_val = RandomForest.predict_proba(x_val)
		rf_y_test = RandomForest.predict_proba(x_test)
		print('roc'.ljust(15,'.'), roc_auc_score(y_val, np.asarray(rf_y_val)[:,0]), ' / ', roc_auc_score(y_test, np.asarray(rf_y_test)[:,0]))
		print('accuracy'.ljust(15,'.'), RandomForest.score(x_val, y_val), ' / ', RandomForest.score(x_test, y_test), '\n')


		number_y_total = len(y_train)
		number_y_positive = y_train.sum()
		missing_y_samples = (number_y_total // 2) - number_y_positive
		available_positives_y = y_train[y_train == 1]
		available_positives_x = x_train[y_train == 1]

		try:
			random_choice = np.random.choice(len(available_positives_y), size=int(missing_y_samples))

			x_train = pd.concat([x_train, available_positives_x.iloc[random_choice]])
			y_train = pd.concat([y_train, available_positives_y.iloc[random_choice]])
		except ValueError:
			print('ValueError')
			pass


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
		
		# feature_map = self.data_df.drop(columns = self.label_cols).fillna(0.)

		feature_map = self.data_df
		for labelcol in self.label_cols:
			try:
				feature_map = feature_map.drop(columns = labelcol)
			except KeyError:
				continue
		feature_map = feature_map.fillna(0.).astype(float)

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

		return self.data_container['x_train'].drop(columns='hospital_id').values, np.reshape(self.data_container['y_train'].values, (-1,1))

	def get_full_val_data(self):

		return self.data_container['x_val'].drop(columns='hospital_id').values, np.reshape(self.data_container['y_val'].values, (-1,1))

	def get_full_test_data(self):

		return self.data_container['x_test'].drop(columns='hospital_id').values, np.reshape(self.data_container['y_test'].values, (-1,1))

	def get_full_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_full'][self.data_container['x_full']['hospital_id'] == hospital_id].drop(columns='hospital_id').values
		y_dummy = self.data_container['y_full'][self.data_container['x_full']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))

	def get_train_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_train'][self.data_container['x_train']['hospital_id'] == hospital_id].drop(columns='hospital_id').values
		y_dummy = self.data_container['y_train'][self.data_container['x_train']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))

	def get_test_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_test'][self.data_container['x_test']['hospital_id'] == hospital_id].drop(columns='hospital_id').values
		y_dummy = self.data_container['y_test'][self.data_container['x_test']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))

	def get_val_data_from_hopital(self, hospital_id):

		x_dummy = self.data_container['x_val'][self.data_container['x_val']['hospital_id'] == hospital_id].drop(columns='hospital_id').values
		y_dummy = self.data_container['y_val'][self.data_container['x_val']['hospital_id'] == hospital_id].values

		return np.asarray(x_dummy), np.reshape(np.asarray(y_dummy), (-1,1))


