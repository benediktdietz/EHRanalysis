from data_management import eICU_DataLoader, DataProcessor, DataSetIterator, DataManager
from models import Embedding, Classifier, Regressor
from network import NetworkTrainer

OUTPATH = '../results/new3/'

# path to the eICU CRD 2.0 CSV files
eICU_path = '../medical_data/eicu/physionet.org/files/eicu-crd/2.0/'
# path to processed DataFrame (combination of patient.csv, diagnosis.csv, medication.csv)
mydata_path = '../mydata/nomeds2k.csv'
# path to encoded DataFrame (one-hot encoding, redundant features dropped)
mydata_path_processed = '../mydata/nomeds2k_processed.csv'
# mydata_path_processed = '../mydata/nomeds_test_processed_consolidated.csv'

# loads orignal CSV files and builds combined DataFrame. uncomment if not yet available
eICU_DataLoader(eICU_path, mydata_path, num_patients=2000)
# loads combined DataFrame and builds encoded, useable DataFrame. uncomment if not yet available
DataProcessor(mydata_path, mydata_path_processed)


# eICU_data = DataManager(
# 	mydata_path_processed, 
# 	[
# 		'length_of_stay',
# 		'length_of_icu',
# 		'will_return',
# 		'will_die',
# 		'will_readmit',
# 		'will_stay_long',
# 		'unit_readmission',
# 		'survive_current_icu',
# 	])


# NetworkTrainer(eICU_data, 'length_of_icu', OUTPATH, 'regression')
# NetworkTrainer(eICU_data, 'length_of_stay', OUTPATH, 'regression')

# NetworkTrainer(eICU_data, 'survive_current_icu', OUTPATH, 'classification')
# NetworkTrainer(eICU_data, 'will_return', OUTPATH, 'classification')
# NetworkTrainer(eICU_data, 'will_die', OUTPATH, 'classification')
# NetworkTrainer(eICU_data, 'will_stay_long', OUTPATH, 'classification')
# NetworkTrainer(eICU_data, 'unit_readmission', OUTPATH, 'classification')
# NetworkTrainer(eICU_data, 'will_readmit', OUTPATH, 'classification')

# Regressor(eICU_data, 'length_of_stay', OUTPATH)
# Regressor(eICU_data, 'length_of_icu', OUTPATH)

# Classifier(eICU_data, 'will_return', OUTPATH)
# Classifier(eICU_data, 'will_die', OUTPATH)
# Classifier(eICU_data, 'will_readmit', OUTPATH)
# Classifier(eICU_data, 'will_stay_long', OUTPATH)
# Classifier(eICU_data, 'unit_readmission', OUTPATH)
# Classifier(eICU_data, 'survive_current_icu', OUTPATH)


# Classifier(eICU_data, 'unit_discharge_location_Death', OUTPATH)
# Classifier(eICU_data, 'hospital_discharge_status_Alive', OUTPATH)
# Classifier(eICU_data, 'hospital_discharge_status_Expired', OUTPATH)

# Embedding(eICU_data, OUTPATH)


