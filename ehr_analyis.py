from data_management import eICU_DataLoader, DataProcessor, DataSetIterator, DataManager
from models import Embedding, Classifier, Regressor

# path to the eICU CRD 2.0 CSV files
eICU_path = '../medical_data/eicu/physionet.org/files/eicu-crd/2.0/'
# path to processed DataFrame (combination of patient.csv, diagnosis.csv, medication.csv)
mydata_path = '../mydata/mydataframe_20k.csv'
# path to encoded DataFrame (one-hot encoding, redundant features dropped)
mydata_path_processed = '../mydata/mydataframe_processed_20k.csv'

# # loads orignal CSV files and builds combined DataFrame. uncomment if not yet available
# eICU_DataLoader(eICU_path, mydata_path)
# # loads combined DataFrame and builds encoded, useable DataFrame. uncomment if not yet available
# DataProcessor(mydata_path, mydata_path_processed)


eICU_data = DataManager(mydata_path_processed, ['unit_discharge_offset'])
Regressor(eICU_data, 'unit_discharge_offset')

eICU_data = DataManager(mydata_path_processed, ['hospital_discharge_offset'])
Regressor(eICU_data, 'hospital_discharge_offset')


eICU_data = DataManager(mydata_path_processed, ['unit_discharge_location_Death'])
Embedding(eICU_data)
Classifier(eICU_data, 'unit_discharge_location_Death')

eICU_data = DataManager(mydata_path_processed, ['hospital_discharge_status_Alive'])
Classifier(eICU_data, 'hospital_discharge_status_Alive')

eICU_data = DataManager(mydata_path_processed, ['hospital_discharge_status_Expired'])
Classifier(eICU_data, 'hospital_discharge_status_Expired')


