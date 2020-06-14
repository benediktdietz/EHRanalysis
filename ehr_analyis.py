import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_loading import load_data, process_data

# path to the eICU CRD 2.0 CSV files
eICU_path = '../medical_data/eicu/physionet.org/files/eicu-crd/2.0/'
# path to processed DataFrame (combination of patient.csv, diagnosis.csv, medication.csv)
mydata_path = '../mydata/mydataframe2k.csv'
# path to encoded DataFrame (one-hot encoding, redundant features dropped)
mydata_path_processed = '../mydata/mydataframe2k_processed.csv'

# loads orignal CSV files and builds combined DataFrame. uncomment if not yet available
load_data(eICU_path, mydata_path)
# loads combined DataFrame and builds encoded, useable DataFrame. uncomment if not yet available
process_data(mydata_path)



eICU_df = pd.read_csv(mydata_path_processed).drop(columns='Unnamed: 0')

print(eICU_df)
print('\n**************\n')
print(eICU_df.keys())