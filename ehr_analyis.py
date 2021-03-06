from data_management_new import eICU_DataLoader, DataProcessor, DataSetIterator, DataManager, ICD10code_transformer, DataAnalysis
from models import Embedding, Classifier, Regressor
from FLnetwork import FederatedLearner
from network import NetworkTrainer
import os, argparse, pandas

OUTPATH = '../results/7_10/'
FOLDER = 'mydata4'

parser = argparse.ArgumentParser()

parser.add_argument('--eICU_path', type=str, default='../medical_data/eicu/physionet.org/files/eicu-crd/2.0/', help='Directory path to original eICU files')
parser.add_argument('--mydata_path_files', type=str, default='../' + FOLDER +'/', help='Directory path to loaded dataframes')
parser.add_argument('--mydata_path', type=str, default='../' + FOLDER +'/loaded.csv', help='Directory path to loaded dataframe')
parser.add_argument('--mydata_path_processed', type=str, default='../' + FOLDER +'/processed_featureset.csv', help='Directory path to processed dataframe')
parser.add_argument('--datapath_federated', type=str, default='../' + FOLDER +'/federated', help='Directory path to processed individual hospital dataframes')
parser.add_argument('--diag_table_path', type=str, default='../' + FOLDER +'/diagnosis_table.csv', help='Directory path to processed individual hospital dataframes')

parser.add_argument('--num_patients_to_load', type=int, default=-1, help='Number of patients to load from original data')
parser.add_argument('--num_hospitals_to_load', type=int, default=8, help='Number of hospitals to load from original data')
parser.add_argument('--min_patients_per_hospital', type=int, default=10, help='Mininum number of patients per hospital for federated datasets')
parser.add_argument('--integrate_past_cases', type=int, default=0, help='Sum over all past + the current ICU stay if set to 1')
parser.add_argument('--big_hospitals_first', type=int, default=0, help='Sum over all past + the current ICU stay if set to 1')

parser.add_argument('--train_split', type=float, default=.7, help='Ratio of sample used for training')
parser.add_argument('--outdir', type=str, default=OUTPATH, help='Directory path to save output files. it will be created if not existent.')
parser.add_argument('--load_data', type=int, default=0, help='Loads dataframe from eICU CSV files if set to 1')
parser.add_argument('--process_data', type=int, default=0, help='processes dataframe from eICU CSV files if set to 1')
parser.add_argument('--process_diagnoses', type=int, default=0, help='processes dataframe from eICU CSV files if set to 1')
parser.add_argument('--process_analyses', type=int, default=0, help='processes dataframe from eICU CSV files if set to 1')

parser.add_argument('--loss', type=str, default='categorical_crossentropy', help='Used loss function for federated classification')
parser.add_argument('--activation', type=str, default='sigmoid', help='Used activation function')
parser.add_argument('--layer_width_0', type=int, default=2048, help='Width of the first MLP layer')
parser.add_argument('--layer_width_1', type=int, default=1024, help='Width of the second MLP layer')
parser.add_argument('--layer_width_2', type=int, default=512, help='Width of the third MLP layer')
parser.add_argument('--layer_width_3', type=int, default=256, help='Width of the fourth MLP layer')

parser.add_argument('--network_kind', type=str, default='classification', help='regression or classification')
parser.add_argument('--target_label', type=str, default='will_return', help='label for prediction')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of (local) epochs')
parser.add_argument('--num_gobal_epochs', type=int, default=1000, help='Number of (global) epochs')
parser.add_argument('--learning_rate', type=int, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--validation_freq', type=int, default=5, help='Validation frequency (number of epochs)')

parser.add_argument('--split_strategy', type=str, default='trainN_testN', help='trainNminus1_test1 / trainN_testN')

args = parser.parse_args()

arg_df = []
for arg in vars(args):
	arg_df.append([
			str(arg), getattr(args, arg)
			])
try: os.makedirs(args.outdir)
except FileExistsError: pass
pandas.DataFrame(arg_df).to_csv(args.outdir + 'args.csv')


# loads orignal CSV files and builds combined DataFrame. uncomment if not yet available
if args.load_data:
	eICU_DataLoader(args)
# loads combined DataFrame and builds encoded, useable DataFrame. uncomment if not yet available
if args.process_data:
	DataProcessor(args)

if args.process_diagnoses:
	ICD10code_transformer(args)

if args.process_analyses:
	DataAnalysis(args.mydata_path_processed[:-4] + '_1000.csv', args.mydata_path_files + 'plots/')


eICU_data = DataManager(args)


# NetworkTrainer(eICU_data, args)


# FL_network = FederatedLearner(args)
# FL_network.train()


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


