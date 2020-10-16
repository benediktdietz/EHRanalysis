# EHRanalysis

Used/ important scripts:

data_management_new.py (data_management.py is the older version of the same thing)
		Used to take care of the entire data processing from the original .csv files to the fully processed X,Y data matrices for training
		Consists of several main classes:

		eICU_DataLoader
			Uses the original .csv files (currently patient.csv, medication.csv, diagnosis.csv, pastHistory.csv, lab.csv, apacheApsVar.csv)
			and builds one dataframe that contains all of the data we want to extract but needs a lot of processing and cleaning
			The consolidated dataframe is saved to memory.

		DataProcessor
			Loads the previously built conslidated dataframe and takes care of all of the processing to get to a useable feature set.
			medication.csv, diagnosis.csv and lab.csv are integrated according to an offset threshold, such that only entries with
			offset < offset_threshold are included.
			While processing, the script goes through each hospital for each of these tables and saves respective outputs to memory.
			This has made the runtime much more manageable and the preprocessing more transparent, since the intermediate steps can be 
			easily inspected, however it does produce a considerable amount of files if running on the entire set, just FYI.
			Finally, a fully processed and useable feature matrix is saved to memory which will be used further

		DataManager
			This class loads the previously processed and loaded data and prepares it for training.
			It has implemented two sampling strategies (as discussed in the meetings), normalises the features, chooses the n most important
			features through a RandomForest permutation analysis and has funmctions built in to feed the data to the model.

		ICD10code_transformer
			This class takes the pre-loaded files and sorts the ICD10 codes into their respective parts.
			(As discussed with Annika)

		All other functions and classes in data_management should be helper functions and similar.
		Let me know if there are questions regarding these.

ehr_analysis.py
		Used to manage the preprocessing steps implemented in data_management_new.py
		Originally implemented to be the 'management' script for everything
	
data_stats.py
		A script to get various plots and statistics from the eICU dataset.
		(Mostly discussed with Annika)
	
benchmark.py
		The script I used to generate all of the 'benchmark model' results.
		For the most part similarly structures as the scripts Arash added.
	
federated_utils.py
		Contains the definition of the torch MLP models
	
fedlearn_exchange.py/ fedlearn_aggregated.py / federated_learning.py (/my_fl.py)
		The scripts implementing the FL setting to train the models
		The old version can be found in FLnetwork.py


	
