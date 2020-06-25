import sys, os, time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc, classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler, RobustScaler

OUTPATH = '../results/test_new/'
print('\nwriting output results to ' + OUTPATH + '\n\n')

class Embedding():

	def __init__(self, dataset, figure_path = OUTPATH, plots = True):

		self.dataset = dataset
		self.figure_path = figure_path
		self.targets = [
			'gender_Female', 
			'unit_discharge_location_Death', 
			'hospital_discharge_status_Alive', 
			'hospital_discharge_status_Expired']
		self.targets_cont = [
			'unit_discharge_offset',
			'hospital_discharge_offset']


		self.lle = self.compute_lle()
		self.make_figures(self.lle, 'LocallyLinearEmbedding')

		self.spectral = self.compute_spectral_embedding()
		self.make_figures(self.spectral, 'SpectralEmbedding')

		self.tsne = self.compute_tsne()
		self.make_figures(self.tsne, 'TSNE')


	def compute_tsne(self):

		start_time = time.time()

		tsne_vec = TSNE(
			n_components=2, 
			perplexity=30.0, 
			early_exaggeration=10.0, 
			learning_rate=100.0, 
			n_iter=10000).fit_transform(self.dataset.training_data['x_full'])

		end_time = time.time()
		print('\nfitted tsne in ', str(np.round(abs(end_time - start_time), 2)), 'sec')

		return tsne_vec

	def compute_lle(self):

		start_time = time.time()

		lle_vec = LocallyLinearEmbedding(
			n_neighbors=3, 
			n_components=2,
			max_iter=10000,
			n_jobs=-1
			).fit_transform(self.dataset.training_data['x_full'])

		end_time = time.time()
		print('\nfitted LocallyLinearEmbedding in ', str(np.round(abs(end_time - start_time), 2)), 'sec')

		return lle_vec

	def compute_spectral_embedding(self):

		start_time = time.time()

		specemb_vec = SpectralEmbedding(
			n_components=2, 
			affinity='nearest_neighbors', 
			n_neighbors=3, 
			n_jobs=-1).fit_transform(self.dataset.training_data['x_full'])

		end_time = time.time()
		print('\nfitted SpectralEmbedding in ', str(np.round(abs(end_time - start_time), 2)), 'sec')

		return specemb_vec

	def make_figures(self, embedding, embedding_name):

		try: os.makedirs(self.figure_path + embedding_name + '/')
		except FileExistsError: pass

		for target in self.targets:

			plt.figure()
			plt.title(target)
			plt.scatter(
				embedding[self.dataset.data_df[target] == 0, 0], 
				embedding[self.dataset.data_df[target] == 0, 1], 
				c='b', s=4, alpha=.5, label='no ' + target)
			plt.scatter(
				embedding[self.dataset.data_df[target] == 1, 0], 
				embedding[self.dataset.data_df[target] == 1, 1], 
				c='r', s=4, alpha=.5, label=target)
			plt.legend()
			plt.grid()
			plt.savefig(self.figure_path + embedding_name + '/' + target + '.pdf')
			plt.close()

		for target in self.targets_cont:

			plt.figure()
			plt.title(target)
			plt.scatter(
				embedding[:,0], 
				embedding[:,1], 
				c=self.dataset.labels[target], 
				cmap= 'jet', s=4, alpha=.5, label=target)
			plt.clim(0., 40000.)
			plt.colorbar()
			plt.legend()
			plt.grid()
			plt.savefig(self.figure_path + embedding_name + '/' + target + '.pdf')
			plt.close()

class Classifier():

	def __init__(self, dataset, target, figure_path = OUTPATH, plots = True):

		self.dataset = dataset
		self.figure_path = figure_path
		self.target = target


		self.sgd_c = SGDClassifier(
			loss='hinge',
			penalty='l2', 
			alpha=0.0001, 
			l1_ratio=0.15, 
			fit_intercept=True, 
			max_iter=2000)
		self.random_forest = RandomForestClassifier(
			n_estimators=512, 
			criterion='gini', 
			max_features='auto', 
			n_jobs=-1)
		self.logistic_regression = LogisticRegression(max_iter=4000)
		self.ada_boost = AdaBoostClassifier(n_estimators=512)
		self.gradient_boost = GradientBoostingClassifier(
			n_estimators=512, 
			tol=1e-5)
		self.knn_classifer = KNeighborsClassifier(
			n_neighbors=3,
			weights='distance',
			algorithm='brute',
			leaf_size=20,
			n_jobs=-1)
		self.mlp_classifier = MLPClassifier(
			hidden_layer_sizes=(512, 256), 
			activation='tanh', 
			learning_rate='adaptive',
			solver='adam',
			max_iter=1000,
			tol=1e-5,
			verbose=False,
			n_iter_no_change=50,
			early_stopping=True,
			validation_fraction=.3
			)

		self.models = [
			# self.sgd_c,
			self.random_forest, 
			self.logistic_regression,
			self.ada_boost,
			self.gradient_boost,
			# self.knn_classifer,
			# self.mlp_classifier,
			]

		self.model_names = [
			# 'SGD Classifier',
			'Random Forest',
			'Logistic Regression',
			'AdaBoost',
			'GradientBoost',
			# 'K Nearest Neighbors',
			# 'MLP Classifier',
			]


		self.fit_models()
		self.roc_analysis()


	def fit_models(self):

		print('\nfitting models to data ...')
		for i in range(len(self.models)):

			start_time = time.time()

			self.models[i].fit(self.dataset.training_data['x_train'], np.ravel(np.reshape(np.asarray(self.dataset.training_data['y_train'][self.target].values), (-1,1))))

			end_time = time.time()
			print('fitted', self.model_names[i], 'in '.ljust(30 - len(self.model_names[i]), '.'), str(np.round(abs(end_time - start_time), 2)), 'sec')
		print('\n')

	def roc_analysis(self, set_recall = .9):

		plt.figure()
		plt.title('Classifier ROC ' + self.target)

		roc_df = []

		y_true = np.abs(self.dataset.training_data['y_test'][self.target].values.astype(int))

		# y_true = self.dataset.training_data['y_test'].values

		for i in range(len(self.models)):


			predictions = self.models[i].predict_proba(self.dataset.training_data['x_test'])
			predictions = np.reshape(np.asarray(predictions[:,1]), (-1,1))


			fp_rate, tp_rate, thresholds = roc_curve(y_true, predictions)

			roc_auc = auc(fp_rate, tp_rate)

			plt.plot(fp_rate, tp_rate, label = self.model_names[i] + '   auroc: ' + str(np.round(roc_auc, 2)))


			roc_dummy = pd.DataFrame({
				'fp_rate': fp_rate,
				'tp_rate': tp_rate,
				'threshold': thresholds,
				'tp_dummy': tp_rate,
				})

			roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] > set_recall] = 0.
			roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] < roc_dummy['tp_dummy'].max()] = 0.
			roc_dummy['tp_dummy'] /= roc_dummy['tp_dummy'].max()
			roc_dummy = np.asarray(roc_dummy['threshold'].loc[roc_dummy['tp_dummy'] > .5].values)
			
			if roc_dummy.shape[0] > 1: roc_dummy = roc_dummy[0]

			recall_threshold = roc_dummy


			predictions_binary = predictions
			predictions_binary[predictions_binary >= recall_threshold] = 1.
			predictions_binary[predictions_binary < recall_threshold] = 0.

			num_true_positives = np.sum(np.abs(predictions_binary) * np.abs(y_true))
			num_false_positives = np.sum(np.abs(predictions_binary) * np.abs(1. - y_true))
			num_true_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(1. - y_true))
			num_false_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(y_true))

			num_total_positives = num_true_positives + num_false_negatives
			num_total_negatives = num_true_negatives + num_false_positives

			num_total_positives_predicted = np.sum(np.abs(predictions_binary))

			recall = num_true_positives / num_total_positives
			selectivity = num_true_negatives / num_total_negatives
			precision = num_true_positives / (num_true_positives + num_false_positives)
			accuracy = (num_true_positives + num_true_negatives) / (num_total_positives + num_total_negatives)
			f1score = (2 * num_true_positives) / (2 * num_true_positives + num_false_positives + num_false_negatives)
			informedness = recall + selectivity - 1.

			roc_df.append({
				'model': self.model_names[i],
				'auroc': roc_auc,
				'recall': recall,
				'selectivity': selectivity,
				'precision': precision,
				'accuracy': accuracy,
				'f1score': f1score,
				'informedness': informedness,
				'#TP': num_true_positives,
				'#FP': num_false_positives,
				'#TN': num_true_negatives,
				'#FN': num_false_negatives,
				})


		roc_df = pd.DataFrame(roc_df)
		roc_df.set_index('model')
		print('\nclassification results on validation set for recall approx.', \
			set_recall, ' taget: ', self.target, '\n', roc_df.round(2).sort_values('auroc', ascending=False), '\n')

		
		plt.plot([0,1],[0,1],'r--')
		plt.legend(loc='lower right')
		plt.xlim([0., 1.])
		plt.ylim([0., 1.])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.grid()
		plt.savefig(self.figure_path + 'classification_roc_' + self.target +'.pdf')
		plt.close()

class Regressor():

	def __init__(self, dataset, target, figure_path = OUTPATH, plots = True):

		self.dataset = dataset
		self.figure_path = figure_path
		self.target = target
	

		self.sgd_r = SGDRegressor(
			loss='squared_loss',
			penalty='l2', 
			alpha=0.0001, 
			l1_ratio=0.15, 
			fit_intercept=True, 
			max_iter=10000)
		self.random_forest = RandomForestRegressor(
			n_estimators=1024, 
			criterion='mse', 
			max_features='auto', 
			n_jobs=-1)
		self.decisiton_tree = DecisionTreeRegressor(criterion='mse')
		self.ada_boost = AdaBoostRegressor(self.decisiton_tree, n_estimators=1024)
		self.gradient_boost = GradientBoostingRegressor(
			n_estimators=1024, 
			tol=1e-5)
		self.knn_regressor = KNeighborsRegressor(
			n_neighbors=5,
			weights='uniform',
			algorithm='brute',
			leaf_size=15,
			n_jobs=-1)
		self.mlp_regressor = MLPRegressor(
			hidden_layer_sizes=(1024, 512), 
			activation='relu', 
			solver='adam',
			learning_rate='adaptive',
			max_iter=1000,
			tol=1e-5,
			verbose=False,
			n_iter_no_change=50,
			early_stopping=True,
			validation_fraction=.3
			)

		self.models = [
			# self.sgd_r,
			self.random_forest, 
			self.decisiton_tree,
			# self.ada_boost,
			# self.gradient_boost,
			self.knn_regressor,
			# self.mlp_regressor,
			]

		self.model_names = [
			# 'SGD Regressor',
			'Random Forest',
			'Decision Trees',
			# 'AdaBoost',
			# 'GradientBoost',
			'K Nearest Neighbors',
			# 'MLP Regressor',
			]


		self.y_all = np.reshape(np.abs(np.asarray(self.dataset.training_data['y_full'][self.target].values.astype(float))), (-1,1))
		self.scaler = StandardScaler().fit(self.y_all)
		# self.y_train = np.ravel(self.scaler.transform(np.reshape(np.asarray(self.dataset.training_data['y_train'][self.target].values), (-1,1))))
		self.y_train = np.ravel(self.scaler.transform(np.reshape(np.abs(np.asarray(self.dataset.training_data['y_train'][self.target].values.astype(float))), (-1,1))))


		self.fit_models()
		self.validate_models()


	def fit_models(self):

		print('\nfitting models to data ...')
		for i in range(len(self.models)):

			start_time = time.time()

			self.models[i].fit(self.dataset.training_data['x_train'], self.y_train)

			end_time = time.time()
			print('fitted', self.model_names[i], 'in '.ljust(30 - len(self.model_names[i]), '.'), str(np.round(abs(end_time - start_time), 2)), 'sec')
		print('\n')

	def validate_models(self):

		try: os.makedirs(self.figure_path + 'regression/')
		except FileExistsError: pass
		try: os.makedirs(self.figure_path + 'regression/' + self.target + '/')
		except FileExistsError: pass

		regression_df = []

		# y_true = np.abs(self.dataset.training_data['y_test'][self.target].values.astype(float))
		y_true = np.reshape(np.abs(np.asarray(self.dataset.training_data['y_test'][self.target].values.astype(float))), (-1,1))

		for i in range(len(self.models)):


			predictions = self.models[i].predict(self.dataset.training_data['x_test'])
			predictions = self.scaler.inverse_transform(np.ravel(np.asarray(predictions)))


			mse = mean_squared_error(y_true, predictions)
			r2 = r2_score(y_true, predictions)
			mae = mean_absolute_error(y_true, predictions)
			error_var = np.var(np.abs(y_true - predictions))

			percentage_errors = np.abs((y_true - predictions) / (y_true + 1e-4))
			precentage_mean_error = np.mean(percentage_errors)
			precentage_median_error = np.median(percentage_errors)

			regression_df.append({
				'model': self.model_names[i],
				'mean_squared_error': mse,
				'mean_absolute_error': mae,
				'mean_percentage_error': precentage_mean_error,
				'precentage_median_error': precentage_median_error,
				'r2_score': r2,
				'error_var': error_var,
				})


			# predictions -= np.amin(predictions)
			# predictions /= np.amax(predictions)

			# y_true_plot = np.asarray(((y_true - np.amin(y_true))/np.amax(y_true - np.amin(y_true))))


			plt.figure(figsize=(12,12))
			plt.title(self.model_names[i] + ' | ' + self.target)
			plt.scatter(predictions/24, y_true/24, s=3)
			plt.xlabel('predictions [days]')
			plt.ylabel('labels [days]')
			if self.target == 'length_of_icu':
				plt.xlim(0., 20.)
				plt.ylim(0., 20.)
			if self.target == 'length_of_stay':
				plt.xlim(0., 60.)
				plt.ylim(0., 60.)
			# plt.yscale('log')
			# plt.xscale('log')
			plt.grid()
			plt.savefig(self.figure_path + 'regression/' + self.target + '/' + self.model_names[i] + '_' + self.target + '.pdf')


		regression_df = pd.DataFrame(regression_df)
		regression_df.set_index('model')
		print('\nregression results on validation set for ' + self.target + ':\n', regression_df.round(2), '\n')
