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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler

OUTPATH = '../results/test/'
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


		self.tsne = self.compute_tsne()
		self.make_figures()


	def compute_tsne(self):

		start_time = time.time()

		tsne_vec = TSNE(
			n_components=2, 
			perplexity=30.0, 
			early_exaggeration=10.0, 
			learning_rate=100.0, 
			n_iter=10000).fit_transform(self.dataset.training_data['x_full'])

		end_time = time.time()
		print('\nfitted tsne in ', str(np.round(abs(end_time - start_time), 2)), 'sec\n')

		return tsne_vec

	def make_figures(self):

		try: os.makedirs(self.figure_path + 'tsne/')
		except FileExistsError: pass

		for target in self.targets:

			plt.figure()
			plt.title(target)
			plt.scatter(
				self.tsne[self.dataset.data_df[target] == 0, 0], 
				self.tsne[self.dataset.data_df[target] == 0, 1], 
				c='b', s=2, label='no ' + target)
			plt.scatter(
				self.tsne[self.dataset.data_df[target] == 1, 0], 
				self.tsne[self.dataset.data_df[target] == 1, 1], 
				c='r', s=2, label=target)
			plt.legend()
			plt.grid()
			plt.savefig(self.figure_path + 'tsne/' + target + '.pdf')
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
			max_iter=1000)
		self.random_forest = RandomForestClassifier(
			n_estimators=512, 
			criterion='gini', 
			max_features='auto', 
			n_jobs=-1)
		self.logistic_regression = LogisticRegression(max_iter=4000)
		self.ada_boost = AdaBoostClassifier(n_estimators=256)
		self.gradient_boost = GradientBoostingClassifier(
			n_estimators=256, 
			tol=1e-5)
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
			self.mlp_classifier,
			]

		self.model_names = [
			# 'SGD Classifier',
			'Random Forest',
			'Logistic Regression',
			'AdaBoost ',
			'GradientBoost ',
			'MLP Classifier',
			]


		self.fit_models()
		self.roc_analysis()


	def fit_models(self):

		print('\nfitting models to data ...')
		for i in range(len(self.models)):

			start_time = time.time()

			self.models[i].fit(self.dataset.training_data['x_train'], np.ravel(np.reshape(np.asarray(self.dataset.training_data['y_train'].values), (-1,1))))

			end_time = time.time()
			print('fitted', self.model_names[i], 'in '.ljust(30 - len(self.model_names[i]), '.'), str(np.round(abs(end_time - start_time), 2)), 'sec')
		print('\n\n\n')

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

			num_total_positives = np.sum(np.abs(y_true))
			num_total_negatives = np.sum(np.abs(1. - y_true))

			num_total_positives_predicted = np.sum(np.abs(predictions_binary))

			recall = num_true_positives / num_total_positives
			selectivity = num_true_negatives / num_total_negatives
			precision = num_true_positives / (num_true_positives + num_false_positives)
			accuracy = (num_true_positives + num_true_negatives) / len(y_true)
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
		print('\n\n\nclassification results on validation set for recall approx.', \
			set_recall, ' taget: ', self.target, '\n', roc_df.round(2).sort_values('auroc', ascending=False), '\n\n\n')

		
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
			n_estimators=512, 
			criterion='mse', 
			max_features='auto', 
			n_jobs=-1)
		self.decisiton_tree = DecisionTreeRegressor(criterion='mse')
		self.ada_boost = AdaBoostRegressor(self.decisiton_tree, n_estimators=256)
		self.gradient_boost = GradientBoostingRegressor(
			n_estimators=256, 
			tol=1e-5)
		self.mlp_regressor = MLPRegressor(
			hidden_layer_sizes=(512, 256), 
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
			self.ada_boost,
			self.gradient_boost,
			self.mlp_regressor,
			]

		self.model_names = [
			# 'SGD Regressor',
			'Random Forest',
			'Decision Trees',
			'AdaBoost ',
			'GradientBoost ',
			'MLP Regressor',
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
		print('\n\n\n')

	def validate_models(self):

		try: os.makedirs(self.figure_path + 'regression/')
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
			precentage_mean_error = np.mean(np.abs(1. - np.abs(y_true - predictions) / np.abs(y_true)))
			precentage_median_error = np.median(np.abs(1. - np.abs(y_true - predictions) / np.abs(y_true)))

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
			plt.scatter(predictions, y_true, s=3)
			plt.xlabel('predictions')
			plt.ylabel('labels')
			if self.target in ['unit_discharge_offset', 'hospital_discharge_offset']:
				plt.xlim(0., 40000.)
				plt.ylim(0., 40000.)
			# plt.yscale('log')
			# plt.xscale('log')
			plt.grid()
			plt.savefig(self.figure_path + 'regression/' + self.model_names[i] + '_' + self.target + '.pdf')


		regression_df = pd.DataFrame(regression_df)
		regression_df.set_index('model')
		print('\n\n\nregression results on validation set for ' + self.target + ':\n', regression_df.round(2), '\n\n\n')
