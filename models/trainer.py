import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import preprocessing

class Trainer:

	def __init__(self, features, pred):
		self.features = features
		self.pred = pred

	def train_rf(self, n_estimators = 200, param_grid = None, folders = 5,score = 'mean_absolute_error'):
		# If customized parameters are not passsed, create standard ones
		if not param_grid:
			param_grid = {
			'max_features': ['auto', 'sqrt', 'log2'],
    		'min_samples_split':[2, 4, 6, 8],
    		'min_samples_leaf' : [1, 2, 4]
			}
		# Create estimator object for the grid search
		clf = RandomForestRegressor(n_estimators=n_estimators,
		                            max_depth=None,
		                            min_samples_split=4,
		                            min_samples_leaf=1,
		                            min_weight_fraction_leaf=0.0,
		                            max_features='log2',
		                            max_leaf_nodes=None,
		                            n_jobs=-1)

		# Create Grid Search Instance. It creates 10 models with randomly choosen parameters
		CV_clf = RandomizedSearchCV(estimator=clf,
									param_distributions = param_grid,
									n_iter = 10,
									cv= folders,
									scoring= score)

		# Fit the models
		CV_clf.fit(self.features, self.pred)

		# Print results
		print("Grid scores on development set:")
		print "best score: {}".format(CV_clf.best_score_)
		return CV_clf.best_estimator_
