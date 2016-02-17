import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from sklearn.metrics import classification_report
from trainer import *

features = pd.read_csv('../data/w2v_features.csv')
test_features = pd.read_csv('../data/w2v_test_features.csv')

X = features[['feature_{0}'.format(index) for index in xrange(1,38)]].as_matrix()
y = features['relevance'].as_matrix()


rf_trainer = Trainer(features = X,
					pred = y,
	)

clf = Trainer.train_rf(self, n_estimators = 200, param_grid = None, folders = 5, score = 'mean_absolute_error')

print()

print("Detailed classification report:")
print()
print("The model is trained on the full training set.")
print("The scores are computed on the full training set.")
print()
y_pred = clf.predict(X)
print(classification_report(y, y_pred))
print()
print('Saving predictions for test set...')
rf_df = test_features[['id']]
rf_df['relevance'] = y_pred
rf_df.to_csv('../submissions/w2v_rf_submission', index=False)
