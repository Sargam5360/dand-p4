#!/usr/bin/python

import sys, pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
features_list = ['poi', 
				 'exercised_stock_options', 'total_stock_value', # Top 2
				 'bonus', 'salary', # Top 4
				 'deferred_income', 'long_term_incentive', # Top 6
				 'restricted_stock', # Top 7
				 # 'total_payments', # Top 8
				 # 'shared_receipt_with_poi', 'loan_advances', # Top 10
         # 'total_income', 'total_incentive', # new 2 features
				 ]

### Count valid values
meaningful_features_num = {} 
print "Total Number : {0}".format(len(data_dict.keys()))
from pprint import pprint
for person, features in data_dict.iteritems():
    for k, v in features.iteritems():
        if v != "NaN":
            try:
                meaningful_features_num[k] += 1
            except Exception, e:
                meaningful_features_num[k] = 1

# pprint(sorted(meaningful_features_num.items(), key=lambda x: x[1], reverse=True))

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('LOCKHART EUGENE E') # salary is NaN
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)
all_features_list = [ # financial
                    'poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 
                    'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                    'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                    'restricted_stock', 'director_fees',
                    # email
                    'to_messages', 'from_poi_to_this_person', 'from_messages', 
                    'from_this_person_to_poi', 'shared_receipt_with_poi'
                    ]

k = len(all_features_list) - 1
data = featureFormat(data_dict, all_features_list)
labels, features = targetFeatureSplit(data)

k_best = SelectKBest(k=k)
k_best.fit(features, labels)

unsorted_pair_list = zip(all_features_list[1:], k_best.scores_)
sorted_pair_list = sorted(unsorted_pair_list, key=lambda x: x[1], reverse=True)
# pprint([pair for pair in sorted_pair_list])

# new feature : total_income
fields = ['salary', 'total_stock_value']
for record in data_dict:
   person = data_dict[record]
   is_valid = True
   for field in fields:
       if person[field] == 'NaN':
           is_valid = False
   if is_valid:
       person['total_income'] = sum([person[field] for field in fields])
   else:
       person['total_income'] = 'NaN'
all_features_list += ['total_income']

# new feature : total_income
fields = ['bonus', 'long_term_incentive']
for record in data_dict:
   person = data_dict[record]
   is_valid = True
   for field in fields:
       if person[field] == 'NaN':
           is_valid = False
   if is_valid:
       person['total_incentive'] = sum([person[field] for field in fields])
   else:
       person['total_incentive'] = 'NaN'
all_features_list += ['total_incentive']

### Store to my_dataset for easy export below.
my_dataset = data_dict

# gnb_clf = GaussianNB()
# for k in range(2,len(features_list)):
# 	test_classifier(gnb_clf, my_dataset, features_list[0:k], folds=1000)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
RANDOM_STATE = 87

dt_clf = DecisionTreeClassifier(random_state=42)
test_classifier(dt_clf, my_dataset, features_list, folds=1000)

# LogisticRegression
lr_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(tol=0.001, random_state=RANDOM_STATE))
])

test_classifier(lr_pipeline, my_dataset, features_list, folds=1000)

# Gaussian NB 
gnb_clf = GaussianNB()
test_classifier(gnb_clf, my_dataset, features_list, folds=1000)

# Gaussian NB + PCA
gnb_pipeline = Pipeline(steps=[
        ('pca', PCA()),
        ('clf', GaussianNB())
])
test_classifier(gnb_pipeline, my_dataset, features_list, folds=1000)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.metrics import make_scorer, precision_recall_fscore_support
# I want to classifier whose (precision, recall) are high at the same time.
def my_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
	p, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
	                                                 labels=labels,
	                                                 pos_label=pos_label,
	                                                 average=average,
	                                                 sample_weight=sample_weight)
	if p < 0.3 or r < 0.3: # To achieve better than 0.3 precision and recall
		return 0.
	return ( p + r ) / 2. # Normalize mix score

# PCA + GaussianNB
gnb_pipeline_parameters = {
            'pca__n_components' : [None] + range(3,8),
            # 'pca__whiten' : [False, True]
            }
clf = GridSearchCV(gnb_pipeline, gnb_pipeline_parameters, scoring=make_scorer(my_score))
clf.fit(features, labels)


gnb_clf =  clf.best_estimator_

test_classifier(gnb_clf, my_dataset, features_list, folds=1000)

# LogisticRegression
lr_pipeline_parameters = {
						'clf__C' : 10.0 ** np.arange(-12, 15, 2),
						'clf__tol' : 10.0 ** np.arange(-12, 15, 2),
						'clf__penalty' : ('l1', 'l2'),
						}
lr_clf = GridSearchCV(lr_pipeline, lr_pipeline_parameters, scoring=make_scorer(my_score))
lr_clf.fit(features, labels)

test_classifier(lr_clf.best_estimator_, my_dataset, features_list, folds=1000)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

clf = lr_clf.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)