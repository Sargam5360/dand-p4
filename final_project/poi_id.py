#!/usr/bin/python

import sys, pickle
sys.path.append("../tools/")

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

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

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

# pprint(sorted(meaningful_features_num.items(), key=lambda x: -x[1]))

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('FUGH JOHN L') #salary is NaN
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)
all_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 
					'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
					'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
					'restricted_stock', 'director_fees']

k = 10
data = featureFormat(data_dict, all_features_list)
labels, features = targetFeatureSplit(data)

k_best = SelectKBest(k=k)
k_best.fit(features, labels)

unsorted_pair_list = zip(all_features_list[1:], k_best.scores_)
sorted_pair_list = sorted(unsorted_pair_list, key=lambda x: x[1], reverse=True)
features_list = [pair[0] for pair in sorted_pair_list]

features_list = ['poi'] + features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# estimators = [('reduce_dim', PCA()), ('nb', GaussianNB())]
# clf = Pipeline(estimators)
# clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

RANDOM_STATE = 87

target_metric = 'f1'
clf_list = [
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    RandomForestClassifier(random_state=RANDOM_STATE),
    LogisticRegression(random_state=RANDOM_STATE),
    SGDClassifier(random_state=RANDOM_STATE),
 #    Pipeline(estimators),
	# GaussianNB(),
]

params_list = [
    {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': range(2, 5),
    },
    {
        'criterion': ['entropy', 'gini'],
        'min_samples_split': range(2, 5),
    },
    {
        'penalty': ['l1', 'l2'],
        'C': [1., 100., 1000.],
    },
    {
        'loss': ['hinge', 'log'],
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    },
    # {},
    # {}
]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

for clf, params in zip(clf_list, params_list):
    clf = GridSearchCV(clf, params, scoring=target_metric)
    test_classifier(clf, my_dataset, features_list, folds=1000)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)