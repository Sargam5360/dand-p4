Identify Fraud from Enron Email
========================================================
by Yoon-gu Hwang, November 15, 2015

## Overview ##
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

## QnA ##
### Question 1 ###
**Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**

The goal of this project is to use the email and financial data to identify fraud, **person of interest**(POI).
The dataset has total 146 people information including their name. Each person information data contains 21 features(poi label + 14 financial + 6 email).
There are 18 POIs and 128 non-POIs.

Looking carefully the dataset, I found 3 outliers as followings and deleted them.
 - `TOTAL` : Just total information of dataset
 - `LOCKHART EUGENE E` : No information : All feature are 'NaN'.
 - `THE TRAVEL AGENCY IN THE PARK` : Not a person

|name|# of features|
|:---|----:|
|poi| 146|
|total_stock_value| 126|
|total_payments| 125|
|email_address| 111|
|restricted_stock| 110|
|exercised_stock_options| 102|
|salary| 95|
|expenses| 95|
|other| 93|
|to_messages| 86|
|shared_receipt_with_poi| 86|
|from_messages| 86|
|from_this_person_to_poi| 86|
|from_poi_to_this_person| 86|
|bonus| 82|
|long_term_incentive| 66|
|deferred_income| 49|
|deferral_payments| 39|
|restricted_stock_deferred| 18|
|director_fees| 17|
|loan_advances| 4|

### Question 2 ###
**What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]**

With `SelectKBest` in `sklearn`, I investigate scores of features and I attached the table.

|name|score|
|:---|----:|
|exercised_stock_options| 24.541175342601967|
|total_stock_value| 23.904270290793864|
|bonus| 20.524645181851792|
|salary| 18.003739993113935|
|deferred_income| 11.321486775141238|
|long_term_incentive| 9.7721035384082544|
|restricted_stock| 9.0790766616708698|
|total_payments| 8.6727512066131069|
|shared_receipt_with_poi| 8.4326354230246814|
|loan_advances| 7.1253824688830685|
|expenses| 5.9545442921972933|
|from_poi_to_this_person| 5.1422191945069704|
|other| 4.1288734042047182|
|from_this_person_to_poi| 2.3388361146462624|
|director_fees| 2.1453342495720547|
|to_messages| 1.5942560277180795|
|deferral_payments| 0.23026270434011689|
|from_messages| 0.1753832041587958|
|restricted_stock_deferred| 0.066023245366887376|

I did not choose features whose score is below 2.00, i.e., `to_messages`, `deferral_payments`, `from_messages`, `restricted_stock_deferred`.

Also, I created 2 new features, `total_incentive` and `total_income`.

- `total_incentive` : `bonus` + `long_term_incentive`
- `total_income` : `salary` + `total_stock_value`

As the following table shows, with increasing features upto 7, overall evaluating metrics are higher. However, after 7, recall metric suddenly drop, so I choose first 7 features for my classifier. To make this table, I used very simple `GaussianNB` classifer.

| # of features | accuracy | precision| recall |
|:-------------:|---------:|----------:|-------:|
| 1 			| 0.82909  | 0.56608   | 0.25700|
| 2 			| 0.83962  | 0.46275   | 0.26400|
| 3 			| 0.84077  | 0.47559   | 0.34100|
| 4 			| 0.85185  | 0.52782   | 0.35100|
| 5 			| 0.85636  | 0.49639   | 0.37800|
| 6 			| 0.84979  | 0.46680   | 0.36200|
| 7 			| 0.85021  | 0.47004   | 0.38050|
| 8 			| 0.84040  | 0.37356   | 0.29100|
| 9 			| 0.83580  | 0.35719   | 0.28950|

|# of features  				| accuracy|precision|recall |
|-------------------------------|---------|---------|-------|
|Top 7 features 				| 0.85021 |0.47004	|0.38050|
|Top 7 features + 2 new features| 0.84671 |0.45866	|0.40500|

By adding 2 new features, there is trade-off. I got higher recall value, but accuracy and precision became lower than before adding 2 new features.

Each feature has variety range of values, so I need to normalize them. I used `StandardScaler` for fianl analysis by adding it to `Pipeline`.

When I used `DecisionTree`, its importance analysis is shown below.

|name|importance|
|:----|----:|
|total_payments| 0.191|
|total_stock_value| 0.167|
|expenses| 0.140|
|other| 0.139|
|from_messages| 0.127|
|bonus| 0.095|
|from_this_person_to_poi| 0.076|
|salary| 0.059|
|exercised_stock_options| 0.000|
|deferred_income| 0.000|
|long_term_incentive| 0.000|
|restricted_stock| 0.000|
|shared_receipt_with_poi| 0.000|
|loan_advances| 0.000|
|from_poi_to_this_person| 0.000|
|director_fees| 0.000|
|to_messages| 0.000|
|deferral_payments| 0.000|
|restricted_stock_deferred| 0.000|

### Question 3 ###
**What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**

I treid 4 algorithms to achieve maximum identification performance. The list of them shows below.

| algorithm								| accuracy | precision| recall |
|--------------------------------------:|:--------:|:---------:|:------:|
| DecisionTreeClassifier				| 0.81293  | 0.29205   | 0.28300|
| StandardScaler + LogisticRegression	| 0.85360  | 0.34295   | 0.10700|
| GaussianNB							| 0.81927  | 0.31280   | 0.29700|
| PCA + GaussianNB						| 0.84467  | 0.37705   | 0.25300|

DecisionTree, GaussianNB, and PCA are covered in the class. So I tried to apply and check their performance. It was suprising that the simple `GaussianNB` had great performance without any trials and erros. 

Logistic Regression is useful to binary classification(True/False) variables. This scheme is exactly what we want in this project. We have binary classification, POI or non-POI. So I ended up using this logistic regression algorithm.

### Question 4 ###
**What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). [relevant rubric item: “tune the algorithm”]**

Tuning parameters of an algorithm means the way to get the best algorithm performance with given dataset and model. If we don't do this process, we cannot get good performance and we might have worse performance than what we expected.

In this project, I trid to tune several parameter sets with 2 algorithms(`LogisticRegression` and `PCA+GaussianNB`). `GaussianNB` had no parameter to be tuned, so I made a pipeline(`PCA` + `GaussianNB`) and tuned parameter of `PCA`.

* `LogisticRegression`
   * `penalty` : Used to specify the norm used in the penalization. The newton-cg and lbfgs solvers support only l2 penalties.
   * `C` : Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
   * `tol` : Tolerance for stopping criteria.
* `PCA`
   * `n_components` : Number of components to keep.

Final parameter for me to be tuned. I can avoid tedious trails by using `GridSearchCV`.
* `LogisticRegression`
   * `penalty` : l2
   * `C` : 1e-12
   * `tol` : 1e-12
* `PCA`
   * `n_components` : None

### Question 5 ###
**What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]**

Validation is the way to confirm the robustness of a classifier with given dataset and model. The class mistake is the over-fitting case. When a classifier is over-fitted, it cannot provide good performance on test dataset. Because the classifier is too much over-fitted to training dataset. We should consider balance between training data and test data.

To avoid this problem, I extracted 10% of dataset for test set and the rest of dataset was used as training set. To split test set from given dataset, I used `StratifiedShuffleSplit`.

I set a `RANDOM_STATE` to 87. I took 1000 times split and fit the traing dataset. This scheme is also used in `test_classifier()` in `tester.py`.

### Question 6 ###
**Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**

I selected precision and recall as evaluation metrics. The definitions are followed.

* precision : the number of true positives over the number of true positives plus the number of false positives.
* recall : the number of true positives over the number of true positives plus the number of false negatives.

In short, high precision is equivalent to low false alram, that means an algorithm is precise. High recall means that high true positive, that is, an algorithm can identify POI as many as possibile.

| algorithm								| accuracy | precision| recall |
|--------------------------------------:|:--------:|:---------:|:------:|
| StandardScaler + LogisticRegression	| 0.85157  | 0.47869   | 0.43800|
| PCA + GaussianNB						| 0.84893  | 0.46169   | 0.34650|

I made my own score function, averaging precision and recall, then I passed it to `GridSearchCV`. So I could get the optimal parameter for the above 2 algorithms. 

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

The result of `GaussianNB` was awesome, because it had quite good performance even though I did nothing. But `LogisticRegression` is the best performance among trials I did. It had high value of precision and recall. This evaluation result is more balanced and higher values.

## Conclusion ##

Final classifier is logistic regression since its performance is the best among many trials. In addition, its algorithm is specialized to binary classification. This is perfect story to identify fraud from Enron dataset. 

## References ##
 * http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
 * http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
 * http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#example-model-selection-grid-search-text-feature-extraction-py
 * http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
 * http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
 * http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html