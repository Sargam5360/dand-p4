Identify Fraud from Enron Email
========================================================
by Yoon-gu Hwang, November 12, 2015

## Question 1 ##
**Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**

The dataset have total 146 items.

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

## Question 2 ##
**What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]**

## Question 3 ##
**What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**


## Question 5 ##
**What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]**

## Question 6 ##
**Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**


## Conclusion ##

## References ##
