Identify Fraud from Enron Email
========================================================
by Yoon-gu Hwang, November 12, 2015

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
 - `FUGH JOHN L` : his salary is `NaN`
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


### Question 3 ###
**What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**


### Question 5 ###
**What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]**

### Question 6 ###
**Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**


## Conclusion ##

## References ##
