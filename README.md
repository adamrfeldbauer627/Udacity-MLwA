# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset contains data about bank customers. We seek to predict whether a customer will subscribe to a fixed term deposit.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was the voting ensemble.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
The Scikit-learn pipeline uses logistic regression. 

The hyperparameters being used with this model were regularization strength and maximum iterations. Regularization changes the variance among the features. The regularization strength values chosen are discrete values from the following set - [0.001, 0.1, 1, 2, 3] where smaller values result in more regularization and the larger values result in less. The values span a wide range so we can test with varying degrees of regularization.  
The maximum iterations parameter represents the maximum number of iterations for convergence. If the solver fails to converge before the maximum iteration, it aborts the run. The maximum iterations are randomly chosen from a discrete list of values as follows - [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]. The values in the maximum iteration list were chosen empirically after a few tests.

The bank marketing data was used for this experiment as well as for the AutoML experiment. A record in the
bank marketing data file represents a banking prospect or customer. Each column in the bank marketing represents
a feature of the prospect or customer.  

Features:  
1 - age (numeric)  
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')  
5 - default: has credit in default? (categorical: 'no','yes','unknown')  
6 - housing: has housing loan? (categorical: 'no','yes','unknown')  
7 - loan: has personal loan? (categorical: 'no','yes','unknown')  
related with the last contact of the current campaign:  
8 - contact: contact communication type (categorical: 'cellular','telephone')  
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')  
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  
other attributes:  
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
14 - previous: number of contacts performed before this campaign and for this client (numeric)  
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')  
social and economic context attributes  
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  
17 - cons.price.idx: consumer price index - monthly indicator (numeric)  
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)  
20 - nr.employed: number of employees - quarterly indicator (numeric)  
  
Output variable (desired target):  
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')  

Feature descriptions were sourced from - https://archive.ics.uci.edu/ml/datasets/bank+marketing

**What are the benefits of the parameter sampler you chose?**
The random parameter sampler is a great sampling method because it is easy to use and accurately represents our sample by randomly sampling over our hyperparameter search space. It also runs faster than other sampling methods given our limited resources. Using the random parameter sampler, we are able to pass in discrete values and continuous ranges in the form of a dictionary object. The random sampler is superior to grid sampling in our case because it does not need to exhaust the search space entirely and it allows for an early exit policy. In our case, the random sampler is also superior to a bayesian sampler because the bayesian sampler requires significantly more resources and time to cover the search space.

**What are the benefits of the early stopping policy you chose?**
The BanditPolicy gives us greater control over which runs to terminate using slack criteria, frequency and a delay interval. The slack factor is a ratio used to calculate the allowed distance from the best performing run. The slack amount is the absoluate distance allowed from the best performing run. The delay evaluation is the number of intervals for which to delay the first policy evaluation. The parameters chosen terminate runs where the best metric is less than 91% of the best run.  
The bandit policy gives us more control over the conditions with which we terminate runs. Other policies such as the median stopping policy and the truncation selection policy do not allow the same control.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The best model was the voting ensemble. A voting ensemble is the average of multiple machine learning models. The hyperparameters used are as follows:
  {
    random_state=0,
    reg_alpha=0,
    reg_lambda=1.0416666666666667,
    scale_pos_weight=1,
    seed=None,
    silent=None,
    subsample=0.9,
    tree_method='auto',
    verbose=-10,
    verbosity=0
  }

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The best HyperDrive run had an accuracy of 0.9102. The best AutoML run had an accuracy of 0.9179. The difference in accuracy was minimal at 0.77%. The data used for testing and training the models was the same; however, the subset of data used for training and testing was randomized. The models used were considerable different. The voting ensemble averages results from multiple machine learning models. Logistic regression is a single model used for regression style problems. The returned value is then categorized for use with classification in our case. The difference in accuracy is likely due to the ensemble learner being optimized for classification and the logistic regression learner being best suited for regression problems.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
We could allocate more resources to the compute cluster and allow for longer training sessions to try to improve the accuracy of our models.
We could use automated feature engineering to ensure we are selecting good features for the predictions we are attempting.
