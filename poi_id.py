#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier,dump_classifier_and_data
import matplotlib.pyplot as plt
from collections import defaultdict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary','deferral_payments','total_payments','loan_advances','bonus','restricted_stock_deferred',
                      'deferred_income','total_stock_value','expenses','exercised_stock_options','other','long_term_incentive',
                      'restricted_stock','director_fees']
email_features = ['to_messages','from_poi_to_this_person','from_messages','from_this_person_to_poi','shared_receipt_with_poi']
poi_label = ['poi']
features_list = poi_label + email_features + financial_features # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Number of data available
print len(data_dict.keys())
### Number of feature available
print len(data_dict[data_dict.keys()[0]])
###Counting Number of POI
poi = int()
non_poi = int()
for key,values in data_dict.items():
    if data_dict [key]['poi'] == True:
        poi +=1
    else :
        non_poi += 1
print poi,non_poi
    
###counting NaN values
count = defaultdict(int)
for key,values in data_dict.items():
    for k,v in values.items():
        if v == 'NaN':
            count[k] += 1
print count
    
### Task 2: Remove outliers
def createplot(data_dict,x):
    plot_data = featureFormat(data_dict,x)
    for data in plot_data:
        x = data[0]
        y = data[1]
        plt.scatter(x,y) 
    plt.show()
createplot(data_dict,['salary','expenses'])

createplot(data_dict,['from_messages','to_messages'])
createplot(data_dict,['from_poi_to_this_person','from_this_person_to_poi'])
salary_outlier =  featureFormat(data_dict,['salary'])
outliers = []
for key,values in data_dict.items():
    if float(data_dict[key]['salary']) == max(salary_outlier):
        outliers.append(key)
###checking for the blank financial and email features
financial_feature_nan_count = defaultdict(int)
for key in data_dict.keys():
    for feature in financial_features:
        if data_dict[key][feature] == 'NaN':
            financial_feature_nan_count[key] +=1
email_feature_nan_count = defaultdict(int)
for key in data_dict.keys():
    for feature in email_features:
        if data_dict [key][feature] == 'NaN':
            email_feature_nan_count[key] += 1
for key in data_dict.keys():
    if (key in financial_feature_nan_count.keys()) and (key in email_feature_nan_count.keys()):
        if financial_feature_nan_count[key] == len(financial_features) and email_feature_nan_count[key] == len(email_features):
            outliers.append(key)
### remove outlier:
outliers = outliers + ['THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier,0)
    print "poped",outlier
createplot(data_dict,['salary','expenses'])
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for key in my_dataset.keys():
    if my_dataset[key]['other'] != 'NaN' and my_dataset [key] ['total_payments'] != 'NaN':
        ratio = my_dataset [key] ['other']/float(my_dataset[key]['total_payments'])
        my_dataset[key]['ratio_of_otherpayment_from_totalpayments'] = ratio
    else:
        my_dataset[key]['ratio_of_otherpayment_from_totalpayments'] = 0.
new_features_list =  features_list + ['ratio_of_otherpayment_from_totalpayments']
        
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
best = SelectKBest(k=9)
best.fit_transform(features,labels)
mask = best.get_support()
score = best.scores_
new_features=[]
score_of_new_features = defaultdict(float)
for bool,feature,score in zip(mask,features_list[1:],score):
    if bool:
        new_features.append(feature)
        score_of_new_features[feature] = score 
print score_of_new_features
new_features = ['poi'] + new_features
from sklearn.preprocessing import MinMaxScaler
def minmax(features):
    scaler = MinMaxScaler(feature_range =(0,1),copy=False)
    features = scaler.fit_transform(features)
    return features
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
###new_features = new_features + ['ratio_of_otherpayment_from_totalpayments']
data = featureFormat(my_dataset,new_features,sort_keys =True)
labels,features = targetFeatureSplit(data)
features = minmax(features)
features_train, features_test, labels_train, labels_test= train_test_split(features,labels,test_size =  0.3,random_state = 42)
clf = GaussianNB()
clf.fit(features_train,labels_train)
clf.predict(features_test)
###code for additional classifiers used
'''
from sklearn import tree
parameters = {'criterion':['gini','entropy'],'min_samples_split' : [10,15,20]}
tree_classifier = tree.DecisionTreeClassifier()
clf = GridSearchCV(tree_classifier,parameters)
clf.fit(features_train,labels_train)
clf.predict(features_test)
print clf.best_params_
from sklearn.cluster import KMeans
parameters={'n_clusters':[2],'max_iter' :[300,350,400]}
means = KMeans()
clf = GridSearchCV(means,parameters)
clf.fit(features_train,labels_train)
clf.predict(features_test)
print clf.best_params_
from sklearn.linear_model import LogisticRegression
parameters = {'C':[0.0005,0.005,0.01,0.05,.1],'tol' : [0.0001,0.001]}
grid = LogisticRegression()
clf = GridSearchCV(grid,parameters)
clf.fit(features_train,labels_train)
clf.predict(features_test)
print clf.best_params_
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset,new_features)
