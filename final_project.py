
# coding: utf-8
#!/usr/bin/python

#libraries and modules used in our analysis
import sys
import pickle
import pandas as pd

sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from feature_format import featureFormat, targetFeatureSplit
from sklearn.ensemble import AdaBoostClassifier
from tester import*
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from xgboost import XGBClassifier

### Task 1: Data Preparation and Feature Selection

### Load the dictionary containing the dataset and convert it to Pandas df:
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient = 'index', )  

#Checking basic information about dataframe
print 'Dataframe shape:', df.shape
print 'Dataframe basic info columnwise:', df.info()


#drop 'email_address' column as we don't need it in our analysis
df = df.drop('email_address',1)

#convert columns to numeric, the function will automatically convert boolean values 
#in ['poi'] column to 0 and 1
df = df.apply(lambda x: pd.to_numeric(x, errors = 'coerce'))
df.poi = df.poi.astype(int)

#detect and remove anomalies in names
for name in df.index.values.tolist():
    #normally person's name will contain first and last name and middle name letters, 
    #so we look for names longer than normal ones
    if len(name.split()) > 3:
        print 'Potential anomalies', name

df = df.drop(['THE TRAVEL AGENCY IN THE PARK'])  
print 'NaN columnwise:', df.isnull().sum()
#remove three columns with most NaNs
df = df.drop(['loan_advances', 'director_fees', 'restricted_stock_deferred'],1)


print 'NaN rowwise:', df.isnull().sum(axis =1).sort_values(ascending = True)
#remove last two rows with most NaNs
df = df.drop(['LOCKHART EUGENE E', 'GRAMM WENDY L'])

#now checking our classes
print 'Number of non-poi:', df['poi'][df['poi'] == 0].count()
print 'Number of poi:', df['poi'][df['poi'] == 1].count()

# We do imputation for financial and email features separately. 
# We impute all missing financial data with 0, and all missing email data with -1. 
# Doing it this way, we are not assigning the missing value to any meaningful number but just 'mark' 
# missing value so we could distinguish them latter in the analysis. 

financial_features = ['salary', 'deferral_payments','total_payments','exercised_stock_options', 
'bonus','restricted_stock', 'total_stock_value','expenses','other','deferred_income','long_term_incentive']
df[financial_features] = df[financial_features].fillna(0)
email_features = ['to_messages', 'from_messages', 'shared_receipt_with_poi', 'from_this_person_to_poi', 
'from_poi_to_this_person']
df[email_features] = df[email_features].fillna(-1)


col_list = df.columns.tolist()
col_list.remove('poi')

features_list = ['poi']
for i in col_list:
    features_list.append(i)


# ### Task 2: Remove outliers

#Let's plot scatterplot matrix to have a quick visualizations of our data. 
#Plot financial data and emails data separately
plot = sns.pairplot(df[['to_messages', 'from_messages', 'shared_receipt_with_poi', 'from_this_person_to_poi', 
    'from_poi_to_this_person','poi']], hue="poi")
# There seem to be no obvious outliers in emails data. 

# Now let's examine financial data
plot = sns.pairplot(df[['salary', 'deferral_payments','total_payments','exercised_stock_options', 'bonus',
    'restricted_stock', 'total_stock_value','expenses','other','deferred_income','long_term_incentive',
    'poi']], hue="poi")


#we have one obvious outlier in ['salary'] column, let's check it and remove it
df[df['salary'] > 25000000]
df = df.drop(['TOTAL'])

# ### Task 3: Create new feature(s)

#Create two new features for email data: ratio of the emails to/from poi to all to/from emails of the person
#ratio of the emails recieved from poi to the total number of incoming emails
df['from_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['to_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
features_list.append('from_poi_ratio')
features_list.append('to_poi_ratio')

features = features_list[1:]
selector = SelectKBest()
selector.fit(df[features], df['poi'])
scores = selector.scores_

#let's have a look on our scores
plt.bar(range(len(features)), scores)
plt.xticks(range(len(features)), features, rotation='vertical')
plt.show()

#first we cut off features with lowest scores, since they will not contribute to our analysis
#for starters let's keep features scored at least 5, however we will find the optimal number of 
#features with GridSearch later in the analysis 

features_list = ['poi']
feat_score_dict = {}
for feature,score in zip(features, scores):
    if score >5:
        feat_score_dict[feature] = score
        features_list.append(feature)

print 'Features to be used:', features_list 

### Store to my_dataset for easy export below.
data_dict = df.to_dict(orient = 'index')
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a variety of classifiers

#### Random Forest
scaler = RobustScaler()
pca = PCA(n_components = 5, random_state = 42)
rf_classifier = RandomForestClassifier(random_state = 42)
selector = SelectKBest(k='all')

#define steps for the pipeline
steps = [('feature_scaling', scaler),
        ('feature_selection', selector),
        ('pc_analysis', pca),
        ('classification_algorithm', rf_classifier)]

clf_rf = Pipeline(steps)
clf_rf.fit(features, labels)

print 'Udacity tester results on Random Forest classifier:'
test_classifier(clf_rf, my_dataset, features_list)

#### AdaBoost
scaler = RobustScaler()
pca = PCA(n_components = 5, random_state = 42)
ab_classifier = AdaBoostClassifier(random_state = 42)
selector = SelectKBest(k='all')


steps = [('feature_scaling', scaler),
        ('feature_selection', selector),
        ('pc_analysis', pca),
        ('classification_algorithm', ab_classifier)]

clf_2 = Pipeline(steps)
clf_2.fit(features, labels)

print 'Udacity tester results on AdaBoost classifier:'
test_classifier(clf_2, my_dataset, features_list)

#### Decision Tree
scaler = RobustScaler()
pca = PCA(n_components = 5, random_state = 42)
dt_classifier = DecisionTreeClassifier(random_state = 42)
selector = SelectKBest(k='all')

steps = [('feature_scaling', scaler),
        ('feature_selection', selector),
        ('pc_analysis', pca),
        ('classification_algorithm', dt_classifier)]

clf_dt = Pipeline(steps)
clf_dt.fit(features, labels)

print 'Udacity tester results on Decision Tree classifier:'
test_classifier(clf_dt, my_dataset, features_list)

#### XGBoost
xgb_classifier = XGBClassifier()
scaler = RobustScaler()
pca = PCA(n_components = 5, random_state = 42)
selector = SelectKBest(k='all')


steps = [('feature_scaling', scaler),
         ('feature_selection', selector),
        ('pc_analysis', pca),
        ('classification_algorithm', xgb_classifier)]

clf_xgb = Pipeline(steps)
clf_xgb.fit(features, labels)

print 'Udacity tester results on XGBClassifier:'
test_classifier(clf_xgb, my_dataset, features_list)

#### Naive Bayes
scaler = RobustScaler()
pca = PCA(random_state = 42)
nb_classifier = GaussianNB()
selector = SelectKBest(k='all')

steps = [('feature_scaling', scaler),
        ('feature_selection', selector),
        ('pc_analysis', pca),
        ('classification_algorithm', nb_classifier)]

clf_nb = Pipeline(steps)
clf_nb.fit(features, labels)

print 'Udacity tester results on Naive Bayes classifier:'
test_classifier(clf_nb, my_dataset, features_list)


#Now we try to ensemble a voting classifier to see if it will enhance our results.
#To do that we ensemble our three most successful models : Decision Tree,
#Random Forest and Naive Bayes
estimators = []

model1 = clf_dt
estimators.append(('dt', model1))

model2 = clf_rf
estimators.append(('rf', model2))

model3 = clf_nb
estimators.append(('nb', model3))

ensemble = VotingClassifier(estimators, weights = [1,1,1])

print 'Udacity tester results for voting classifier:'
test_classifier(ensemble, my_dataset, features_list)

### Task 5: Tuning classifier
param_grid = {'feature_selection__k' : [5, 7,10,'all'],'pc_analysis__n_components': [2,3,5,None]}
c_val = StratifiedShuffleSplit(n_splits= 300, random_state=42)

clf_5 = GridSearchCV(clf_nb, param_grid = param_grid, scoring = 'f1', cv = c_val)
clf_5.fit(features, labels)

print 'Udacity tester results for tuned Naive Bayes algorithm:'
test_classifier(clf_5.best_estimator_, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_5.best_estimator_, my_dataset, features_list)

