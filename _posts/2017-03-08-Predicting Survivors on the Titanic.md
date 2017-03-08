---
layout: post
image: '/img/titanic.jpg'
title: "Predicting Survivors on the Titanic"
subtitle: "Category: Model Hyper-tuning + Bagging"
---


For this project we will be applying logistic regression, kNN and decision tree classification models the Titanic dataset to predict whether or not a passenger survives.  As part of our modeling approach we will compare the base estimator models with models that are hyper-tuned and bagged.  Using this classic dataset we hope to demonstrate model performance gains by way of model optimization as well as resampling methods.  We will perform some light feature engineering in addition to the imputation of several missing values in the dataset.  The [dataset](https://www.kaggle.com/c/titanic/data) can be obtained from Kaggle along with a description of the variables.  The project will conclude with a discussion of the precision, recall and ROC curves of the model predictions.

We will assume the following risks and assumptions:

- Model does not generalize and should only be used to predict survivability on the Titanic (not other crashes)
- Mean imputation of age is an acceptable estimation
- All continuous features satisfy normality condition
- Floorplan on the Titanic is identical among different floors/levels
- Cabin numbers close in value are also close in proximity on ship (reason for binning cabin numbers)

This project will be focused around the following sections and is intended for a technical audience:

- Data Cleaning and Feature Engineering
- Logistic Regression
- k-Nearest Neighbors
- Decision Trees
- Model Comparisons

## Data Cleaning and Feature Engineering

Here we prepare the data for our models.  We try things like searching the 'Name' column for salutations such as *Miss*, *Dr.* or *Rev.*  Also, we break apart the 'Cabin' column to see if we can discern anything from the cabin letter or grouping of cabin numbers.  The categorical variables (e.g., 'Pclass', 'Sex', 'Embarked', etc.) were dummified.  The 'Age' variable was imputed with the mean age of the dataset.  


```python
# Load packages and import data
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler

from sqlalchemy import create_engine
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic')

df = pd.read_sql('SELECT * FROM train', engine)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Function to retrieve cabin number
def cabin_num(x):
    if x is None:
        return x
    else:
        num = ''.join(i for i in x.split()[0] if i.isdigit())
        if num == '':
            return None
        else:
            return int(num)

df['Cabin_num'] = df['Cabin'].apply(cabin_num)

# Function to retrieve cabin letter
def cabin_letter(x):
    if x is None:
        return x
    else:
        letter = ''.join(i for i in x.split()[0] if i.isalpha())
        if letter == '':
            return None
        else:
            return letter
df['Cabin_letter'] = df['Cabin'].apply(cabin_letter)

# Function to find out whether or not person is a Mrs.
def find_Mrs(x):
    if 'Mrs' in x:
        return 1
    else:
        return 0

df['Mrs'] = df['Name'].apply(find_Mrs)

# Function to find out whether or not person is a Ms.
def find_Ms(x):
    if 'Ms' in x:
        return 1
    else:
        return 0
df['Ms'] = df['Name'].apply(find_Ms)

# Function to find out whether or not person is a Miss
def find_Miss(x):
    if 'Miss' in x:
        return 1
    else:
        return 0 
df['Miss'] = df['Name'].apply(find_Miss)

# Function to find out whether or not person is a Dr.
def find_Dr(x):
    if 'Dr' in x:
        return 1
    else:
        return 0     
df['Dr'] = df['Name'].apply(find_Dr)

# Function to find out whether or not person is a Rev.
def find_Rev(x):
    if 'Rev' in x:
        return 1
    else:
        return 0     
df['Rev'] = df['Name'].apply(find_Rev)
```


```python
# Convert Pclass to categorical variable
df['Pclass'] = df['Pclass'].astype('category')

# Bin cabin numbers into groups of nearby cabins 
bins = range(0,160,10)
group_names = map(str,range(0,15))
df['Cabin_num'] = pd.cut(df['Cabin_num'], bins, labels = group_names)

# Impute NaN ages with mean age
imputer = Imputer(strategy='mean', axis=0)
df['Age'] = imputer.fit_transform(df['Age'].reshape(-1,1))

# Drop columns that are no longer needed
df.drop(['index','PassengerId','Ticket','Name','Ticket', 'Cabin'],axis=1, inplace=True)

# Dummify categorical variables
df = pd.get_dummies(df, drop_first=True)
```


```python
# Set feature and target variable inputs
X = df.iloc[:,1:]
y = df['Survived']

# Scale input variables
standard_scaler = StandardScaler()
Xt = standard_scaler.fit_transform(X)
```

## Logistic Regression

Now we will run our first base model: Logistic Regression.


```python
# Import packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report
```


```python
# Initialize lists to keep track of model scores
model_score = []
gridsearch_score = []
bagged_score = []
```


```python
# Instantiate logistic regression model
logreg = LogisticRegression(solver='liblinear')

# Calculate logreg CV score and append to score list
logreg_cvs= cross_val_score(logreg,Xt,y,cv=5).mean()
model_score.append(logreg_cvs)
print logreg_cvs

# Set up gridsearch parameter space
C_vals = np.logspace(-5,2,21)
penalties = ['l1','l2']

# Fit gridsearch logreg model
gs = GridSearchCV(logreg, {'penalty': penalties, 'C': C_vals},\
                  verbose=False, cv=5)
gs.fit(Xt, y)
```

    0.801391381689





    GridSearchCV(cv=5, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'penalty': ['l1', 'l2'], 'C': array([  1.00000e-05,   2.23872e-05,   5.01187e-05,   1.12202e-04,
             2.51189e-04,   5.62341e-04,   1.25893e-03,   2.81838e-03,
             6.30957e-03,   1.41254e-02,   3.16228e-02,   7.07946e-02,
             1.58489e-01,   3.54813e-01,   7.94328e-01,   1.77828e+00,
             3.98107e+00,   8.91251e+00,   1.99526e+01,   4.46684e+01,
             1.00000e+02])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=False)




```python
gs.best_params_
```




    {'C': 0.79432823472428049, 'penalty': 'l2'}




```python
# Instantiate logreg model with optimized parameters
logreg_gs = LogisticRegression(C=gs.best_params_['C'],\
                            penalty=gs.best_params_['penalty'])

# Calculate gridsearched logreg CV score and append to score list
logreg_gs_cvs = cross_val_score(logreg_gs,Xt,y,cv=5).mean()
gridsearch_score.append(logreg_gs_cvs)
print logreg_gs_cvs
```

    0.803632295631



```python
# Instantiate logreg model with bagging
logreg_bagger = BaggingClassifier(logreg,n_estimators=100)

# Calculate bagged logreg CV score and append to score list
logreg_bagged_cvs = cross_val_score(logreg_bagger,Xt,y,cv=5).mean()
bagged_score.append(logreg_bagged_cvs)
print logreg_bagged_cvs
```

    0.798007970106


## K Nearest Neighbors

Now we will run our next base model: k-Nearest Neighbors.


```python
# Instantiate kNN model
knn = KNeighborsClassifier()

# Calculate knn CV score and append to score list
knn_cvs = cross_val_score(knn,Xt,y,cv=5).mean()
model_score.append(knn_cvs)
print knn_cvs

# Set up gridsearch parameter space
n_neighbors = range(1,10)
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

# Fit gridsearch knn model
gs = GridSearchCV(knn, {'n_neighbors': n_neighbors, 'algorithm': algorithm},\
                  verbose=False, cv=5)
gs.fit(Xt, y)
```

    0.775636209402





    GridSearchCV(cv=5, error_score='raise',
           estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform'),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=False)




```python
gs.best_params_
```




    {'algorithm': 'auto', 'n_neighbors': 9}




```python
# Instantiate knn model with optimized parameters
knn_gs = KNeighborsClassifier(n_neighbors=gs.best_params_['n_neighbors'],\
                            algorithm=gs.best_params_['algorithm'])

# Calculate gridsearched knn CV score and append to score list
knn_gs_cvs = cross_val_score(knn_gs,Xt,y,cv=5).mean()
gridsearch_score.append(knn_gs_cvs)
print knn_gs_cvs
```

    0.785691862541



```python
# Instantiate knn model with bagging
knn_bagger = BaggingClassifier(knn,n_estimators=100)

# Calculate bagged knn CV score and append to score list
knn_bagged_cvs = cross_val_score(knn_bagger,Xt,y,cv=5).mean()
bagged_score.append(knn_bagged_cvs)
print knn_bagged_cvs
```

    0.781248051715


## Decision Trees

Now we will run our last base model: Decision Trees.


```python
# Instantiate decision tree model
dt = DecisionTreeClassifier()

# Calculate dt CV score and append to score list
dt_cvs = cross_val_score(dt,Xt,y,cv=5).mean()
model_score.append(dt_cvs)
print dt_cvs

# Set up gridsearch parameter space
max_depth = (range(1,5))
max_features = ((.25,.5,.75,1))
max_leaf_nodes = (range(2,4))
min_samples_leaf = (range(2,6))
min_samples_split = (range(2,6))

# Fit gridsearch dt model
gs = GridSearchCV(dt, {'max_depth': max_depth, 'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split},\
                  verbose=False, cv=5)
gs.fit(Xt, y)
```

    0.776753598765





    GridSearchCV(cv=5, error_score='raise',
           estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best'),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'max_features': (0.25, 0.5, 0.75, 1), 'max_leaf_nodes': [2, 3], 'min_samples_split': [2, 3, 4, 5], 'max_depth': [1, 2, 3, 4], 'min_samples_leaf': [2, 3, 4, 5]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=False)




```python
gs.best_params_
```




    {'max_depth': 4,
     'max_features': 0.25,
     'max_leaf_nodes': 3,
     'min_samples_leaf': 3,
     'min_samples_split': 2}




```python
# Instantiate dt model with optimized parameters
dt_gs = DecisionTreeClassifier(max_depth=gs.best_params_['max_depth'],\
                            max_leaf_nodes=gs.best_params_['max_leaf_nodes'], min_samples_leaf=gs.best_params_['min_samples_leaf'],\
                           min_samples_split=gs.best_params_['min_samples_split'])

# Calculate gridsearched dt CV score and append to score list
dt_gs_cvs = cross_val_score(dt_gs,Xt,y,cv=5).mean()
gridsearch_score.append(dt_gs_cvs)
print dt_gs_cvs
```

    0.7733072037



```python
# Instantiate dt model with bagging
dt_bagger = BaggingClassifier(dt,n_estimators=100)

# Calculate bagged dt CV score and append to score list
dt_bagged_cvs = cross_val_score(dt_bagger,Xt,y,cv=5).mean()
bagged_score.append(dt_bagged_cvs)
print dt_bagged_cvs
```

    0.823863575511


## Model Comparisons

Overall, we find that all three models perform very similarly in terms of their cross validation accuracies.  The table below shows slight improvements were made on the KNN and decision tree base models with GridSearchCV optimization and bagging (100 trees).  The best model in terms of accuracy seems to be the GridSearchCV optimized logistic regression model.  Although the decision tree bagging model has a slightly higher accuracy score, we believe the bagged decision tree model is overfit (high variance) given that the model gives near perfect accuracy on the full training set.  Here the bagging was applied to the base model, not the hyper-tuned model.  Very little improvement was observed in the logistic regression model (if any), which indicates that the base model was already very close to its optimum.


```python
# Compare model accuracies
model_comp = pd.DataFrame(zip(model_score, gridsearch_score, bagged_score), columns=['Model Accuracy','GridSearch Accuracy', 'Bagging Accuracy'], index=['Log Reg', 'KNN', 'Decision Tree'])
model_comp
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model Accuracy</th>
      <th>GridSearch Accuracy</th>
      <th>Bagging Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Log Reg</th>
      <td>0.801391</td>
      <td>0.803632</td>
      <td>0.798008</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.775636</td>
      <td>0.785692</td>
      <td>0.781248</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.776754</td>
      <td>0.773307</td>
      <td>0.823864</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confusion matrix for logistic regression model with bagging
y_pred_lr_gs = logreg_gs.fit(Xt,y).predict(Xt)
conmat = np.array(confusion_matrix(y,y_pred_lr_gs))
lr_confusion = pd.DataFrame(conmat, index=['true_survived', 'true_perished'],
                            columns=['predicted_survived','predicted_perished'])

# Confusion matrix for knn model with bagging
y_pred_knn_gs = knn_gs.fit(Xt,y).predict(Xt)
conmat = np.array(confusion_matrix(y,y_pred_knn_gs))
knn_confusion = pd.DataFrame(conmat, index=['true_survived', 'true_perished'],
                            columns=['predicted_survived','predicted_perished'])

# Confusion matrix for dt model with bagging
y_pred_dt_gs = dt_gs.fit(Xt,y).predict(Xt)
conmat = np.array(confusion_matrix(y,y_pred_dt_gs))
dt_confusion = pd.DataFrame(conmat, index=['true_survived', 'true_perished'],
                            columns=['predicted_survived','predicted_perished'])

# Print confusion matrices
print 'Logistic Regression'
print lr_confusion
print '\n'
print 'KNN'
print knn_confusion
print '\n'
print 'Decision Tree'
print dt_confusion
```

    Logistic Regression
                   predicted_survived  predicted_perished
    true_survived                 482                  67
    true_perished                  86                 256
    
    
    KNN
                   predicted_survived  predicted_perished
    true_survived                 471                  78
    true_perished                  79                 263
    
    
    Decision Tree
                   predicted_survived  predicted_perished
    true_survived                 540                   9
    true_perished                 181                 161



```python
# Print classification reports
print 'Logistic Regression Classification Report'
print classification_report(y, y_pred_lr_gs)
print '\n'
print 'KNN Classification Report'
print classification_report(y, y_pred_knn_gs)
print '\n'
print 'Decision Tree Classification Report'
print classification_report(y, y_pred_dt_gs)
```

    Logistic Regression Classification Report
                 precision    recall  f1-score   support
    
              0       0.85      0.88      0.86       549
              1       0.79      0.75      0.77       342
    
    avg / total       0.83      0.83      0.83       891
    
    
    
    KNN Classification Report
                 precision    recall  f1-score   support
    
              0       0.86      0.86      0.86       549
              1       0.77      0.77      0.77       342
    
    avg / total       0.82      0.82      0.82       891
    
    
    
    Decision Tree Classification Report
                 precision    recall  f1-score   support
    
              0       0.75      0.98      0.85       549
              1       0.95      0.47      0.63       342
    
    avg / total       0.82      0.79      0.77       891
    


Printed above are the classification reports for all three grid-searched models.  In addition to having a slightly higher cross validation accuracy score, the logistic regression model also has higher precision and recall scores as compared to the KNN and decision tree models.  A higher precision score means that when we predict a passenger survived we are more confident that they actually survived.  Logistic regression also gives the highest recall score.  A higher recall score means that when a passenger actually survives we are more likely to predict that they have survived.  In terms of precision and recall the decision tree performs the worst.  


```python
# Plot ROC curves for each model on same plot
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
%matplotlib inline

Y_score_lr = logreg_gs.fit(Xt,y).predict_proba(Xt)[:,1]
Y_score_knn = knn_gs.fit(Xt,y).predict_proba(Xt)[:,1]
Y_score_dt = dt_gs.fit(Xt,y).predict_proba(Xt)[:,1]

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For logreg, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y, Y_score_lr)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for logreg
plt.figure(figsize=[11,9])
plt.plot(FPR[1], TPR[1], label='LogReg ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)

# For knn, find the area under the curve
FPR[2], TPR[2], _ = roc_curve(y, Y_score_knn)
ROC_AUC[2] = auc(FPR[2], TPR[2])

# Plot of a ROC curve for knn model
plt.plot(FPR[2], TPR[2], label='KNN ROC curve (area = %0.2f)' % ROC_AUC[2], linewidth=4)

# For dt, find the area under the curve
FPR[3], TPR[3], _ = roc_curve(y, Y_score_dt)
ROC_AUC[3] = auc(FPR[3], TPR[3])

# Plot of a ROC curve for dt model
plt.plot(FPR[3], TPR[3], label='DT ROC curve (area = %0.2f)' % ROC_AUC[3], linewidth=4)


plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()
```


![png](/img/project-5_files/project-5_34_0.png)


The above plot compares the performance of the grid-searched logistic regression, KNN and decision tree models as its discrimination threshold is varied.  We find that the KNN model has the highest area under the curve, followed by the logistic regression model, and then decision tree model.  The plot demonstrates that no one model is best for all cases.  For example, although we've said the logistic regression model yields the highest cross validation accuracy, precision and recall, in cases where the false positive rate is high (above 0.20) we may actually prefer the KNN model as it provides a higher true positive rate and, therefore, a higher precision score.  Based on the above figure, there doesn't appear to be many cases where the grid-searched decision tree model would be preferable over the other two models.  When comparing the bagged models we noticed the decicion tree model was extremely overfit.  Although the cross validation accuracy went up with the bagged decision tree model, we also found that the accuracy score, precision and recall were all around 0.99 when applied to the full training set, which indicates a high degree of overfitting.  In the future, to improve our model comparison study we could include grid-searched random forest or gradient boosting models.  We would expect these models to perform better since the bagging/boosting resampling is built into these models themselves.  This means when we would perform the grid-search it will not just be applied to the base estimator, it will instead be applied to the bagged/boosted model, which will further reduce the variance of the model.
