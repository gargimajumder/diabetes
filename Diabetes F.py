#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

#Import necessary libraries
import numpy as np
import pandas as pd

#Import vizualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Import models and metrices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, RocCurveDisplay


# In[2]:


#Loading the dataset
diabetes_data = pd.read_csv(r'C:\Users\GARGI\Downloads\diabetes.csv')

#Print the first 5 rows of the dataframe.
diabetes_data.head()


# In[3]:


diabetes_data.shape


# In[4]:


diabetes_data.info(verbose=True)


# In[5]:


diabetes_data.describe()


# In[6]:


diabetes_data.describe().T


# # Scatter matrix of uncleaned data

# In[8]:


from pandas.plotting import scatter_matrix

# Assuming you have a DataFrame named diabetes_data
p=scatter_matrix(diabetes_data, figsize=(25, 25))


# Following columns or variables have an invalid zero value:
# 
# 1.Glucose
# 
# 2.BloodPressure
# 
# 3.SkinThickness
# 
# 4.Insulin
# 
# 5.BMI
# 
# It is better to replace zeros with nan since after that counting them would be easier and zeros need to be replaced with suitable values

# In[9]:


diabetes_data_copy = diabetes_data.copy(deep = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(diabetes_data_copy.isnull().sum())


# To fill these Nan values the data distribution needs to be understood

# In[10]:


p = diabetes_data.hist(figsize = (20,20))


# Aiming to impute nan values for the columns in accordance with their distribution

# In[12]:


diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)


# # Plotting after Nan removal

# In[13]:


p = diabetes_data_copy.hist(figsize = (20,20))


# In[14]:


diabetes_data_copy.shape


# # Pair plot for clean data

# In[15]:


p=sns.pairplot(diabetes_data_copy, hue = 'Outcome')


# # Heatmap for clean data

# In[16]:


plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data_copy.corr(), annot=True)  # seaborn has very simple solution for heatmap


# In[17]:


# Data and Target Split
x = diabetes_data_copy.drop('Outcome', axis=1)  # 'Outcome' is your target column
y = diabetes_data_copy['Outcome']


# In[18]:


# Train and Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

x_train = pd.DataFrame(x_train, columns=x.columns)
x_test = pd.DataFrame(x_test, columns=x.columns)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[19]:


# Perform Feature Scaling

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)
x_train = pd.DataFrame(x_train, columns=x.columns)
x_test = pd.DataFrame(x_test, columns=x.columns)


# In[20]:


outcome_counts = diabetes_data['Outcome'].value_counts()
# Count Plot for Diabetes
plt.bar(outcome_counts.index, outcome_counts.values, color=['steelblue', 'peru'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Count of Outcomes')
plt.xticks(outcome_counts.index, ['No Diabetes(0)', 'Diabetes(1)'])  # Assuming 0 is 'No Diabetes' and 1 is 'Diabetes'
plt.show()
print(diabetes_data['Outcome'].value_counts())


# The above graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. The number of non-diabetics is almost twice the number of diabetic patients

# In[71]:


# Define a dictionary to store the results

results = {}


# # LOGISTIC REGRESSION 

# In[72]:


# Logistic Regression

lgr = LogisticRegression()
lgr.fit(x_train, y_train)


# In[73]:


y_pred = lgr.predict(x_test)
results['Logistic Regression'] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]


# In[74]:


print(results['Logistic Regression'])


# In[75]:


RocCurveDisplay.from_estimator(lgr, x_test, y_test, name='Logistic')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
plt.show()


# In[76]:


ConfusionMatrixDisplay.from_estimator(lgr, x_test, y_test, colorbar = False)
plt.title('Confusion Matrix of Logistic Regression')
plt.grid()


# # K-Nearest Neighbors (KNN)

# In[77]:


# K-Nearest Neighbors (KNN)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)


# In[78]:


y_pred = knn.predict(x_test)
results['K-Nearest Neighbors'] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]


# In[79]:


print(results['K-Nearest Neighbors'])


# In[80]:


RocCurveDisplay.from_estimator(knn, x_test, y_test, name='KNN')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
plt.show()


# In[81]:


ConfusionMatrixDisplay.from_estimator(knn, x_test, y_test, colorbar = False)
plt.title('Confusion Matrix of KNN')
plt.grid()


# # Support Vector Machines with Linear Kernel

# In[82]:


# Support Vector Machines with Linear Kernel

svm_linear = SVC(kernel = 'linear')
svm_linear.fit(x_train, y_train)


# In[83]:


y_pred = svm_linear.predict(x_test)
results['Support Vector Machines with Linear Kernel'] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]


# In[84]:


print(results['Support Vector Machines with Linear Kernel'])


# In[85]:


RocCurveDisplay.from_estimator(svm_linear, x_test, y_test, name='Support Vector Mchines with Linear Kernel')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
plt.show()


# In[86]:


ConfusionMatrixDisplay.from_estimator(svm_linear, x_test, y_test, colorbar = False)
plt.title('Confusion Matrix of Support Vector Machines with Linear Kernel')
plt.grid()


# # Support Vector Machines with RBF Kernel

# In[87]:


# Support Vector Machines with RBF Kernel

svm_rbf = SVC(kernel = 'rbf')
svm_rbf.fit(x_train, y_train)


# In[88]:


y_pred = svm_rbf.predict(x_test)
results['Support Vector Machines with RBF Kernel'] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]


# In[89]:


print(results['Support Vector Machines with RBF Kernel'] )


# In[90]:


RocCurveDisplay.from_estimator(svm_rbf, x_test, y_test, name='Support Vector Mchines with RBF Kernel')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
plt.show()


# In[91]:


ConfusionMatrixDisplay.from_estimator(svm_rbf, x_test, y_test, colorbar = False)
plt.title('Confusion Matrix of Support Vector Machines with RBF Kernel')
plt.grid()


# # Decision Tree

# In[92]:


# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[93]:


y_pred = dt.predict(x_test)
results['Decision Tree'] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]


# In[94]:


print(results['Decision Tree'])


# In[95]:


RocCurveDisplay.from_estimator(dt, x_test, y_test, name='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
plt.show()


# In[96]:


ConfusionMatrixDisplay.from_estimator(dt, x_test, y_test, colorbar = False)
plt.title('Confusion Matrix of Decision Tree')
plt.grid()


# # Random Forest

# In[97]:


# Random Forest

rf = RandomForestClassifier()
rf.fit(x_train, y_train)


# In[98]:


y_pred = rf.predict(x_test)
results['Random Forest'] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]


# In[99]:


print(results['Random Forest'])


# In[100]:


RocCurveDisplay.from_estimator(rf, x_test, y_test, name='Random Forest')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
plt.show()


# In[101]:


ConfusionMatrixDisplay.from_estimator(rf, x_test, y_test, colorbar = False)
plt.title('Confusion Matrix of Random Forest')
plt.grid()


# In[102]:


# Create a DataFrame for the results
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
df_results


# # OUTPUT DISCUSSION

# The randomforest exhibits 75% accuracy, Decision Tree has 73% accuracy, SVM with RBF Kernel has 74% accuracy, SVM with Linear Kernel has 75% accuracy, KNN models has 72% accuracy and the logistic regression has 76% accuracy. 
# Clearly the best model for prediction based on test dataset is Logistic regression 
# model. 

# In[ ]:




