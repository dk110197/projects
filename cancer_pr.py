#importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing Data
cancer_data=pd.read_csv('datasets_180_408_data (1).csv')
cancer_data.head()
cancer_data.shape
cancer_data.info()
cancer_data.isnull()
cancer_data.isnull().sum()
cancer_data.describe()
cancer_data.columns
cancer_data.drop(["id",'Unnamed: 32'],axis=1,inplace=True)
cancer_data.head()
cancer_data.describe()
cancer_data.shape
cancer_data.info()
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
cancer_data['diagnosis']= label_encoder.fit_transform(cancer_data['diagnosis']) 
  
cancer_data['diagnosis'].unique()
cancer_data.head()
# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(cancer_data)
cancer_data.corr()#gives the correlation between them


# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(cancer_data.corr(), annot = True, cmap ='coolwarm', linewidths=2)

plt.figure(figsize=(16,9))
sns.heatmap(cancer_data)
cancer_data.corrwith(cancer_data.diagnosis) # visualize correlation barplot
plt.figure(figsize = (16,5))
ax = sns.barplot(cancer_data.corrwith(cancer_data.diagnosis).index, cancer_data.corrwith(cancer_data.diagnosis))
ax.tick_params(labelrotation = 90) 

#Fitting into models
X = cancer_data.drop(['diagnosis'], axis = 1) 
X.head(6)

y=cancer_data['diagnosis']
y.head(5)

# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 5)

#Feature scaling of data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  #for classification report
#Svm model
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuarcy_svm=accuracy_score(y_test, y_pred_scv)
print(accuarcy_svm)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty='l2')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)
accuarcy_lr=accuracy_score(y_test, y_pred_lr)
print(accuarcy_lr)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion ='entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuarcy_dt=accuracy_score(y_test, y_pred_dt)
print(accuarcy_dt)

from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create randomized search 5-fold cross validation and 100 iterations
clf = RandomizedSearchCV(lr_classifier, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

# Fit randomized search
best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Predict target vector
y_predict=best_model.predict(X_test)
y_predict


accuracy_score(y_test, y_predict)

 
CF = confusion_matrix(y_test, y_predict) 
  
print('Matrix:')
print(CF) 
print('Accuracy:',accuracy_score(y_test, y_predict))

print(classification_report(y_test, y_predict))

#Saving model for deployment
import pickle
#for dumping the model or we can use joblib library
pickle.dump(best_model,open('model.pkl','wb'))

# load model
best_model=pickle.load(open('model.pkl','rb'))
# predict the output
print(best_model.predict([[15.300,25.27,102.40,732.4,0.10820,0.16970,0.16830,0.087510,0.1926,0.06540,1.0950,0.9053,8.589,153.40,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,20.27,36.71,149.30,1269.0,0.1641,0.6110,0.63350,0.20240,0.4027,0.09876]]))





















