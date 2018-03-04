import pandas as pd
import numpy as np
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
filename = 'Indian Liver Patient Dataset (ILPD).csv'
#loading dataset
dataset = pd.read_csv(filename, header=None)

#dimensions of the dataset
print(dataset.shape)

#peeking into the dataset
set_option('display.width',50)
print(dataset.head(20))

#descriptive statistics
print(dataset.describe(include = 'all'))
#9th feature of the dataset has missing values
print(dataset.info())

#distribution of the target labels
print(dataset.groupby(10).size())

# drop rows with missing values
dataset[[9]] = dataset[[9]].replace(' ', np.NaN)
dataset.dropna(inplace=True)

#info of the dataset after removing missing values
print(dataset.info())

#re-mapping the target labels from 2 to 0
dataset[[10]] = dataset[[10]].replace(2, 0)

#dropping the gender column
dataset.drop(dataset.columns[[1]],axis=1,inplace=True)

#Feature Scaling
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[0:5,:])

#kfold cross validation and training the model using all the remaining features
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, rescaledX, Y, cv=kfold)
print('Accuracy: %.3f   %.3f'%(results.mean()*100.0, results.std()*100.0))

#evaluation metrics
#confusion matrix
TestSize = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TestSize,
random_state=seed)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)

#precision and recall scores
print(precision_score(Y_test, predicted))
print(recall_score(Y_test, predicted))

#F1 score
print(f1_score(Y_test, predicted))

