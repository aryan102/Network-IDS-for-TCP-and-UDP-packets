import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing Dataset
dataset = pd.read_csv('MainDataset.csv')
X = dataset.iloc[:, 1:26].values
y = dataset.iloc[:, 26].values
# Splitting the dataset into Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer = 'rmsprop'):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 25 ))
    classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 1000, nb_epoch = 1)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 3, n_jobs = 1)
mean = accuracies.mean()
print("accuracy : ",mean)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_test,y_pred))



  



