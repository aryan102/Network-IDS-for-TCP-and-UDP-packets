import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing Dataset
dataset = pd.read_csv('MainDataset.csv')
X = dataset.iloc[:, 1:26].values
y = dataset.iloc[:, 26].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 64, activation = 'relu', input_dim = 25 ))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 1000, nb_epoch = 1)
parameters ={'batch_size':[1000],
            'nb_epoch':[1],
            'optimizer':['adam']
            }
random_search= RandomizedSearchCV(estimator=classifier, param_distributions=parameters,n_iter=20,n_jobs=-1,cv=5)
random_search = random_search.fit(X_train,y_train)
best_parameters = random_search.best_params_ 
best_accuracy = random_search.best_score_
print("best accuracy : ",best_accuracy)
print("best params : ",best_parameters)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_test,y_pred))

count1=dataset['TCP Protocol'].value_counts().UDP
count2=dataset['TCP Protocol'].value_counts().TCP
x=[]
for i in range(0,count1) :
    x.insert(i,0)
for i in range(0,count2):
    x.insert(i,1)
num_bins=2
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7), 
                        tight_layout = True)
axs.hist(x, bins = num_bins)
rects = axs.patches
labels = ["TCP Packets", "UDP Packets"]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    axs.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
            ha='center', va='bottom')
plt.show()
