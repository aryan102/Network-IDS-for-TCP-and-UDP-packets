import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing Dataset
dataset = pd.read_csv('MainDataset.csv')
X = dataset.iloc[:, 1:26].values
y = dataset.iloc[:, 26].values
# Splitting the dataset into Training Set and Testing Set
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
