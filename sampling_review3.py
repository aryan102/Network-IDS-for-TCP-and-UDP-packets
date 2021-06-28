import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
df=pd.read_csv("dataset_review3.csv")
df = df.iloc[:,1:28]
count_class_0, count_class_1, count_class_2 = df.TCP_Protocol.value_counts()
df_class_0 = df[df['TCP_Protocol'] == 'TCP']
df_class_1 = df[df['TCP_Protocol'] == 'UDP']
df_class_2 = df[df['TCP_Protocol'] == 'Others']
df_class_0_under = df_class_0.sample(10000)
df_class_1_under = df_class_1.sample(10000)
df_under = pd.concat([df_class_0_under, df_class_1_under], axis=0)
df_under = pd.concat([df_under, df_class_2], axis=0)
df_n = df_under
df_n.to_csv("MainDataset.csv")