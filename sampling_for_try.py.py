from typing import Protocol
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score , accuracy_score
df = pd.read_csv('MainDataset.csv')
print(df.shape)
print(df.head())
df = df.iloc[:,1:28]
df_class = [TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,TCP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP,UDP]