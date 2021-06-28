import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
dataset = pd.read_csv('h213c.csv')
x=[]
count=dataset['OMXP19X'].sum(axis=0)
print(count)