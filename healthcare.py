import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
dataset = pd.read_csv('h213c.csv')
x=[]
count=2402
for i in range(0,int(count)) :
    x.insert(i,0)
count=1919
for i in range(0,int(count)) :
    x.insert(i,1)
count=9392
for i in range(0,int(count)) :
    x.insert(i,2)
count=1835
for i in range(0,int(count)) :
    x.insert(i,3)
count=1000
for i in range(0,int(count)) :
    x.insert(i,4)
count=6567
for i in range(0,int(count)) :
    x.insert(i,5)
count=9838
for i in range(0,int(count)) :
    x.insert(i,6)
num_bins=7
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7), 
                        tight_layout = True)
axs.hist(x, bins = num_bins)
axs.xaxis.set_tick_params(pad = 5) 
axs.yaxis.set_tick_params(pad = 10) 

axs.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.6) 
N, bins, patches = axs.hist(x, bins = num_bins)
  
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())
  
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

rects = axs.patches
labels = ['OMSF19X', 'OMMR19X', 'OMMD19X', 'OMPV19X', 'OMXP19X', 'OMTC19X', 'OMOT19X'] 
for rect, label in zip(rects, labels):
    height = rect.get_height()
    axs.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
            ha='center', va='bottom')

plt.xlabel("Diseases")
plt.ylabel("Amount of Expenditure")
plt.title('Histogram')
  
# Show plot
plt.show()

labels = ['OMSF19X', 'OMMR19X', 'OMMD19X', 'OMPV19X', 'OMXP19X', 'OMTC19X', 'OMOT19X'] 
sizes = [2402, 1919, 9392, 1835, 10005, 6567, 9838]

explode = (0, 0, 0, 0.1, 0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=30)
plt.show()