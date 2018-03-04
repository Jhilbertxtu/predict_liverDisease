from matplotlib import pyplot
from pandas import read_csv
import numpy as np
#loading the dataset
filename = 'Indian Liver Patient Dataset (ILPD).csv'
names = ['age','gender','total bilirubin','direct bilirubin','alkaline phosphotase','Alamine Aminotransferase','Aspartate Aminotransferase','Total Protiens','Albumin','Albumin and Globulin Ratio','Dataset']
data = read_csv(filename, names=names)

#Histogram for the features
data.hist()
pyplot.show()
#some of the features assume gaussian distribution while others have exponential distribution

#Density plots
data.plot(kind='density', subplots=True, sharex=False)
pyplot.show()

#correlation heatmap
correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

"""
from the correlation heatmap it is observed that the following features are highly correlated 
Direct_Bilirubin & Total_Bilirubin
Aspartate_Aminotransferase & Alamine_Aminotransferase
Total_Protiens & Albumin
Albumin_and_Globulin_Ratio & Albumin
"""