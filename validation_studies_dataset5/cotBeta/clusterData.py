import numpy as np
import pandas as pd
from pandas import read_csv
import math
import sys
from csv import writer
from scipy.stats import sem

arg1, arg2 = sys.argv[1], sys.argv[2]

df1 = pd.read_csv('cotBeta.csv')

def counterNonZeroRows(X):
    countRows = 0
    for i in X:
        checkOne = False
        for j in i:
            #not there was an extra factor of 10, added in to the original dataset, so had to use 8000 instead of 800 to fix that factor
            #if j >= 800:
            if j >= 8000:
                checkOne = True
        if checkOne == True:
            countRows +=1
    #print(countRows)
    return countRows

def Average(lst):
    return sum(lst) / len(lst)


df2 = df1[df1['cotBeta'] < float(arg2)]
df3 = df2[df2['cotBeta'] > float(arg1)]

df3 = df3.drop(['cotBeta'], axis=1)
df3 = df3.reset_index(drop=True)

df3.to_csv('dfThreshold.csv',index=False)
dfThreshold = pd.read_csv('dfThreshold.csv')
dfThreshold.head()

#LOOP IS HERE
list1 = []
for index, row in dfThreshold.iterrows():
    X = row.values
    X = np.reshape(X,(13,21))
    #X = np.transpose(X)
    nonZeroRows = counterNonZeroRows(X)
    list1.append(nonZeroRows)

print(len(list1))
print(Average(list1))
print(sem(list1))

list_data = []
list_data.append(arg1)
list_data.append(arg2)
list_data.append(len(list1))
list_data.append(Average(list1))
list_data.append(sem(list1))

with open('clusterData.csv', 'a', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(list_data)
    f_object.close()
