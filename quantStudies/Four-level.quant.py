import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

df1 = pd.read_csv('recon650k.csv')
def replace(row):
  for i, item in enumerate(row):
     if row[i] < 800:
        row[i] = 0
     elif ( row[i] >= 800 and row[i] <= 2250):
        row[i] = 1 
     elif ( row[i] > 2250 and row[i] < 5400):
        row[i] = 2
     else:
        row[i] = 3
  return row

df1 = df1.progress_apply(lambda row : replace(row)) 
df1.head()

df1.to_csv('quantized650k.csv', index=False)

