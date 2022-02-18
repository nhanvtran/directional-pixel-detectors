import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import seaborn as sns
import numpy as np

df = pd.read_csv("recon650k.csv")

df.head()

#plotting every entry
sns.distplot(df, kde=False)
plt.xlabel('charge')
plt.ylabel('frequency')
plt.title("charge")

print(df.shape[1])
print(df.shape[0])

n = 177450000
X = df.values
X = np.reshape(X, (n,1))

df2 = pd.DataFrame(X)
df2.describe()

above800 = df2[df2 > 800]

above800.describe()

sns.distplot(above800, kde=False)
plt.xlabel('charge')
plt.ylabel('frequency')
plt.title("charge")

below20000 = above800[above800 < 20000]
below20000.describe()

sns.distplot(below20000, kde=False)
plt.xlabel('charge')
plt.ylabel('frequency')
plt.title("charge")

