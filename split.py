import sys
import numpy as np
import pandas as pd
import math

def split(index):

        df1 = pd.read_csv("labels_d"+str(index)+".csv")
        df2 = pd.read_csv("recon_d"+str(index)+".csv")

        print(len(df1),len(df2))

        # unflipped, pos charge
        df1[(df1['z-entry']==0) & (df1['pt']>0)].to_csv("unflipped-positive/labels_d"+str(index)+".csv", index=False)
        df2[(df1['z-entry']==0) & (df1['pt']>0)].to_csv("unflipped-positive/recon_d"+str(index)+".csv", index=False)

        # unflipped, neg charge                                                                                                                                           
        df1[(df1['z-entry']==0) & (df1['pt']<0)].to_csv("unflipped-negative/labels_d"+str(index)+".csv", index=False)
        df2[(df1['z-entry']==0) & (df1['pt']<0)].to_csv("unflipped-negative/recon_d"+str(index)+".csv", index=False)
        
        # flipped, pos charge                                                                                                                                           
        df1[(df1['z-entry']==100) & (df1['pt']>0)].to_csv("flipped-positive/labels_d"+str(index)+".csv", index=False)
        df2[(df1['z-entry']==100) & (df1['pt']>0)].to_csv("flipped-positive/recon_d"+str(index)+".csv", index=False)

        # flipped, neg charge                                                                                                                                           
        df1[(df1['z-entry']==100) & (df1['pt']<0)].to_csv("flipped-negative/labels_d"+str(index)+".csv", index=False)
        df2[(df1['z-entry']==100) & (df1['pt']<0)].to_csv("flipped-negative/recon_d"+str(index)+".csv", index=False)

def main():
        
        for i in range(16501,16601):
                split(i)

if __name__ == "__main__":
    main()
