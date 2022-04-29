# This is a sample Python script.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from numpy import asarray
from numpy import savetxt
import os

def parseFile(filein,nevents):

        with open(filein) as f:
            lines = f.readlines()

        header = lines.pop(0).strip()
        pixelstats = lines.pop(0).strip()

        print("Header: ", header)
        print("Pixelstats: ", pixelstats)

        clusterctr = 0
        b_getclusterinfo = False
        cluster_truth =[]
        timeslice = 0
        # instantiate 4-d np array [cluster number, time slice, pixel row, pixel column]
        cur_slice = []
        cur_cluster = []
        events = []

        for line in lines:
            #uncomment line.strip() to get the print out of each one
            #print(line.strip())
            ## Get cluster truth information
            if "<cluster>" in line:
                # save the last time slice too
                if timeslice > 0: cur_cluster.append(cur_slice)
                cur_slice = []
                timeslice = 0

                b_getclusterinfo = True

                # save the last cluster
                if clusterctr > 0:
                    # print("len of cur_cluster = ", len(cur_cluster))
                    events.append(cur_cluster)
                cur_cluster = []
                #print("New cluster ",clusterctr)
                clusterctr += 1

                # Let's just look at the first 10 clusters for now
                if clusterctr > nevents: break
                continue

            if b_getclusterinfo:
                cluster_truth.append(line.strip().split())
                b_getclusterinfo = False

            ## Put cluster information into np array
            if "time slice" in line:
                print("time slice ", timeslice, ", ", line.strip())
                print("Length of cur_slice = ", len(cur_slice))
                if timeslice > 0 and timeslice < 8: cur_cluster.append(cur_slice)
                cur_slice = []
                timeslice += 1
                continue
            if timeslice > 0 and b_getclusterinfo == False:
                cur_row = line.strip().split()
                # print(len(cur_row))
                cur_slice.append([10*float(item) for item in cur_row])

        print("Number of clusters = ", clusterctr)
        print(cluster_truth)
        print("Number of events = ",len(events))
        print("Number of time slices in cluster = ", len(events[0]))

        arr_truth = np.array(cluster_truth)
        arr_events = np.array( events )

        #convert into pandas DF
        df = {}
        #truth quantities - all are dumped to DF
        df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs'])
        df['n_x']=df['n_x'].astype(float)
        df['n_y']=df['n_y'].astype(float)
        df['n_z']=df['n_z'].astype(float)
        df['cotBeta'] = df['n_y']/df['n_z']
        df.drop('x-entry', axis=1, inplace=True)
        df.drop('y-entry', axis=1, inplace=True)
        df.drop('z-entry', axis=1, inplace=True)
        df.drop('n_x', axis=1, inplace=True)
        df.drop('n_y', axis=1, inplace=True)
        df.drop('n_z', axis=1, inplace=True)
        df.drop('number_eh_pairs', axis=1, inplace=True)

        df.to_csv("labels.csv", index=False)

        return arr_events, arr_truth

def main():

    arr_events, arr_truth = parseFile(filein="pixel_clusters_d00000.out", nevents=10)

    print("The shape of the event array: ", arr_events.shape)
    print("The ndim of the event array: ", arr_events.ndim)
    print("The dtype of the event array: ", arr_events.dtype)
    print("The size of the event array: ", arr_events.size)
    print("The max value in the array is: ", np.amax(arr_events))
    # print("The shape of the truth array: ", arr_truth.shape)

    for i, e in enumerate(arr_events):
        #integrated_cluster = np.sum(e,axis=0)
        print("event number = ", i)
        print("event array shape = ", e.shape)

        os.chdir('/Users/jieunyoo/april12_3dCNN/figures')
        path=os.getcwd()
        os.mkdir('cluster'+str(i))
        os.chdir('cluster'+str(i))

        max_val = np.amax(e)
        for j,s in enumerate(e):
            np.save('event{0:006d}_frame{1:02d}.npy'.format(i,j),s)
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
