import numpy as np
import pandas as pd
import math
from numpy import asarray
import os
import sys

def parseFile(filein,tag,nevents=-1):

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

            if len(events) >= nevents and nevents > 0: break

            ## Get cluster truth information
            if "<cluster>" in line:

                # save the last time slice too
                if timeslice > 0: cur_cluster.append(cur_slice)
                cur_slice = []
                timeslice = 0

                b_getclusterinfo = True

                # save the last cluster
                if clusterctr > 0:
                    events.append(cur_cluster)
                    
                cur_cluster = []
                clusterctr += 1
                continue

            # the line after cluster
            if b_getclusterinfo:
                cluster_truth.append(line.strip().split())
                b_getclusterinfo = False

            ## Put cluster information into np array
            if "time slice" in line:
                if timeslice > 0 and timeslice < 9: cur_cluster.append(cur_slice)
                cur_slice = []
                timeslice += 1
                continue
            if timeslice > 0 and timeslice < 9 and b_getclusterinfo == False:
                cur_row = line.strip().split()
                cur_slice.append([float(item) for item in cur_row])
                #print('timeslice', timeslice, 'cur_row', cur_row)
                
        events.append(cur_cluster)
        #print('events', events)
        print("Number of clusters = ", clusterctr)
        print(cluster_truth)
        print("Number of events = ",len(events))
        print("Number of time slices in cluster = ", len(events[0]))
        arr_truth = np.array(cluster_truth)
        arr_events = np.array( events )
        #print('array of events', arr_events)
        #convert into pandas DF
        df = {}
        #truth quantities - all are dumped to DF
        #df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs'])
        df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'pt'])
        df['n_x']=df['n_x'].astype(float)
        df['n_y']=df['n_y'].astype(float)
        df['n_z']=df['n_z'].astype(float)
        df['cotBeta'] = df['n_y']/df['n_z']
        df['cotAlpha'] = df['n_x']/df['n_z']
        df.drop('x-entry', axis=1, inplace=True)
        df.drop('y-entry', axis=1, inplace=True)
        df.drop('n_x', axis=1, inplace=True)
        df.drop('n_y', axis=1, inplace=True)

        df.to_csv("labels_"+tag+".csv", index=False)
        return arr_events, arr_truth

def main():

    i = int(sys.argv[1])
    tag = "d"+str(i)
    arr_events, arr_truth = parseFile(filein="pixel_clusters_d"+str(i)+".out",tag=tag)

    #arr_events, arr_truth = parseFile(filein="pixel_clusters_d"+str(i)+".out", tag=tag)
    print("The shape of the event array: ", arr_events.shape)
    print("The ndim of the event array: ", arr_events.ndim)
    print("The dtype of the event array: ", arr_events.dtype)
    print("The size of the event array: ", arr_events.size)
    #print("The max value in the array is: ", np.amax(arr_events))
    # print("The shape of the truth array: ", arr_truth.shape)
 
    df2 = {}
    list1 = []
    list2 = []
    b = {}

    print('length or arary', len(arr_events))

    for i, e in enumerate(arr_events):
        print("event number = ", i)
        #os.chdir('/home/jieun201/convert16TS/')
        path=os.getcwd()
        list1 = []
        for i in e:
            #print(i)
            for j in i:
                array1 = (np.sum(j))
                list1.append(array1)

        b = np.array(list1)
        b = b.flatten()
        list2.append(b)
        #print('list2', list2)
        
    df2 = pd.DataFrame(list2)
    print(df2)
    df2.to_csv("recon_"+tag+".csv", index=False)

if __name__ == "__main__":
    main()

