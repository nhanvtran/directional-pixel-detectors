import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from global_vars import *

def parseFile(filein , nevents) :
    with open ( filein ) as f :
        lines = f.readlines ( )

    header = lines.pop ( 0 ).strip ( )
    pixelstats = lines.pop ( 0 ).strip ( )

    print ( "Header: " , header )
    print ( "Pixelstats: " , pixelstats )

    clusterctr = 0
    t = 0
    b_getclusterinfo = False
    cluster_truth = []
    timeslice = 0
    # instantiate 4-d np array [cluster number, time slice, pixel row, pixel column]
    cur_slice = []
    cur_cluster = []
    events = []

    for line in lines :
        # print(line.strip())
        ## Get cluster truth information
        if "<cluster>" in line :
            # save the last time slice too
            if timeslice > 0 : cur_cluster.append ( cur_slice )
            cur_slice = []
            timeslice = 0

            b_getclusterinfo = True
            # save the last cluster
            if clusterctr > 0 :
                # print("len of cur_cluster = ", len(cur_cluster))
                events.append ( cur_cluster )
            cur_cluster = []
            print ( "New cluster " , clusterctr )
            clusterctr += 1

            # Let's just look at the first 10 clusters for now
            if clusterctr > nevents : break
            continue

        if b_getclusterinfo :
            cluster_truth.append ( line.strip ( ).split ( ) )
            b_getclusterinfo = False

        ## Put cluster information into np array
        if "time slice" in line :
            # print("time slice ", timeslice, ", ", line.strip())
            # print("Length of cur_slice = ", len(cur_slice))
            if timeslice > 0 : cur_cluster.append ( cur_slice )
            cur_slice = []
            timeslice += 1
            continue
        if timeslice > 0 and b_getclusterinfo == False :
            cur_row = line.strip ( ).split ( )
            # print(len(cur_row))
            cur_slice.append ( [10 * float ( item ) for item in cur_row] )

    print ( "Number of clusters = " , clusterctr )
    print ( cluster_truth )
    print ( "Number of events = " , len ( events ) )
    print ( "Number of time slices in cluster = " , len ( events[0] ) )

    arr_truth = np.array ( cluster_truth )
    arr_events = np.array ( events )

    # convert into pandas DF
    df = {}
    # truth quantities - all are dumped to DF
    df = pd.DataFrame ( arr_truth ,
                        columns = ['x-entry' , 'y-entry' , 'z-entry' , 'n_x' , 'n_y' , 'n_z' , 'number_eh_pairs'] )
    df['n_x'] = df['n_x'].astype ( float )
    df['n_y'] = df['n_y'].astype ( float )
    df['n_z'] = df['n_z'].astype ( float )

    # added angular variables
    # df['spherR'] = df['n_x']**2 + df['n_y']**2 + df['n_z']**2
    # df['theta'] = np.arccos(df['n_z']/df['spherR'])*180/math.pi
    # df['phi'] = np.arctan2(df['n_y'],df['n_x'])*180/math.pi
    # df['cosPhi'] = np.cos(df['phi'])
    df['cotAlpha'] = df['n_x'] / df['n_z']
    df['cotBeta'] = df['n_y'] / df['n_z']
    df.to_csv ( DATA_DIR + "/" + "labels_{}.csv".format(nevents) , index = False )
    dfAngles = df[['cotAlpha','cotBeta']]
    dfAngles.to_csv ( DATA_DIR + "/"+ "angles_{}.csv".format(nevents) , index = False )

    np.savez(DATA_DIR + "/" + "events_{}.npz".format(nevents), arr_events)
    np.savez(DATA_DIR + "/" + "truth_{}.npz".format(nevents), arr_truth)

    return arr_events , arr_truth

def get_recon_csv(arr_events):

    print ( "The shape of the event array: " , arr_events.shape )
    print ( "The ndim of the event array: " , arr_events.ndim )
    print ( "The dtype of the event array: " , arr_events.dtype )
    print ( "The size of the event array: " , arr_events.size )
    print ( "The max value in the array is: " , np.amax ( arr_events ) )

    events_list = []
    for i, e in enumerate(arr_events):
        integrated_cluster = np.sum ( e , axis = 0 )
        print ( "event number = " , i )
        print ( "event array shape = " , e.shape )
        print ( "integrated cluster shape = " , integrated_cluster.shape )
        print ( "total charge = " , np.sum ( integrated_cluster ) )

        events_list.append(integrated_cluster.ravel())

    # df is a df with the reconstructed clusters
    df = pd.DataFrame ( events_list )
    df.to_csv ( DATA_DIR + '/reconstructed_{}.csv'.format(NEVENTS) , index = False )

