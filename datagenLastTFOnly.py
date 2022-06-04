import numpy as np
import pandas as pd

global DATA_DIR
DATA_DIR = "./download/processed"

def parseFile(filein , tag , nevents = -1) :
    with open ( filein ) as f :
        lines = f.readlines ( )

    header = lines.pop ( 0 ).strip ( )
    pixelstats = lines.pop ( 0 ).strip ( )

    print ( "Header: " , header )
    print ( "Pixelstats: " , pixelstats )

    clusterctr = 0
    b_getclusterinfo = False
    cluster_truth = []
    timeslice = 0
    # instantiate 4-d np array [cluster number, time slice, pixel row, pixel column]
    cur_slice = []
    cur_cluster = []
    events = []

    for line in lines :

        if len ( events ) >= nevents and nevents > 0 : break

        ## Get cluster truth information
        if "<cluster>" in line :

            # save the last time slice too
            if timeslice > 0 : cur_cluster.append ( cur_slice )
            cur_slice = []
            timeslice = 0

            b_getclusterinfo = True

            # save the last cluster
            if clusterctr > 0 :
                events.append ( cur_cluster )

            clusterctr += 1
            cur_cluster = []
            print ( "New cluster " , clusterctr )
            continue

        # the line after cluster
        if b_getclusterinfo :
            cluster_truth.append ( line.strip ( ).split ( ) )
            b_getclusterinfo = False

        ## Put cluster information into np array
        if "time slice" in line :
            if timeslice > 0 : cur_cluster.append ( cur_slice )
            cur_slice = []
            timeslice += 1
            continue
        if timeslice > 0 and b_getclusterinfo == False :
            cur_row = line.strip ( ).split ( )
            cur_slice.append ( [10 * float ( item ) for item in cur_row] )

    events.append ( cur_cluster )
    print ( "Number of clusters = " , len ( cluster_truth ) )
    print ( "Number of events = " , len ( events ) )
    print ( "Number of time slices in cluster = " , len ( events[0] ) )

    arr_truth = np.array ( cluster_truth )
    arr_events = np.array ( events )

    # convert into pandas DF
    df = {}
    # truth quantities - all are dumped to DF
    df = pd.DataFrame ( arr_truth ,
                        columns = ['x-entry' , 'y-entry' , 'z-entry' , 'n_x' , 'n_y' , 'n_z' , 'number_eh_pairs' ,
                                   'y-local' , 'pt'] )
    df['n_x'] = df['n_x'].astype ( float )
    df['n_y'] = df['n_y'].astype ( float )
    df['n_z'] = df['n_z'].astype ( float )

    df['cotAlpha'] = df['n_x'] / df['n_z']
    df['cotBeta'] = df['n_y'] / df['n_z']
    df.to_csv ( DATA_DIR + '/' + "labels_LT_" + tag + ".csv" , index = False )

    return arr_events , arr_truth


def get_recon_csv(arr_events , tag) :
    print ( "The shape of the event array: " , arr_events.shape )
    print ( "The ndim of the event array: " , arr_events.ndim )
    print ( "The dtype of the event array: " , arr_events.dtype )
    print ( "The size of the event array: " , arr_events.size )
    print ( "The max value in the array is: " , np.amax ( arr_events ) )

    events_list = []
    for i , e in enumerate ( arr_events ) :
        latest_time_slice_info = np.array ( e )[-1 : , : , :]
        print ( "event number = " , i )
        events_list.append ( latest_time_slice_info.flatten ( ) )

    df = pd.DataFrame ( events_list )
    df.to_csv ( DATA_DIR + '/' + 'recon_LT_{}.csv'.format ( tag ) , index = False )

