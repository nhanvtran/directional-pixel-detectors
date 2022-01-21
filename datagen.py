# This is a sample Python script.

import matplotlib.pyplot as plt
import numpy as np

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
            # print(line.strip())
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
                print("New cluster ",clusterctr)
                clusterctr += 1

                # Let's just look at the first 10 clusters for now
                if clusterctr > nevents: break

                continue

            if b_getclusterinfo:
                cluster_truth.append(line.strip().split())
                b_getclusterinfo = False

            ## Put cluster information into np array
            if "time slice" in line:
                # print("time slice ", timeslice, ", ", line.strip())
                # print("Length of cur_slice = ", len(cur_slice))
                if timeslice > 0: cur_cluster.append(cur_slice)
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

        integrated_cluster = np.sum(e,axis=0)
        print("event number = ", i)
        print("event array shape = ", e.shape)
        print("integrated cluster shape = ", integrated_cluster.shape)
        print("total charge = ", np.sum(integrated_cluster))

        cur_img = plt.imshow(integrated_cluster)
        cur_img.get_figure().savefig('figures/integ_ev{0:02d}.png'.format(i))

        max_val = np.amax(e)
        for j,s in enumerate(e):

            cur_img = plt.imshow(s,vmin=0,vmax=max_val)
            cur_img.get_figure().savefig('figures/slices_ev{0:02d}_sl{1:02d}.png'.format(i,j))


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
