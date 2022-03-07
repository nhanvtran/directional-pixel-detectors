import sys
# sys.path.append(".")

DATA_DIR = sys.path[-1] + "/data"

DATA_FILE_NAME = "pixel_clusters_d00000.out"

IMAGE_DIR = sys.path[-1] + "/images"

LABELS_FILE = DATA_DIR + "/" + "labels650K.csv"

CLUSTERS_FILE = DATA_DIR + "/" + "recon650K.csv"

NEVENTS = 650000