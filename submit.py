#!/usr/bin/python

# Submission script for parallel processing of Morris' .gz files on lxplus
# For a new dataset, make sure the range of integers below matches the file names

import os, sys
import subprocess
import argparse

# Main method                                                                          
def main():

    os.system('mkdir -p logs')

    for i in range(16501,16601):

        # Recreate the condor submission file                                                                   
        subfile = "logs/d"+str(i)+".sub"
        if os.path.isfile(subfile):
            os.remove(subfile)

        f = open(subfile,"w")
        
        f.write("universe                = vanilla \n")
        f.write("executable              = datagen.sh \n")
        f.write("arguments               = "+str(i)+" \n")
        f.write("request_memory          = 4000 \n")
        f.write("should_transfer_files   = YES \n")
        f.write("when_to_transfer_output = ON_EXIT \n")
        f.write("transfer_input_files    = datagen.py \n")
        f.write("transfer_output_files   = labels_d"+str(i)+".csv,recon_d"+str(i)+".csv \n")                                              
        f.write("output                  = logs/d"+str(i)+".out \n")
        f.write("error                   = logs/d"+str(i)+".err \n")
        f.write("log                     = logs/d"+str(i)+".log \n")
        
        # Job flavour determines job wall time                                                                  
        # https://batchdocs.web.cern.ch/local/submit.html#job-flavours                                          
        f.write("+JobFlavour             = \"nextweek\" \n")
        f.write("queue \n")
        
        f.close()
    
        # submit the job                                                                                        
        os.system("condor_submit " + subfile)
    
    return 

if __name__ == "__main__":
    main()
