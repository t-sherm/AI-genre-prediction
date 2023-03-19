#!/usr/bin/env python
"""
This script will extract the desired features from the MSD data
and output them into a csv file.
Ensure that the following features are extracted so that the data
can be fed into the mb_label_creator.py

Notes: 
1) Make sure that you run this script in the AI-genre-prediction directory.
2) Make sure that you have a local copy of the MillionSongSubset and put it in
   a folder called: AI-genre-prediction/data.
"""

# usual imports
import os
import sys
import time
import glob
import datetime
import sqlite3
import numpy as np


# path to the Million Song Dataset data (uncompressed)
msd_subset_data_path='data/MillionSongSubset'
assert os.path.isdir(msd_subset_data_path),'wrong path' # sanity check
# path to the Million Song Dataset code
msd_code_path='tools/MSD_python_code'
import pdb; pdb.set_trace()
assert os.path.isdir(msd_code_path),'wrong path' # sanity check
# add tools to Python path
sys.path.append( msd_code_path)

# imports specific to the MSD
import hdf5_getters as GETTERS


def strtimedelta(starttime,stoptime):
    """
    The following function simply gives us a nice string for
    a time lag in seconds.
    INPUT
        starttime
        stoptime
    OUTPUT
        time difference (sec)
    """
    return str(datetime.timedelta(seconds=stoptime-starttime))

def apply_to_all_files(basedir,func=lambda x: x,ext='.h5'):
    """
    From a base directory, go through all subdirectories,
    find all files with the given extension, apply the
    given function 'func' to all of them.
    If no 'func' is passed, we do nothing except counting.
    INPUT
       basedir  - base directory of the dataset
       func     - function to apply to all filenames
       ext      - extension, .h5 by default
    RETURN
       number of files
    """
    cnt = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        # count files
        cnt += len(files)
        # apply function to all files
        for f in files :
            func(f)       
    return cnt

def func_to_get_artist_name(filename):
    """
    This function does 3 simple things:
    - open the song file
    - get artist ID and put it
    - close the file
    """
    h5 = GETTERS.open_h5_file_read(filename)
    artist_name = GETTERS.get_artist_name(h5)
    all_artist_names.add( artist_name )
    h5.close()


if __name__ == '__main__':
    # we can now easily count the number of files in the dataset
    print('number of song files:',apply_to_all_files(msd_subset_data_path))

    # let's now get all artist names in a set(). One nice property:
    # if we enter many times the same artist, only one will be kept.
    all_artist_names = set()
        
    # let's apply the previous function to all files
    # we'll also measure how long it takes
    t1 = time.time()
    apply_to_all_files(msd_subset_data_path,func=func_to_get_artist_name)
    t2 = time.time()
    print('all artist names extracted in:',strtimedelta(t1,t2))

    # let's see some of the content of 'all_artist_names'
    print('found',len(all_artist_names),'unique artist names')
    for k in range(5):
        print(list(all_artist_names)[k])
