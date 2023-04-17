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
import numpy as np
import pandas as pd


# path to the Million Song Dataset data (uncompressed)
msd_subset_data_path='data/MillionSongSubset'
assert os.path.isdir(msd_subset_data_path),'wrong path' # sanity check
# path to the Million Song Dataset code
msd_code_path='tools/MSD_python_code'
assert os.path.isdir(msd_code_path),'wrong path' # sanity check
# add tools to Python path
sys.path.append( msd_code_path)

# Read in Labeled_Data.csv
assert os.path.isfile('data_processed/Labeled_Data.csv'), 'file does not exist' # sanity check
labeled_data_file = glob.glob('data_processed/Labeled_Data.csv')
labeled_data = pd.read_csv(labeled_data_file[0])

# imports specific to the MSD
import hdf5_getters as GETTERS


# Need to dictionary to house all of the desired features globally
desired_columns = {'Genre': []}
for i in range(1, 13):
    desired_columns['Segment Pitch ' + str(i) + ' Average'] = []
    desired_columns['Segment Timbre ' + str(i) + ' Average'] = []
    for j in range(i, 13):
        desired_columns['Segment Pitch ' + str(i) + '-' + str(j) + ' Covariance'] = []
        desired_columns['Segment Timbre ' + str(i) + '-' + str(j) + ' Covariance'] = []

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

def func_to_get_desired_features(filename):
    """
    This function does 3 simple things:
    - open the song file
    - get desired features and put them in a pandas dataframe
    - close the file
    """
    h5 = GETTERS.open_h5_file_read(filename)

    artist_name = GETTERS.get_artist_name(h5)
    title = GETTERS.get_title(h5)
    segment_pitches = GETTERS.get_segments_pitches(h5)
    segment_timbre = GETTERS.get_segments_timbre(h5)

    if artist_name.decode('UTF-8') in np.array(labeled_data['Artist Name']) and title.decode('UTF-8') in np.array(labeled_data['Song Title']):
        index_artist = [x for x in range(np.size(labeled_data['Artist Name'])) if labeled_data['Artist Name'][x] == artist_name.decode('UTF-8')]
        index_title = [x for x in range(np.size(labeled_data['Song Title'])) if labeled_data['Song Title'][x] == title.decode('UTF-8')]

        for i in index_title:
            if i in index_artist:
                index = i
                break
        
        try:
            desired_columns['Genre'].append(labeled_data['MB Genre'].values[index])
        except UnboundLocalError:
            print(artist_name.decode('UTF-8'), "with song", title.decode('UTF-8'), "not in Labeled_Data.csv")
            return

        # Calculate the average of each segment pitch and segment timbre
        segment_pitches_avg_arr = np.mean(segment_pitches, axis=0)
        segment_timbre_avg_arr = np.mean(segment_timbre, axis=0)

        # Calculate the covariance of segment_pitches and segment_timbre
        segment_pitches_cov_arr = np.cov(segment_pitches, rowvar=False)
        segment_timbre_cov_arr = np.cov(segment_timbre, rowvar=False)

        # Populate columns
        for i in range(1, 13):
            desired_columns['Segment Pitch ' + str(i) + ' Average'].append(segment_pitches_avg_arr[i - 1])
            desired_columns['Segment Timbre ' + str(i) + ' Average'].append(segment_timbre_avg_arr[i - 1])
            for j in range(i, 13):
                desired_columns['Segment Pitch ' + str(i) + '-' + str(j) + ' Covariance'].append(segment_pitches_cov_arr[i-1][j-1])
                desired_columns['Segment Timbre ' + str(i) + '-' + str(j) + ' Covariance'].append(segment_timbre_cov_arr[i-1][j-1])

    h5.close()


if __name__ == '__main__':
    # we can now easily count the number of files in the dataset
    num_song_files = apply_to_all_files(msd_subset_data_path)
    print('number of song files:', num_song_files)

    # Extract desired features and store them in a dictionary
    print('Extracting desired features...')
    apply_to_all_files(msd_subset_data_path, func=func_to_get_desired_features)

    # Create pandas dataframe to house desired features
    msd_dataframe = pd.DataFrame(desired_columns)
    
    # Export pandas dataframe to csv file
    print('Exporting to csv file...')
    msd_dataframe.to_csv('data_processed/MSD_Desired_Features_With_Labels.csv')

    