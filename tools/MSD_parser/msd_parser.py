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
msd_data_path='data/MillionSongDataFinal'
assert os.path.isdir(msd_data_path),'wrong path' # sanity check
# path to the Million Song Dataset code
msd_code_path='tools/MSD_python_code'
assert os.path.isdir(msd_code_path),'wrong path' # sanity check
# add tools to Python path
sys.path.append( msd_code_path)

# Read in csv with labeled data in preparation to grab more features from h5 files
assert os.path.isfile('data_processed/Iteration_1/Labeled_Data_Final.csv'), 'file does not exist' # sanity check
labeled_data_file = glob.glob('data_processed/Iteration_1/Labeled_Data_Final.csv')
labeled_data = pd.read_csv(labeled_data_file[0])

# imports specific to the MSD
import hdf5_getters as GETTERS


# Need a dictionary to house all of the desired features globally
desired_columns = {#'Genre': [],
                   'Artist Name': [],
                   'Release Name': [],
                   'Song Title': [],
                   'Duration': [],
                   'Key': [],
                   'Mode': [],
                   'Loudness': [],
                   'Tempo': [],
                   'Time Signature': [],
                   'End of Fade In': [],
                   'Danceability': [],
                   'Energy': [],
                   'Song Hotness': [],
                   'Start of Fade Out': []}
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
        # This was added to ensure that we are using the same amount of data as the first iteration
        if len(desired_columns['Artist Name']) > 89502:
            break
        files = glob.glob(os.path.join(root,'*'+ext))
        # count files
        cnt += len(files)
        # apply function to all files
        for f in files :
            if len(desired_columns['Artist Name']) > 89502:
                break
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
    release = GETTERS.get_release(h5)
    title = GETTERS.get_title(h5)
    duration = GETTERS.get_duration(h5)
    key = GETTERS.get_key(h5)
    mode = GETTERS.get_mode(h5)
    loudness = GETTERS.get_loudness(h5)
    tempo = GETTERS.get_tempo(h5)
    time_signature = GETTERS.get_time_signature(h5)
    end_of_fade_in = GETTERS.get_end_of_fade_in(h5)
    danceability = GETTERS.get_danceability(h5)
    energy = GETTERS.get_energy(h5)
    song_hotness = GETTERS.get_artist_hotttnesss(h5)
    start_of_fade_out = GETTERS.get_start_of_fade_out(h5)
    segment_pitches = GETTERS.get_segments_pitches(h5)
    segment_timbre = GETTERS.get_segments_timbre(h5)
        
    if artist_name.isascii() and release.isascii():
        # Get desired features
        desired_columns['Artist Name'].append(artist_name.decode('UTF-8'))
        desired_columns['Release Name'].append(release.decode('UTF-8'))
        desired_columns['Song Title'].append(title.decode('UTF-8'))
        desired_columns['Duration'].append(duration)
        desired_columns['Key'].append(key)
        desired_columns['Mode'].append(mode)
        desired_columns['Loudness'].append(loudness)
        desired_columns['Tempo'].append(tempo)
        desired_columns['Time Signature'].append(time_signature)
        desired_columns['End of Fade In'].append(end_of_fade_in)
        desired_columns['Danceability'].append(danceability)
        desired_columns['Energy'].append(energy)
        desired_columns['Song Hotness'].append(song_hotness)
        desired_columns['Start of Fade Out'].append(start_of_fade_out)

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
            # Only take the non-repeating values in the covariance matrix of each song
            for j in range(i, 13):
                desired_columns['Segment Pitch ' + str(i) + '-' + str(j) + ' Covariance'].append(segment_pitches_cov_arr[i-1][j-1])
                desired_columns['Segment Timbre ' + str(i) + '-' + str(j) + ' Covariance'].append(segment_timbre_cov_arr[i-1][j-1])

    h5.close()


if __name__ == '__main__':
    # we can now easily count the number of files in the dataset
    num_song_files = apply_to_all_files(msd_data_path)
    print('number of song files:', num_song_files)

    # Extract desired features and store them in a dictionary
    print('Extracting desired features...')
    apply_to_all_files(msd_data_path, func=func_to_get_desired_features)

    # Create pandas dataframe to house desired features
    msd_dataframe = pd.DataFrame(desired_columns)
    
    # Export pandas dataframe to csv file
    print('Exporting to csv file...')
    msd_dataframe.to_csv('data_processed/Iteration_2/MSD_Desired_Features.csv')

    