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

# imports specific to the MSD
import hdf5_getters as GETTERS


# Need to dictionary to house all of the desired features globally
desired_features = {'Artist Name': [],
                    'Release': [],
                    'Song Title': [],
                    'Artist Location': [],
                    'Duration (sec)': [],
                    'Key': [],
                    'Key Confidence': [],
                    'Mode': [],
                    'Mode Confidence': [],
                    'Loudness': [],
                    'Tempo': [],
                    'Time Signature': [],
                    'Time Signature Confidence': [],
                    'End of Fade In': [],
                    'Interval 1 Count': [],
                    'Interval 2 Count': [],
                    'Interval 3 Count': [],
                    'Interval 4 Count': [],
                    'Interval 5 Count': [],
                    'Interval 6 Count': [],
                    'Interval 7 Count': [],
                    'Interval 8 Count': [],
                    'Interval 9 Count': [],
                    'Interval 10 Count': [],
                    'Interval 11 Count': [],
                    'Interval 12 Count': []}


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
    - get artist name and put it in a pandas dataframe
    - close the file
    """
    h5 = GETTERS.open_h5_file_read(filename)

    artist_name = GETTERS.get_artist_name(h5)
    release = GETTERS.get_release(h5)
    title = GETTERS.get_title(h5)
    artist_location = GETTERS.get_artist_location(h5)
    duration = GETTERS.get_duration(h5)
    key = GETTERS.get_key(h5)
    key_confidence = GETTERS.get_key_confidence(h5)
    mode = GETTERS.get_mode(h5)
    mode_confidence = GETTERS.get_mode_confidence(h5)
    loudness = GETTERS.get_loudness(h5)
    tempo = GETTERS.get_tempo(h5)
    time_signature = GETTERS.get_time_signature(h5)
    time_signature_confidence = GETTERS.get_time_signature_confidence(h5)
    end_of_fade_in = GETTERS.get_end_of_fade_in(h5)
    segment_pitches = GETTERS.get_segments_pitches(h5)

    if artist_name.isascii() and release.isascii():
        desired_features['Artist Name'].append(artist_name.decode('UTF-8'))
        desired_features['Release'].append(release.decode('UTF-8'))
        desired_features['Song Title'].append(title.decode('UTF-8'))
        desired_features['Artist Location'].append(artist_location.decode('UTF-8'))
        desired_features['Duration (sec)'].append(duration)
        desired_features['Key'].append(key)
        desired_features['Key Confidence'].append(key_confidence)
        desired_features['Mode'].append(mode)
        desired_features['Mode Confidence'].append(mode_confidence)
        desired_features['Loudness'].append(loudness)
        desired_features['Tempo'].append(tempo)
        desired_features['Time Signature'].append(time_signature)
        desired_features['Time Signature Confidence'].append(time_signature_confidence)
        desired_features['End of Fade In'].append(end_of_fade_in)

        # Calculate the intervals and count them using segment pitches
        intervals_arr = np.zeros(12) # Create intervals array
        index_locations = []

        for i in range(np.shape(segment_pitches)[0]):
            index_locations.append(np.where(segment_pitches[i] == 1.0)[0][0])
        index_locations = np.asarray(index_locations)

        for i in range(np.size(index_locations)):
            if (i == np.size(index_locations) - 1):
                break
            intervals_arr[np.abs(index_locations[i+1] - index_locations[i])] += 1

        # Populate Interval Count columns
        desired_features['Interval 1 Count'].append(intervals_arr[0])
        desired_features['Interval 2 Count'].append(intervals_arr[1])
        desired_features['Interval 3 Count'].append(intervals_arr[2])
        desired_features['Interval 4 Count'].append(intervals_arr[3])
        desired_features['Interval 5 Count'].append(intervals_arr[4])
        desired_features['Interval 6 Count'].append(intervals_arr[5])
        desired_features['Interval 7 Count'].append(intervals_arr[6])
        desired_features['Interval 8 Count'].append(intervals_arr[7])
        desired_features['Interval 9 Count'].append(intervals_arr[8])
        desired_features['Interval 10 Count'].append(intervals_arr[9])
        desired_features['Interval 11 Count'].append(intervals_arr[10])
        desired_features['Interval 12 Count'].append(intervals_arr[11])

    h5.close()


if __name__ == '__main__':
    # we can now easily count the number of files in the dataset
    num_song_files = apply_to_all_files(msd_subset_data_path)
    print('number of song files:', num_song_files)

    # Extract desired features and store them in a dictionary
    print('Extracting desired features...')
    apply_to_all_files(msd_subset_data_path, func=func_to_get_desired_features)

    # Create pandas dataframe to house desired features
    msd_dataframe = pd.DataFrame(desired_features)
    
    # Export pandas dataframe to csv file
    print('Exporting to csv file...')
    msd_dataframe.to_csv('data_processed/MSD_Desired_Features.csv')

    