#!/usr/bin/env python
"""A simple script that searches for a release in the MusicBrainz
database and prints out a few details about the first 5 matching release.
    $ python populate_genres.py "Casual" "Fear Itself"

"""
from __future__ import print_function
from __future__ import unicode_literals
import musicbrainzngs
import sys
import csv
import pandas as pd

def set_mb_useragent():
    musicbrainzngs.set_useragent(
        "python-musicbrainzngs-example",
        "0.1",
        "https://github.com/alastair/python-musicbrainzngs/",
    )

def load_mb_genres():
    '''
    Load Genre List
    This list is used to determine if a tag is a genre
    '''
    genre_file = open("mb_genre_list.csv", "r")
    mb_genre_list = list(list(csv.reader(genre_file, delimiter='\n')))
    genre_file.close()
    mb_genre_list = [''.join(ele) for ele in mb_genre_list]
    # print(mb_genre_list[1:5])

    return mb_genre_list

def pull_song_info(csv_file):
    '''
    Pull Artist Name and Release from an existing database
    This info will be used to search the MB database for genre
    '''
    song_search_info = pd.read_csv("../../data_processed/MSD_Desired_Features.csv")
    #song_search_info = pd.read_csv("../../data_processed/MSD_Desired_Features.csv", usecols=["Artist Name", 'Release'])
    return song_search_info

def get_song_genre(artist_name, release_name, mb_genre_list):
    genre_match = 0

    result = musicbrainzngs.search_releases(artist=artist_name,
                                            release=release_name,
                                            limit=1)

    # On success, result is a dictionary with a single key:
    # "release-list", which is a list of dictionaries.
    if not result['release-list']:
        sys.exit("no release found")


    # Grab MBIDs
    artist_mbid = result["release-list"][0]["artist-credit"][0]["artist"]["id"]
    #release_mbid = result["release-list"][0]["id"]
    release_group_mbid = result["release-list"][0]["release-group"]["id"]

    # Show what MB actually found from the search.
    artist_name = result["release-list"][0]["artist-credit"][0]["artist"]["name"]
    release_name = result["release-list"][0]["title"]

    '''
    Search for genre by Album which is called a release group
    If artist genre does not exist, search release
    '''
    release_group = musicbrainzngs.get_release_group_by_id(release_group_mbid,
                     includes=["tags"])


    # Check to see whether the release group has tags
    release_group_tag_exist = 1
    if not ("tag-list" in release_group["release-group"]):
        print("Release group tags non-existent")
        release_group_tag_exist = 0

    # If a release groups has tags, see if a genre exists. If so, store as the song genre
    i = 0
    while not genre_match and release_group_tag_exist:
        tag = release_group["release-group"]["tag-list"][i]["name"]
        genre_match = tag in mb_genre_list

        if genre_match:
            song_genre = tag
            break
        i = i+1
    #print(inter.items())


    if not genre_match:
        artist = musicbrainzngs.get_artist_by_id(artist_mbid,
                         includes=["tags"])

        artist_tag_exist = 1
        if not ("tag-list" in artist["artist"]):
            print("Artist tags non-existent")
            artist_tag_exist = 0


        i = 0
        while not genre_match and artist_tag_exist:
            tag = artist["artist"]["tag-list"][i]["name"]
            genre_match = tag in mb_genre_list
            if genre_match:
                song_genre = tag
                break
            i = i+1

    if not genre_match:
        return artist_name, release_name, 0
    return artist_name, release_name, song_genre

def print_song_info(artist_name, release_name, song_genre):
    print("Artist Name: ", artist_name)
    print("Album Name: ", release_name)
    print("Genre: ", song_genre)

def populate_genres(song_info):
    mb_genre_list = load_mb_genres()
    print(song_info)
    genres = []

    artist_names = song_info["Artist Name"]
    release_names = song_info["Release"]
    print(artist_names.loc[1])
    #print_song_info(artist_names, release_names, 0)
    c = 0
    for i in range(len(song_info.index)):

        try:
            [artist_name, release_name, genre] = get_song_genre(artist_names.loc[i], release_names.loc[i], mb_genre_list)
        except:
            print("Error finding genre for song id: ", i)
            genre = 0

        genres.append(genre)
        # End loop


    #song_info.insert(0, "MB Genre", genres)
    #print(song_info)
    return genres



if __name__ == '__main__':

    set_mb_useragent()

    dataset = pull_song_info("../../data_processed/MSD_Desired_Features.csv")
    data_subset = dataset

    data_labels = populate_genres(data_subset)

    # Add genre labels and remove index column
    data_subset.insert(1, "MB Genre", data_labels)
    dataset_clean = data_subset.drop(columns=["Unnamed: 0"])
    #print(dataset_clean)

    # Export pandas dataframe to csv file
    print('Exporting to csv file...')
    dataset_clean.to_csv('../../data_processed/Labeled_Data.csv')





