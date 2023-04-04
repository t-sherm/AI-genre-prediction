#!/usr/bin/env python
"""A script that looks for a release in the MusicBrainz database by disc ID
    $ python musicbrainz_song_by_id.py c6ff7243-a317-4e99-9ce2-a8a9a3a29ef2
    disc:
        Sectors: 295099
        London Calling
            MusicBrainz ID: 174a5513-73d1-3c9d-a316-3c1c179e35f8
            EAN/UPC: 5099749534728
            cat#: 495347 2
        ...
"""

from __future__ import unicode_literals
import musicbrainzngs
import sys

musicbrainzngs.set_useragent(
    "python-musicbrainzngs-example",
    "0.1",
    "https://github.com/alastair/python-musicbrainzngs/",
)

def show_release_details(rel):
    """Print some details about a release dictionary to stdout.
    """
    print("\t{}".format(rel['title']))
    print("\t\tMusicBrainz ID: {}".format(rel['id']))
    if rel.get('barcode'):
        print("\t\tEAN/UPC: {}".format(rel['barcode']))
    for info in rel['label-info-list']:
        if info.get('catalog-number'):
            print("\t\tcat#: {}".format(info['catalog-number']))

def show_offsets(offset_list):
    offsets = None
    for offset in offset_list:
        if offsets == None:
            offsets = str(offset)
        else:
            offsets += " " + str(offset)
    print("\toffsets: {}".format(offsets))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: {} RELEASE_ID".format(sys.argv[0]))
    releaseid = args[0]

    try:
        # the "labels" include enables the cat#s we display
        result = musicbrainzngs.get_artist_by_id(releaseid,
                includes=["tags"]
                                                        )
    except musicbrainzngs.ResponseError as err:
        if err.cause.code == 404:
            sys.exit("release not found")
        else:
            sys.exit("received bad response from the MB server")

   # On success, result is a dictionary with a single key:
    # "release-list", which is a list of dictionaries.
    #if not result['release-list']:
    #    sys.exit("no release found")
    #for (idx, release) in enumerate(result):
    #    print("match #{}:".format(idx+1))
    #    show_release_details(release)
    print(result)