"""Fetch the Sentinel-2 hand-labelled chips from the public Sen1Floods11 bucket.

The optical chips were never downloaded because the models in this paper read SAR
only. The labeller survey does not train anything, so it can use them: what it needs
is imagery plus a rule, and the rules the published pipeline uses are colour rules
that have no SAR analogue.
"""
import concurrent.futures as cf
import json
import os
import urllib.parse
import urllib.request

API = 'https://storage.googleapis.com/storage/v1/b/sen1floods11/o'
MEDIA = 'https://storage.googleapis.com/sen1floods11/'
PREFIX = 'v1.1/data/flood_events/HandLabeled/S2Hand/'
DEST = 'data/sen1floods11/S2Hand'


def listing():
    names, tok = [], None
    while True:
        u = API + '?prefix=' + urllib.parse.quote(PREFIX) + '&maxResults=1000'
        if tok:
            u += '&pageToken=' + tok
        d = json.load(urllib.request.urlopen(u, timeout=180))
        names += [i['name'] for i in d.get('items', []) if i['name'].endswith('.tif')]
        tok = d.get('nextPageToken')
        if not tok:
            return sorted(names)


def get(name):
    dest = os.path.join(DEST, os.path.basename(name))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0
    urllib.request.urlretrieve(MEDIA + urllib.parse.quote(name), dest)
    return os.path.getsize(dest)


names = listing()
print('{} chips listed'.format(len(names)), flush=True)
with cf.ThreadPoolExecutor(8) as ex:
    got = list(ex.map(get, names))
print('done, {} fetched, {:.2f} GB'.format(sum(1 for g in got if g), sum(got) / 1e9))
