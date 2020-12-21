import os
from secret import *
import pandas as pd

import spotipy
import spotipy.util as util
import spotipy.oauth2 as oauth2
client_credentials_manager = oauth2.SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
                                                      client_secret=spotify_credentials['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = client_credentials_manager.get_access_token()

if token:
    sp = spotipy.Spotify(auth=token)
    print("Token Granted")
	
else:
    print("No token")

tracks = pd.read_csv('data/tracks_genre.csv', sep=';', index_col=0)

artist_ids = tracks['artist_id'].unique()[20000:30000]

artist_to_genre = {}
for i, artist_id in enumerate(artist_ids):
	search = sp.artist(artist_id)
	genre_list = search['genres']

	artist_to_genre[artist_id] = genre_list
	os.system('cls')
	print(f"{i+1}/{len(artist_ids)} artists found")


l = len(artist_to_genre.keys())
for j, (key, value) in enumerate(artist_to_genre.items()):

	idx = tracks[tracks['artist_id'] == key].index
	tracks.loc[idx, 'subgenres'] = str(value)

	try:
		tracks.loc[idx, 'top_genre'] = value[0]

	except:
		tracks.loc[idx, 'top_genre'] = str(value)
	
	os.system('cls')
	print(f"{j+1}/{l} genres assigned")
	

tracks.to_csv('data/tracks_genre.csv', sep=";")

