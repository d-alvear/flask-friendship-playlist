from secret import *
import requests
import pandas as pd
from pandas.io.json import json_normalize

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

# categories I want to get playlists for
category_ids = ['toplists','in_the_car','pop','hiphop','mood','decades','country','rock','latin','focus','chill','edm_dance','rnb','indie_alt',
                'roots','party','jazz','soul','romance','blues','punk','funk']

# single playlists I want songs from
single_playlists = ['37i9dQZF1DWTLSN7iG21yC','37i9dQZF1DX8CopunbDxgW','37i9dQZF1DX0b1hHYQtJjp','37i9dQZF1DX0BxHamIEkKV','37i9dQZF1DWTQwRw56TKNc','37i9dQZF1DWY0DyDKedRYY']


def get_playlist_tracks(playlist_id):
    '''
    Uses 'playlist_id' to get songs from a specific playlist
    Returns a list of tracks
    '''
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


library = {}
for c in category_ids:
    try:
        results = sp.category_playlists(category_id=c, country='US', limit=50, offset=0)
        playlists = results['playlists']['items']

        playlist_ids = [p['id'] for p in playlists]

        playlist_tracks = []
        for q in playlist_ids:
            r = get_playlist_tracks(q)
            playlist_tracks.extend(r)
            
        library[c] = playlist_tracks

    except:
        pass

all_tracks = {}

failed = []
for k in library.keys():
    for t in library[k]:
        try:
            track_id = t['track']['id']
            track_name = t['track']['name']
            artist = t['track']['artists'][0]['name']
            artist_id = t['track']['artists'][0]['id']
            track_album_album_type = t['track']['album']['album_type']
            track_album_id = t['track']['album']['id']
            track_album_name = t['track']['album']['name']
            track_duration_ms = t['track']['duration_ms']
            track_popularity = t['track']['popularity']
            track_preview_url = t['track']['preview_url']
            subgenres = None
            genre = None 
            top_subgenre = None
            
            all_tracks[track_id] = [track_name, artist, artist_id, track_album_album_type,
                                    track_album_id, track_album_name, track_duration_ms,
                                    track_popularity, track_preview_url, subgenres, genre,
                                    top_subgenre]

        except TypeError:
            failed.append(t)
            pass
       

print(len(all_tracks), len(failed))

# with open('failed_tracks.txt', 'w') as file:
#     for f in failed:
#         file.write('%s\n' % f)

norm_df = json_normalize(all_tracks)
norm_df = norm_df.transpose().reset_index()


final_df = pd.DataFrame(norm_df[0].to_list(), columns=['track_name', 'artist', 'artist_id', 'track_album_album_type',
                                    'track_album_id', 'track_album_name', 'track_duration_ms',
                                    'track_popularity', 'track_preview_url', 'subgenres', 'genre', 'top_genre'])

final_df.insert(0,'track_id',norm_df['index'])

final_df.to_csv('tracks.csv', sep=";")