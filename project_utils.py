import sys
from os import environ
from secret import sql_password, spotify_credentials
import pandas as pd
import numpy as np
import psycopg2 as pg
from psycopg2 import Error
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
import librosa
import spotipy
import requests
import pickle

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
                                                      client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

##--------------SQL Utils--------------------------#
conn = pg.connect(database="spotify_db",
                  user="postgres", 
                  password=sql_password)

def run_query(q):
    '''a function that takes a SQL query as an argument
    and returns a pandas dataframe of that query'''
    with conn:
        try:
            cur = conn.cursor()
            cur.execute(q)
            return pd.read_sql(q, conn)

        except (Exception, pg.DatabaseError) as error:
            print(error)

def run_command(c):
    '''a function that takes a SQL command as an argument
    and executes it using the psycopg2 module'''
    with conn:
        try:
            cur = conn.cursor()
            cur.execute(c)
            cur.close()
            conn.commit()

        except (Exception, pg.DatabaseError) as error:
            print(error)

##-----------------Spotify Utils-------------------------##
def search_and_extract(track_query):
    '''A function that takes in a song query and returns
    the track id and preview url for that track in a dict.'''

    track_query = str(track_query)

    #uses the API to search for a track
    search = sp.search(track_query, type='track', limit=1, market='US')

    track_id = search['tracks']['items'][0]['id']
    preview_url = search['tracks']['items'][0]['preview_url']
    track_name = search['tracks']['items'][0]['name']
    artist = search['tracks']['items'][0]['artists'][0]['name']
    
    return track_id, preview_url, track_name, artist

def extract_features(track_id):
    '''A function that takes in a spotify track id, requests the audio
    features using the 'audio_features' endpoint from the Spotify API,
    and returns the features as a dataframe'''
    track_id = str(track_id)
    features = sp.audio_features(track_id)
	#     features[0].values()

    spotify_features = pd.DataFrame(data=features[0].values(),index=features[0].keys())
    spotify_features = spotify_features.transpose()
    spotify_features.drop(['type','uri','track_href','analysis_url'],axis=1,inplace=True)

    return spotify_features

##---------------Librosa Utils--------------------------##
def check_for_track_preview(url):
    '''Given a url object, checks if the track has a
        preview'''
    if url == None:
        return False
    else:
        return True

def get_mp3(url,track_id):
    '''A function that takes an mp3 url, and writes it to the local
        directory "audio-files"'''
    try:
        doc = requests.get(url)
        with open(f'audio-files/track_{track_id}.mp3', 'wb') as f:
            f.write(doc.content)
    except:
        pass

def librosa_pipeline(track_id):
    '''This function takes in a spotify track_id as a string
        and uploads the cooresponding mp3 preview from a local
        directory. The mp3 then goes through the feature
        extraction process. A dictionary is returned with each
        audio feature as a key and their cooresponding value.

        REQUIREMENTS:
        * MP3 file must be in the directory in the form below
        '''

    track = f'audio-files/track_{track_id}.mp3'

    d = {}
    d['track_id'] = track_id

    #load mp3
    y, sr = librosa.load(track, mono=True, duration=30)

    #feature extraction
    rmse = librosa.feature.rmse(y=y)
    d['rmse'] = np.mean(rmse)

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    d['spectral_centroid'] = np.mean(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    d['spectral_bandwidth'] = np.mean(spec_bw)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    d['rolloff'] = np.mean(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    d['zero_crossing_rate'] = np.mean(zcr)

    tempo = librosa.beat.tempo(y=y, sr=sr)
    d['tempo_bpm'] = tempo[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i,e in zip(range(1, 21),mfcc):
            d[f'mfcc{i}'] = np.mean(e)

    chroma = ['C', 'C#', 'D','D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for c,p in zip(chroma,chroma_stft):
        d[c] = np.mean(p)

    return d    

##---------------General Utils--------------------------##
def cos_sim(a,b):
    '''Calculates the cosine similarity between two feature
        vectors'''
    d = np.dot(a, b)
    l = (np.linalg.norm(a))*(np.linalg.norm(b))
    return d/l

def get_features_and_cluster(query):
    '''Queries the database for a track's
        features and cluster label, query must be
        in form : "track name, artist name" '''
    query = str(query)
    query = query.split(",")

    q = f'''
    SELECT * FROM track_clusters 
    WHERE track_name ILIKE '{query[0]}'
    AND artist ILIKE '{query[1]};'''

    result = run_query(q)
    seed_features = result.iloc[0,3:-1]
    cluster = result.loc[0,'cluster']

    return seed_features, cluster

def check_database(query):
    query = str(query)
    query = query.split(",")
    q = f'''
    SELECT * FROM track_clusters
    WHERE track_name ILIKE '%{query[0]}%'
    AND artist ILIKE '%{query[1]}%'
    ;'''
    r = run_query(q)

    if len(r) > 0:
        return True
    else:
        return False

def create_playlist(sp, recommended_tracks):
    user_all_data = sp.current_user()
    user_id = user_all_data["id"]

    playlist_all_data = sp.user_playlist_create(user_id, "Friendship Playlist")
    playlist_id = playlist_all_data["id"]
    playlist_uri = playlist_all_data["uri"]
    # try:
    sp.user_playlist_add_tracks(user_id, playlist_id, recommended_tracks)
    # except spotipy.client.SpotifyException as s:
    # 	print("could not add tracks")

    return playlist_uri

def parse_query(query):
    query = query[:-1] if query.endswith(';') else query
    query = query.split(";")

    for track in query:
        track = track.split(",")
        try:
            name = track[0].strip()
            artist = track[1].strip()
        except IndexError:
            return False


def sort_inputs(query):
    not_in_db = []
    in_db = []
    query = query.replace("'","_")
    query = query[:-1] if query.endswith(';') else query
    query = query.split(";")
    

    for track in query:
        track = track.split(",")

        name = track[0].strip()
        artist = track[1].strip()

        q = f'''SELECT * FROM track_clusters
        WHERE track_name ILIKE '%{name}%'
        AND artist ILIKE '%{artist}%';
        '''
        r = run_query(q)
        
        name = name.replace("_","'")
        
        if len(r) > 0:
            in_db.append(name + "," + artist)
        else:
            not_in_db.append(name + " " + artist)

    return in_db, not_in_db

#================================== IN DATABASE ===============================#
def in_database(in_db):
    in_db_df = pd.DataFrame()
    for t in in_db:
        track = t.split(",")
        name = track[0]

        q = f'''SELECT * FROM track_clusters
            WHERE track_name ILIKE '%{name}%';
            '''
        r = run_query(q)
        in_db_df = in_db_df.append(r)

    return in_db_df
#================================ NOT IN DATABASE =============================#
def not_in_database(not_in_db):
    #search for a track and extract metadata from results
    metadata = {}
    for track in not_in_db:
        track_id, preview_url, track_name, artist = search_and_extract(track) #using the input track name as the query to search spotify
        metadata[track_id] = [preview_url,track_name,artist]

    not_in_db_df = pd.DataFrame()
    no_url = {}
    for track_id in metadata.keys():
        if metadata[track_id][0] == None:
            no_url[track_id] = [metadata[track_id][1],metadata[track_id][2]]
            continue
        
        spotify_features = extract_features(track_id)
        get_mp3(metadata[track_id][0],track_id)

        #use librosa to extract audio features
        r = librosa_pipeline(track_id)

        #turning dict into datframe
        librosa_features = pd.DataFrame(r,index=[0])

        #concatenating the two dfs so the feature vector will be in the same format as the db
        all_features = pd.concat([librosa_features,spotify_features],axis=1)
        all_features.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','mfcc20'],axis=1, inplace=True)

        #insert metadata into dataframe
        all_features.insert(1,'track_name',metadata[track_id][1])
        all_features.insert(2,'artist',metadata[track_id][2])
        
        not_in_db_df = not_in_db_df.append(all_features)
    
    not_in_db_df = not_in_db_df.reset_index(drop=True)
    return not_in_db_df, no_url

def scale_features(not_in_db_df):
    # min-max scaling
    #querying for the database
    q = '''SELECT a.*, b.*, c.track_name, c.artist
        FROM librosa_features a 
        JOIN spotify_features b ON a.track_id = b.id
        JOIN tracks c ON a.track_id = c.track_id;'''

    database = run_query(q)
    database.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','track_name','artist'],axis=1, inplace=True)

    #append feature vector to bottom of the db
    database = pd.concat([database.iloc[:,1:],not_in_db_df.iloc[:,3:]],ignore_index=True)

    # #apply a lambda function that does min-max normalization on the db
    database = database.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    #overwrite features vector df
    i = not_in_db_df.shape[0]
    not_in_db_df.iloc[:,3:] = database.iloc[-i:,:].values
    return not_in_db_df

#============================= combining step ============================#
def combine_frames(in_db, not_in_db, in_db_df, not_in_db_df):
    if len(in_db_df) == 0 and len(not_in_db_df) == 0:
        input_df = pd.DataFrame()
    
    if len(in_db) > 0 and len(not_in_db) > 0:
        input_df = in_db_df.iloc[:,:-1]
        input_df = input_df.append(not_in_db_df)

    elif len(in_db) == 0 and len(not_in_db) > 0:
        input_df = not_in_db_df

    elif len(in_db_df) > 0 and len(not_in_db_df) == 0:
        input_df = in_db_df.iloc[:,:-1]

    input_df = input_df.reset_index(drop=True)
    return input_df

#============================== final steps ==============================#
def get_cluster_df(input_df, km):
    #get the mean
    new_fv = input_df.mean()

    #predict cluster
    new_fv = new_fv.values
    new_fv = new_fv.reshape(1,-1)

    c = km.predict(new_fv)

    #query the db for other tracks in the cluster
    q = f'''
    SELECT * FROM track_clusters WHERE cluster = {c[0]};'''
    cluster_df = run_query(q)
    return cluster_df, new_fv

def get_results(cluster_df, new_fv, input_df, no_url):

    distances = {}
    for i,row in cluster_df.iterrows():
        track_id = row['track_id']
        dist = cos_sim(new_fv,row[3:-1])
        distances[track_id] = dist

    # #sorts the resulting dict for the top 3 most similar songs
    top_three = sorted(distances, key=distances.get,reverse=True)[:10]
    
    if len(input_df) == 1:
        input_ids = str(input_df.loc[0,'track_id'])
        
        q = f'''
            SELECT track_id, track_name, artist FROM track_clusters
            WHERE (track_id IN {tuple(top_three)})
            AND track_id != '{str(input_ids)}'
            LIMIT 3;'''
        res = run_query(q)

    elif len(input_df) == 0:
        print("Sorry could not get recommendations for any track you supplied. Please try different tracks.")
        sys.exit()
    
    elif len(input_df) > 1:
        input_ids = tuple(input_df['track_id'])

        q = f'''
        SELECT track_id, track_name, artist FROM track_clusters
        WHERE (track_id IN {tuple(top_three)})
        AND track_id NOT IN {input_ids}
        LIMIT 3;'''
        res = run_query(q)

    #returns a df with the track name, artist, and distance value for recommended tracks
    for i,row in res['track_id'].iteritems():
        res.loc[i,'distance'] = distances[row]
    
    res_df = res[['track_name','artist','distance']].sort_values('distance',ascending=False)
    # if len(no_url) > 0:
    #     print("")
    #     for k,v in no_url.items():
    #         print(f"Could not analyze: {v[0]}, by {v[1]}")
    return res_df, no_url
    
    