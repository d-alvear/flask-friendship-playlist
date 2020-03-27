import sys
from os import environ
from secret import sql_password, spotify_credentials
import pandas as pd
import numpy as np
import psycopg2 as pg
from psycopg2 import Error
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import spotipy
import requests
from sklearn.metrics.pairwise import cosine_similarity

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
                                                      client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#=============================== SQL Utils ====================================#
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

#============================= Spotify Utils ==================================#
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
    artist_id = search['tracks']['items'][0]['artists'][0]['id']

    return track_id, preview_url, track_name, artist, artist_id

def get_artist_genre(artist_id):
    '''A function that takes in a Spotify artist id, calls the Spotify 
    API, and returns the artist genres, as a list'''
    search = sp.artist(artist_id)
    return search['genres']
    
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

#============================= Librosa Utils ==================================#
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
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    d['spectral_centroid'] = np.mean(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    d['spectral_bandwidth'] = np.mean(spec_bw)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    d['rolloff'] = np.mean(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    d['zero_crossing_rate'] = np.mean(zcr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i,e in zip(range(1, 21),mfcc):
            d[f'mfcc{i}'] = np.mean(e)

    chroma = ['C', 'C#', 'D','D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for c,p in zip(chroma,chroma_stft):
        d[c] = np.mean(p)

    return d    

#============================= General Utils ==================================#
def check_query_format(query):
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

    return [in_db, not_in_db]

def parse_and_sort_inputs(user_a_query, user_b_query):
    '''Takes in both user's input strings, and sets up a 
    dictionary to keep track of each user's inputs and 
    whether they are in the database or not. Calls the 
    sort_inputs function to parse and sort query strings.
    Returns the resulting dictionary'''
    # example user inputs
    # user_a = "malibu, miley cyrus; video games, lana del rey; you're no good, linda ronstadt"
    # user_b = "don't stop me now, queen; rocket man, elton john; toxic, britney spears"

    # combines the form input into a list for interation; dict to store tracks
    users = [user_a_query, user_b_query]
    initial_inputs = {'user_a':None,
                    'user_b':None}

    # for each set of tracks, I need to keep track which tracks are in/not in the DB
    for key,user in zip(initial_inputs.keys(),users):   
        in_db, not_in_db = sort_inputs(user)
        initial_inputs[key] = [in_db, not_in_db]


    return initial_inputs

def cos_sim(a,b):
    '''Calculates the cosine similarity between two feature
        vectors'''
    d = np.dot(a, b)
    l = (np.linalg.norm(a))*(np.linalg.norm(b))
    return d/l


# # for creating a spotify playlist from track_uris
# def create_playlist(sp, recommended_tracks):
#     user_all_data = sp.current_user()
#     user_id = user_all_data["id"]

#     playlist_all_data = sp.user_playlist_create(user_id, "Friendship Playlist")
#     playlist_id = playlist_all_data["id"]
#     playlist_uri = playlist_all_data["uri"]
#     # try:
#     sp.user_playlist_add_tracks(user_id, playlist_id, recommended_tracks)
#     # except spotipy.client.SpotifyException as s:
#     # 	print("could not add tracks")

#     return playlist_uri
#================================== IN DATABASE ===============================#
def in_database(in_db):
    '''takes in a list of tracks, parses it,
    queries the db for each track's feature 
    vector and genre, the appends each to
    a df, then returns the df'''

    in_db_df = pd.DataFrame()
    for t in in_db:
        track = t.split(",")
        name = track[0]
        artist = track[1]

        q = f'''SELECT a.*, b.genre 
            FROM track_clusters a JOIN tracks b
            ON a.track_id = b.track_id
            WHERE a.track_name ILIKE '%{name}%'
            AND a.artist ILIKE '%{artist}%';
            '''
        r = run_query(q)
        in_db_df = in_db_df.append(r)

    return in_db_df
#================================ NOT IN DATABASE =============================#
def not_in_database(not_in_db):
    #search for a track and extract metadata from results
    metadata = {}
    for track in not_in_db:
        track_id, preview_url, track_name, artist, artist_id = search_and_extract(track) #using the input track name as the query to search spotify
        genres = get_artist_genre(artist_id)
        metadata[track_id] = [preview_url,track_name,artist,artist_id,genres]

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
        all_features.drop(['id','duration_ms','time_signature','mode','key'],axis=1, inplace=True)

        #insert metadata into dataframe
        all_features.insert(1,'track_name',metadata[track_id][1])
        all_features.insert(2,'artist',metadata[track_id][2])
        all_features.insert(48,'genre',metadata[track_id][4][0])
        
        not_in_db_df = not_in_db_df.append(all_features)
    
    not_in_db_df = not_in_db_df.reset_index(drop=True)
    return not_in_db_df, no_url

def scale_features(not_in_db_df):
    # min-max scaling
    #querying for the database
    q = '''SELECT a.*, b.*
        FROM librosa_features a 
        JOIN spotify_features b ON a.track_id = b.id;'''

    database = run_query(q)
    database.drop(['id','duration_ms','time_signature','mode','key'],axis=1, inplace=True)
    i = len(database)
    fv = not_in_db_df.drop(['track_name','artist','genre'],axis=1)

    #append feature vector to bottom of the db
    database = pd.concat([database.iloc[:,1:],fv.iloc[:,1:]],ignore_index=True)

    # #apply a lambda function that does min-max normalization on the db
    database = database.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    #overwrite features vector df
    not_in_db_df.iloc[:,3:-1] = database.iloc[i:,:].values
    return not_in_db_df


#============================= Combining Steps ================================#
def generate_user_df(user_lists):
    '''MUST BE CALLED ON EACH USER KEY SEPARATELY
    Takes in the keys of the initial_inputs dictionary.
    This function calls the in_database and not_in_database
    functions, then concatenates them to create the final
    user dataframes needed to make recommendations. It
    also stores the songs that could not be analyzed in the
    no_url dictionary'''
    
    in_db_df = in_database(user_lists[0])
    not_in_db_df, no_url = not_in_database(user_lists[1])
    
    if not_in_db_df.empty:
        user_df = in_db_df
    else:
        not_in_db_df = scale_features(not_in_db_df)
        user_df = pd.concat([in_db_df,not_in_db_df])
        
    return user_df, no_url

def get_similar_track_ids(input_track_df):
    '''
    IMPORTANT:THIS FUNCTION IS MEANT FOR ITERATION
    ----------------------------------------------
    Takes in a pandas series of a single track
    that contains track_id, and genre. Then queries
    the db for all tracks in the same genre as the
    input track. The cosine similarity is then 
    calculated between the input track and all
    other tracks within the genre. The top two
    most similar track ids are returned in a list'''
    
    track_id = input_track_df['track_id']
    genre = input_track_df['genre']
    
    q =  f'''
    SELECT * FROM track_clusters
    WHERE track_id = '{track_id}';'''
    features = run_query(q)

    
    q2 = f'''
    SELECT a.*, b.genre 
    FROM tracks b
    JOIN track_clusters a ON b.track_id = a.track_id
    WHERE b.genre = '{genre}'
    AND a.track_id != '{track_id}';'''
    genre_tracks = run_query(q2)
    
    
    all_scores = {}
    for i,row in genre_tracks.iterrows():
        track_id = row['track_id']
        score = cos_sim(features.iloc[0,3:],row[3:-1])
        all_scores[track_id] = score

    most_similar = sorted(all_scores, 
                          key=all_scores.get,
                          reverse=True)[:2]
    return most_similar

def get_feature_vector_array(id_list):
    '''
    IMPORTANT:THIS FUNCTION IS MEANT FOR ITERATION
    ----------------------------------------------
    Takes in a list of track_ids, queries the
    db for each track's feature vector, and returns
    a 2D array of the feature vectors and cooresponding
    track_ids as an index.
    '''
    id_list = set(id_list)
    q = f'''
    SELECT * FROM track_clusters
    WHERE track_id IN {tuple(id_list)};'''
    fv = run_query(q)

    fv = fv.set_index('track_id')
    index = fv.index
    fv = fv.iloc[:,2:]
    array = fv.values
    
    return index, array
#============================== Final Steps ==================================#
def create_similarity_matrix(user_a_array, user_a_index, user_b_array, user_b_index):
    '''Takes in two 2D user arrays and their corresponding 
    track_id indices, calculates the cosine similarity
    between all tracks in each 2D array. Then sets up a
    pandas dataframe of the similarity scores
    '''
    cosine_matrix = cosine_similarity(user_a_array,user_b_array)

    cosine_df = pd.DataFrame(cosine_matrix,
                            columns=user_b_index,
                            index=user_a_index)

    return cosine_df

def get_combined_recommendations(cosine_df):
    '''Takes in the cosine similarity dataframe as an
    input, then finds the pairs of track that have 
    the top 3 similarity scores. Queries the db
    for the track metadata and uses the results as the
    final recommendations'''
    
    scores = {}
    for i,row in cosine_df.iterrows():
        scores[max(row)] = [i,row.idxmax()]
        
    top_three = sorted(scores,reverse=True)[:3]

    ids = [scores[i][0] for i in top_three] + [scores[i][1] for i in top_three]
    ids = set(ids)

    q = f'''
    SELECT track_id, track_name, artist, genre FROM tracks
    WHERE track_id IN {tuple(ids)};'''
    final = run_query(q)
    return final