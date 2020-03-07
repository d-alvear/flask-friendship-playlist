import sys
from os import environ
from sklearn.cluster import KMeans
from secret import sql_password, spotify_credentials
import pandas as pd
import numpy as np
import spotipy
import spotipy.util as util
import psycopg2 as pg
from spotipy.oauth2 import SpotifyClientCredentials
import project_utils as pu
import requests
import pickle

client_credentials_manager = SpotifyClientCredentials(client_id=environ.get('SPOTIPY_CLIENT_ID'),client_secret=environ.get('SPOTIPY_CLIENT_SECRET'))
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

conn = pg.connect(database="spotify_db",
                  user="postgres", 
                  password=sql_password)

def in_database(seed):
    #seed = input track name
    seed = str(seed)
    #step one: query the db for cluster and normalized 'track score' for seed
    q = f'''
    SELECT * FROM track_clusters WHERE track_name ILIKE '{seed}';'''

    result = pu.run_query(q)
    seed_features = result.iloc[0,3:-1]
    cluster = result.loc[0,'cluster']
    
    #step two: query the db for other tracks in the same cluster as seed
    q = f'''
    SELECT * FROM track_clusters WHERE cluster = {cluster};'''

    df = pu.run_query(q)

    #step three: calculate euclidean distance between the seed and other tracks in the cluster
    distances = {}
    for i,row in df.iterrows():
        track_id = row['track_id']
        dist = pu.cos_sim(seed_features,row[3:-1])
    
        distances[track_id] = dist
    #sorts the resulting dict for the top 3 most similar songs
    top_three = sorted(distances, key=distances.get,reverse=True)[1:4] #start at 1 because seed is in the results 
    
    # step four: query the db for the track name and artist of the most similar tracks
    q = f'''
    SELECT track_id, track_name, artist
    FROM tracks
    WHERE track_id IN {tuple(top_three)};'''

    recs_df = pu.run_query(q)

    #returns a df with the track name, artist, and distance value for recommended tracks
    for i,row in recs_df['track_id'].iteritems():
        recs_df.loc[i,'distance'] = distances[row]
    result = recs_df[['track_name','artist','distance']].sort_values('distance',ascending=False)
    
    return result.to_html()

def not_in_database(seed):

    #search for a track and extract metadata from results
    result = pu.search_and_extract(seed) #using the input track name as the query to search spotify

    track_id = result.loc[0,'track_id']
    features = sp.audio_features(track_id)

    spotify_features = pd.DataFrame(data=features[0].values(),index=features[0].keys())
    spotify_features = spotify_features.transpose()
    spotify_features.drop(['type','uri','track_href','analysis_url'],axis=1,inplace=True)

    #get preview url and dowload locally
    track_id = spotify_features.loc[0,'id']
    url = result.loc[0,'preview_url']

    if url == None:
        print("Sorry this song can't be analyzed, try a different song")
        sys.exit()
    else:
        pass
    
    pu.get_mp3(url,track_id)

    #use librosa to extract audio features
    r = pu.librosa_pipeline(track_id)

    #turning dict into datframe
    librosa_features = pd.DataFrame(r,index=[0])

    #bringing id column to the front of the df
    col = spotify_features.pop('id')
    spotify_features.insert(0,col.name,col)
    
    seed_features = pd.concat([librosa_features,spotify_features],axis=1)
    seed_features.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','mfcc20'],axis=1, inplace=True)

    q = '''
    SELECT a.*, b.*, c.track_name, c.artist
    FROM librosa_features a 
        JOIN spotify_features b ON a.track_id = b.id
        JOIN tracks c ON a.track_id = c.track_id;'''

    database = pu.run_query(q)
    database.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','track_name','artist'],axis=1, inplace=True)
    
    database = pd.concat([database,seed_features],ignore_index=True)

    #getting features column headers
    features = database.columns[1:]

    #getting actual feature columns
    data = database.loc[:,features].values

    #standardizing the features
    data = MinMaxScaler().fit_transform(data) 
        
    database_df = pd.DataFrame(data,columns=features)

    seed_features = database_df.iloc[-1,:]
    
    # #bringing in kmeans clustering model
    km = pickle.load(open('kmeans_model.pkl','rb'))

    X = np.array(seed_features)
    X = X.reshape(1,-1)

    #predicting cluster
    cluster = km.predict(X)

    #step two: query the db for other tracks in the same cluster as seed
    q = f'''
    SELECT * FROM track_clusters WHERE cluster = {cluster[0]};'''

    df = pu.run_query(q)

    #step three: calculate euclidean distance between the seed and other tracks in the cluster
    distances = {}
    for i,row in df.iterrows():
        track_id = row['track_id']
        dist = pu.cos_sim(seed_features,row[3:-1])
    
        distances[track_id] = dist
    #sorts the resulting dict for the top 3 most similar songs
    top_three = sorted(distances, key=distances.get,reverse=True)[1:4] #start at 1 because seed is in the results 
    # #step four: query the db for the track name and artist of the most similar tracks
    q = f'''
    SELECT track_id, track_name, artist
    FROM tracks
    WHERE track_id IN {tuple(top_three)};'''

    recs_df = pu.run_query(q)

    #returns a df with the track name, artist, and distance value for recommended tracks
    for i,row in recs_df['track_id'].iteritems():
        recs_df.loc[i,'distance'] = distances[row]
    
    result = recs_df[['track_name','artist','distance']].sort_values('distance',ascending=False)
    return result.to_html()

def check_database(seed):
    seed = str(seed)
    q = f'''
    SELECT * FROM track_clusters
    WHERE track_name ILIKE '{seed}'
    ;'''

    r = pu.run_query(q)

    if len(r) > 0:
        return True
    else:
        return False

# if __name__ == "__main__":
#     seed = sys.argv[1]
    
#     if check_database(seed) == True:
#         print("Searching...")
#         print(in_database(seed))

#     elif check_database(seed) == False:
#         print("Searching...")
#         print(not_in_database(seed))