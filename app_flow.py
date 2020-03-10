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

# in db or not in db
# not in db ---> get features using API (not_in_db flow)
# in db ---> query the db for feature vectors (in_db flow)
# -------- Once you have normalized feature vectors --------- #
# whatever the combo is, get new fv by calculating the mean ---> put through kmeans
# ---> normal steps from notebook
# *** Don't need clusters in advance because either way we get a new vector and
# put it through the model
#  
def sort_inputs(query):
    not_in_db = []
    in_db = []
    if "by" in query:
        query = query.replace("by","").split(";")
        
    for track in query:
        track = track.split(",")

        name = track[0].strip()
        artist = track[1].strip()
        q = f'''SELECT * FROM track_clusters
        WHERE track_name ILIKE '%{name}%';
        '''
        r = run_query(q)
        
        if len(r) > 0:
            in_db.append(name + "," + artist)
        else:
            not_in_db.append(name + " " + artist)
    
    return in_db, not_in_db

# in_db, not_in_db = sort_inputs(seed)
#================================== IN DATABASE ===============================#
in_db_df = pd.DataFrame()
for t in in_db:
    track = t.split(",")
    name = track[0]

    q = f'''SELECT * FROM track_clusters
        WHERE track_name ILIKE '%{name}%';
        '''
    r = run_query(q)
    in_db_df = in_db_df.append(r)

in_db_df.iloc[:,3:-1]
#================================ NOT IN DATABASE =============================#
# create an empty dataframe 
not_in_db_df = pd.DataFrame()
for track in not_in_db:
    #whole feature vector; concat step in orig function
    fv = not_in_database(track)
    not_in_db_df = not_in_db_df.append(fv)

# min-max scaling
#querying for the database
	q = '''
		SELECT a.*, b.*, c.track_name, c.artist
		FROM librosa_features a 
			JOIN spotify_features b ON a.track_id = b.id
			JOIN tracks c ON a.track_id = c.track_id;'''

	database = run_query(q)
	database.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','track_name','artist'],axis=1, inplace=True)

	#append feature vector to bottom of the db
	database = pd.concat([database,not_in_db_df],ignore_index=True)

#apply a lambda function that does min-max normalization on the db
	database = database.iloc[:,1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

	#overwrite features vector df
    i = not_in_db_df.shape[0]
	not_in_db_df = database.iloc[-i:,:]

    #============================= combining step ============================#
    if len(in_db) > 0:
        input_df = pd.DataFrame()
        input_df = input_df.append(in_db_df)
    else:
        pass
    #============================== final steps ==============================#
    #get the mean
    new_fv = {final_df}.mean()

    #predict cluster
    c = km.predict(new_fv.values)
    
    #query the db for other tracks in the cluster
    q = f'''
    SELECT * FROM track_clusters WHERE cluster = {c[0]};'''
    cluster_df = run_query(q)

    distances = {}
    for i,row in cluster_df.iterrows():
        track_id = row['track_id']
        dist = cos_sim(fv,row[3:-1])
        distances[track_id] = dist

    # #sorts the resulting dict for the top 3 most similar songs
    top_three = sorted(distances, key=distances.get,reverse=True)[:10]
    for i in top_three:
        print(i,distances[i])

    q = f'''
    SELECT * FROM track_clusters
    WHERE track_id IN {tuple(top_three)}
    AND track_id NOT IN {tuple(y['track_id'])}
    LIMIT 3;'''
    res = run_query(q)
    res