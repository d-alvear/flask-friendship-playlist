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
def search_and_extract(query):
	'''A function that takes in a song query and returns
	the track id and preview url for that track.'''
	query = str(query)
	
	#need to strip "by" from string to get accurate result
	if "by" in query == True:
		query = query.replace("by","")
	else:
		pass
	#uses the API to search for a track
	search = sp.search(query, type='track', limit=1, market='US')
	track_id = search['tracks']['items'][0]['id']
	preview_url = search['tracks']['items'][0]['preview_url']

	return track_id, preview_url

def extract_features(track_id):
	'''A function that takes in a spotify track id, requests the audio
	features using the 'audio_features' endpoint from the Spotify API,
	and returns the features as a dataframe'''
	track_id = str(track_id)
	features = sp.audio_features(track_id)
	features[0].values()

	spotify_features = pd.DataFrame(data=features[0].values(),index=features[0].keys())
	spotify_features = spotify_features.transpose()
	spotify_features.drop(['type','uri','track_href','analysis_url'],axis=1,inplace=True)
	
	return spotify_features

##---------------Librosa Utils--------------------------##
def check_for_track_preview(url):
	'''Given a url object, checks if the track has a
		preview'''
	if url == None:
		print("Sorry this song can't be analyzed, try a different song")
		sys.exit()
	else:
		pass

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

def get_features_and_cluster(seed):
	'''Queries the database for a track's
		features and cluster label'''
	seed = str(seed)

	q = f'''
	SELECT * FROM track_clusters WHERE track_name ILIKE '{seed}';'''

	result = run_query(q)
	seed_features = result.iloc[0,3:-1]
	cluster = result.loc[0,'cluster']
	
	return seed_features, cluster

##----------------------Friendship App-------------------##
def in_database(seed):
	sf, c = get_features_and_cluster(seed)

	#query the db for other tracks in the same cluster as seed
	q = f'''
	SELECT * FROM track_clusters WHERE cluster = {c};'''

	df = run_query(q)

	distances = {}
	for i,row in df.iterrows():
		track_id = row['track_id']
		dist = cos_sim(sf,row[3:-1])
		
		distances[track_id] = dist
		
	#sorts the resulting dict for the top 3 most similar songs
	top_three = sorted(distances, key=distances.get,reverse=True)[1:4]
	return top_three

def not_in_database(seed):
	#search for a track and extract metadata from results
	track_id, preview_url = search_and_extract(seed) #using the input track name as the query to search spotify

	#get audio features using the api
	features = sp.audio_features(track_id)

	spotify_features = pd.DataFrame(data=features[0].values(),index=features[0].keys())
	spotify_features = spotify_features.transpose()
	spotify_features.drop(['type','uri','track_href','analysis_url'],axis=1,inplace=True)

	#check for track preview then download it locally
	check_for_track_preview(preview_url)
	get_mp3(preview_url,track_id)

	# #use librosa to extract audio features
	r = librosa_pipeline(track_id)

	#turning dict into datframe
	librosa_features = pd.DataFrame(r,index=[0])

	#concatenating the two dfs so the feature vector will be in the same format as the db
	seed_features = pd.concat([librosa_features,spotify_features],axis=1)
	seed_features.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','mfcc20'],axis=1, inplace=True)

	#querying for the database
	q = '''
		SELECT a.*, b.*, c.track_name, c.artist
		FROM librosa_features a 
			JOIN spotify_features b ON a.track_id = b.id
			JOIN tracks c ON a.track_id = c.track_id;'''

	database = run_query(q)
	database.drop(['rmse','tempo_bpm','id','duration_ms','time_signature','mode','key','track_name','artist'],axis=1, inplace=True)

	#append feature vector to bottom of the db
	database = pd.concat([database,seed_features],ignore_index=True)

	#apply a lambda function that does min-max normalization on the db
	database = database.iloc[:,1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

	#overwrite seed features vector
	seed_features = database.iloc[-1,:]

	#bringing in kmeans clustering model
	km = pickle.load(open('kmeans_model.pkl','rb'))

	X = np.array(seed_features)
	X = X.reshape(1,-1)

	#predicting cluster
	c = km.predict(X)

	#query the db for other tracks in the cluster
	q = f'''
	SELECT * FROM track_clusters WHERE cluster = {c[0]};'''
	df = run_query(q)

	distances = {}
	for i,row in df.iterrows():
		track_id = row['track_id']
		dist = cos_sim(seed_features,row[3:-1])
		
		distances[track_id] = dist
		
	# #sorts the resulting dict for the top 3 most similar songs
	top_three = sorted(distances, key=distances.get,reverse=True)[1:4]
	return top_three

def check_database(seed):
    seed = str(seed)
    q = f'''
    SELECT * FROM track_clusters
    WHERE track_name ILIKE '{seed}'
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