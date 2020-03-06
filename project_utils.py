import pandas as pd
import numpy as np
import psycopg2 as pg
import pandas as pd
from os import environ
from psycopg2 import Error
from secret import sql_password
import librosa
from secret import spotify_credentials
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import librosa
client_credentials_manager = SpotifyClientCredentials(client_id=environ.get('SPOTIPY_CLIENT_ID'),client_secret=environ.get('SPOTIPY_CLIENT_SECRET'))
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
	'''A function that takes in a track as a string, uses spotify search,
	and returns pertinent metadata on the track as a df, assumes user 
	will type in full track name and the track they want is the first result'''

	#uses the API to search for a track
	query = str(query)
	search = sp.search(query, type='track', limit=10, market='US')
	tracks = search['tracks']['items']
	result = pd.DataFrame(columns=['track_id','track_name','artist','preview_url'])

	#iterates over results
	j=0
	for t in tracks:
		result.loc[j,'track_id'] = t['id']
		result.loc[j,'track_name'] = t['name']
		result.loc[j,'artist'] = t['artists'][0]['name']
		result.loc[j,'preview_url'] = t['preview_url']

		j += 1

	return result

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