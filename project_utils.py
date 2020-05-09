from secret import *
import pandas as pd
import numpy as np
import psycopg2 as pg
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import json
import requests
from genre_replace import genre_replace
from sklearn.metrics.pairwise import cosine_similarity

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#=============================== SQL Utils ====================================#
conn = pg.connect(database=sql_credentials['database'],
				  user=sql_credentials['user'], 
				  password=sql_credentials['password'],
				  host=sql_credentials['host'])


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
		finally:
			pass

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

	search = sp.artist(artist_id)
	genre_list = search['genres']

	if preview_url == None:
		url, new_genre = get_missing_url(artist, track_name)
		track_data = [track_id, url, track_name, artist, artist_id, new_genre]
	
	elif preview_url != None:
		track_data = [track_id, preview_url, track_name, artist, artist_id, genre_list]
	
	return track_data

def extract_features(track_list):
	'''A function that takes in a spotify track id, requests the audio
	features using the 'audio_features' endpoint from the Spotify API,
	and returns the features as a json'''
	features = sp.audio_features(track_list)
	return features
#============================= Librosa Utils ==================================#
def get_mp3(track_dict):
	'''A function that takes an mp3 url, and writes it to the local
		directory "audio-files"'''
	for track_id, properties in track_dict.items():
		try:
			doc = requests.get(properties[0])
			with open(f'/tmp/track_{track_id}.wav', 'wb') as f:
				f.write(doc.content)
		except:
			pass 

def librosa_pipeline(track_id):
	
	path = f'/tmp/track_{track_id}.wav'

	d = {}
	d['track_id'] = track_id

	#load mp3
	y, sr = librosa.load(path, mono=True, duration=30)

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
def get_missing_url(artist,song):
	'''falls back on the iTunes API to get a 30 sec. preview of a song if Spotify
		doesn't provide one, also assigns a different genre since iTunes uses
		more traditional genres, returns track metadata'''

	artist = artist.replace(" ","+")
	song = song.replace(" ","+")
	
	try:
		r = requests.get(f"https://itunes.apple.com/search?term={artist}+{song}&limit=1")
		content = json.loads(r.text)
		preview = content['results'][0]["previewUrl"]
		genre = content['results'][0]["primaryGenreName"]
		return str(preview), genre.lower()
	
	except:
		pass

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
	user_df = pd.DataFrame()
	if "'" in query:
		query = query.replace("'","_")
	elif "." in query:
		query = query.replace(".","_")
	elif "-" in query:
		query = query.replace("-","_")
	
	query = query[:-1] if query.endswith(';') else query
	query = query.split(";")
	
	for track in query:
		track = track.split(",")

		name = track[0].strip()
		artist = track[1].strip()

		q = f'''SELECT a.*, b.genre 
			FROM norm_tracks a 
				JOIN track_metadata b
				ON a.track_id = b.track_id
			WHERE a.track_name ILIKE '%{name}%'
			AND a.artist ILIKE '%{artist}%'
			LIMIT 1
			'''
		r = run_query(q)
		
		if "'" in query:
			name = name.replace("_","'")
		elif "." in query:
			name = name.replace("_",".")
		elif "-" in query:
			name = name.replace("_","-")
		
		if len(r) > 0:
			user_df = user_df.append(r,ignore_index=True)
		else:
			not_in_db.append(name + " " + artist)
	
	return user_df, not_in_db

def cos_sim(a,b):
	'''Calculates the cosine similarity between two feature
		vectors'''
	d = np.dot(a, b)
	l = (np.linalg.norm(a))*(np.linalg.norm(b))
	return d/l

#================================ NOT IN DATABASE =============================#
def gather_metadata(not_in_db):
	#search for a track and extract metadata from results
	if len(not_in_db) > 0:
		metadata = {}
		for track in not_in_db:
			track_data = search_and_extract(track) #using the input track name as the query to search spotify
			metadata[track_data[0]] = track_data[1:]
		return metadata
	else:
		return None

def get_spotify_features(metadata):
	'''Iterates over the metadata dict to create the not_null 
	dict {track_id : [url, ...]} by appending key, value pairs
	where the track preview url isn't null. If lenght of 
	not_null dict is greater than 0, then all the keys (track_ids)
	get passed in to extract_features() which queries the Spotify
	API for the audio features of all the tracks and appends the
	features to a dict. Then not-null gets passed into get_mp3() 
	which downloads the track preview.
	Returns the not_null dict and spotify_features dict, or None
	'''
	
	if metadata != None:
		not_null = {}
		for track_id, properties in metadata.items():
			if properties[0] != None:
				not_null[track_id] = properties
			else:
				pass
		
		if len(not_null) > 0:
			spotify_features = extract_features(list(not_null.keys()))
			get_mp3(not_null)
			return not_null, spotify_features
		
		elif len(not_null) == 0:
			return None, None
	
	else:
		return None, None

def combine_all_features(metadata, librosa_features, spotify_features):
	# concatenating the two dfs so the feature vector will be in the same format as the db
	if (metadata != None) & (librosa_features != None) & (spotify_features != None):
		all_features = pd.DataFrame(librosa_features).merge(pd.DataFrame(spotify_features),left_on='track_id',right_on='id')
		all_features.drop(['id','duration_ms','time_signature','mode','key','type','uri','track_href','analysis_url'],axis=1, inplace=True)

		#insert metadata into dataframe
		for i,row in all_features.iterrows():
			for k in metadata.keys():
				if row['track_id'] == k:
					all_features.loc[i,'track_name'] = metadata[k][1]
					all_features.loc[i,'artist'] = metadata[k][2]
					all_features.loc[i,'genre'] = metadata[k][4][0]

		all_features = all_features[['track_id','track_name', 'artist', 'spectral_centroid', 'spectral_bandwidth', 'rolloff',
									'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
									'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
									'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
									'mfcc20', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#',
									'B', 'danceability', 'energy', 'loudness', 'speechiness',
									'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'genre']]
		return all_features

	else:
		return pd.DataFrame()	

def scale_features(not_in_db_df):
	fv = not_in_db_df.drop(['track_id','track_name','artist','genre'],axis=1)

	db_min = pd.DataFrame([168.8868, 227.2588, 263.5151, 0.0024, -1041.1220004684726, -14.337252373694344,
						-94.7838, -37.9753, -64.7201, -46.7442, -38.7988, -53.7201, -50.589676096301964, -28.2891,
 						-37.7447, -30.5451, -31.8016, -28.1815, -35.1259, -31.553, -24.370357929737853, -26.2627,
 						-30.4465, -25.3481, 0.012, 0.0071, 0.0135, 0.0082, 0.0034, 0.002, 0.001, 0.0043, 0.0067,
						 0.0083, 0.0101, 0.01730265644633166, 0.0, 0.001, -49.645, 0.0, 0.0, 0.0, 0.0124, 0.0, 0.0]).transpose()
	
	db_max = pd.DataFrame([6472.795208183058, 3495.5789, 9725.607340052851, 0.678866248060512, 95.3198, 262.1956, 119.8897,
						102.1222, 64.5132, 54.0076, 30.9822, 36.385, 30.1297, 38.9629, 22.5568, 29.3508, 20.4935, 27.4429,
						29.0006, 20.9568, 20.23785160907396, 24.7055, 23.011, 21.0686, 0.9212, 0.9761, 0.9952, 0.9784898468181313,
						0.9626, 0.9773305692855647, 0.9534980854249783, 0.9608, 0.9404, 1.0, 0.9269, 0.9841, 0.981, 1.0, 1.342,
						0.947, 0.996, 0.997, 0.997, 0.993, 244.162]).transpose()

	db_min.columns = fv.columns
	db_max.columns = fv.columns

	combined = pd.concat([fv, db_min, db_max], ignore_index=True)
	scaled = ((combined - combined.min())/(combined.max()-combined.min()))
	
	#overwrite features vector df
	i = len(fv)
	not_in_db_df.iloc[:,3:-1] = scaled.iloc[:i,:]
	# not_in_db_df.iloc[:,3:-1] = database.values
	return not_in_db_df

def remap_genres(df):
	for i,genre in df['genre'].iteritems():
		if isinstance(genre,list):
			for g in genre:
				if g in genre_replace.keys():
					df.loc[i,'genre'] = genre_replace[g]
				else:
					continue
		elif isinstance(genre,str):
			if genre in genre_replace.keys():
				df.loc[i,'genre'] = genre_replace[genre]
			elif genre not in genre_replace.keys():
				df.loc[i,'genre'] = None
				
	return df

#============================= Combining Steps ================================#
def generate_user_df(user_input_df, all_features_df):
	'''MUST BE CALLED ON EACH USER KEY SEPARATELY
	Takes in the keys of the initial_inputs dictionary.
	This function calls the in_database and not_in_database
	functions, then concatenates them to create the final
	user dataframes needed to make recommendations. It
	also stores the songs that could not be analyzed in the
	no_url dictionary'''
		
	if (user_input_df.empty == False) & (all_features_df.empty == False):
		scaled = scale_features(all_features_df)
		user_df = pd.concat([user_input_df,scaled],ignore_index=True)
		
	elif (user_input_df.empty == False) & (all_features_df.empty == True):
		user_df = user_input_df
	
	elif (user_input_df.empty == True) & (all_features_df.empty == False):
		user_df = scale_features(all_features_df) 

	return user_df

def get_similar_track_ids(user_df, user_in_db):
	''''''
	if user_in_db.empty == False:
		track_ids = list(user_in_db.loc[:,'track_id'])
		not_in_db = user_df[~user_df['track_id'].isin(track_ids)]
		in_db = user_in_db
	elif user_in_db.empty == True:
		not_in_db = user_df
		in_db = None
	
	genres = not_in_db.loc[:,'genre'].unique()
	
	if None in genres:
		print("Generating Window function")
		q = f'''
		SELECT a.*, b.genre 
		FROM ( SELECT track_id, genre, 
				row_number() OVER (PARTITION BY genre ORDER BY RANDOM()) 
				FROM track_metadata
			) as b
		JOIN norm_tracks a
		ON a.track_id = b.track_id
		WHERE row_number < 10
			AND b.genre IN ('pop','rock','rap','hip hop','indie')
		ORDER BY b.genre;'''
		all_tracks = run_query(q)
		all_tracks.set_index('track_id',inplace=True)
	
	genres = genres[genres != None]
	print("Getting genre tracks")
	if len(genres) > 1:
		q = f'''
		SELECT a.*, b.genre 
		FROM norm_tracks a
		JOIN track_metadata b ON a.track_id = b.track_id
		WHERE b.genre IN {tuple(genres)};'''
		genre_tracks = run_query(q)
		genre_tracks.set_index('track_id',inplace=True)

	elif len(genres) == 1:
		q = f'''
		SELECT a.*, b.genre 
		FROM norm_tracks a
		JOIN track_metadata b ON a.track_id = b.track_id
		WHERE b.genre = '{genres[0]}';'''
		genre_tracks = run_query(q)
		genre_tracks.set_index('track_id',inplace=True)
	
	# for tracks that aren't already in the database
	recs = []
	print("Not in DB Block")
	if not_in_db.empty == False:
		for i,row in not_in_db.iterrows():
			if row['genre'] == None:
				matrix = all_tracks.apply(lambda x: cos_sim(x[2:-1], row[3:-1]),axis=1,result_type='reduce')
				tracks = matrix.sort_values(ascending=False)[1:4].index
				recs.extend(list(tracks))

			else:
				g = genre_tracks[genre_tracks['genre'] == row['genre']]
				matrix = g.apply(lambda x: cos_sim(x[2:-1], row[3:-1]),axis=1,result_type='reduce')
				tracks = matrix.sort_values(ascending=False)[1:4].index
				recs.extend(list(tracks))
	
	if in_db is not None:
		print("Getting similarities")
		set_ids = list(in_db.loc[:,'track_id'])
		q = f'''
			SELECT track_id_2
			FROM (SELECT track_id_1,track_id_2,score, 
				row_number() OVER (PARTITION BY track_id_1)
				FROM similarities
				WHERE track_id_1 IN {tuple(set_ids)}
				ORDER BY 1,3 DESC) as b
			WHERE row_number < 3
			'''
		tracks = run_query(q)
		tracks = tracks.loc[:,'track_id_2']
		recs.extend(list(tracks))
	return recs

def get_feature_vector_array(id_list):
	'''
	Takes in a list of track_ids, queries the
	db for each track's feature vector, and returns
	a 2D array of the feature vectors and cooresponding
	track_ids as an index.
	'''
	id_list = set(id_list[0])
	q = f'''
	SELECT * FROM norm_tracks
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

	scores = {max(row): [i,row.idxmax()] for i, row in cosine_df.iterrows() }
		
	top_songs = sorted(scores,reverse=True)[:4]

	ids = [scores[i][0] for i in top_songs] + [scores[i][1] for i in top_songs]
	ids = set(ids)

	q = f'''
	SELECT track_name, artist, genre FROM track_metadata
	WHERE track_id IN {tuple(ids)};'''
	final = run_query(q)
	return final

def not_in_database_pipeline(to_get,in_db):
		metadata = gather_metadata(to_get)
		not_null, spotify_features = get_spotify_features(metadata)
		
		librosa_features = [librosa_pipeline(n) for n in not_null.keys()]

		user_df = combine_all_features(metadata,librosa_features,spotify_features)
		user_df = generate_user_df(in_db,user_df)

		user_df = remap_genres(user_df)

		return user_df