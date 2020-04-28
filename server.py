#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os import environ
from flask import Flask, request, render_template, redirect, Response
from project_utils import *
from genre_replace import genre_replace
from secret import *
import pandas as pd
import numpy as np
import psycopg2 as pg
from psycopg2 import Error
import librosa
import spotipy
import requests
from multiprocessing import Pool

# connect to spotify_db
# conn = pg.connect(database=sql_credentials['database'],
#                   user=sql_credentials['user'], 
#                   password=sql_credentials['password'],
#                   host =sql_credentials['host'])

conn = pg.connect(database="spotify_db",
				  user="postgres", 
				  password="damara1004")


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
													  client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


app = Flask(__name__, static_folder='static', template_folder='views')
pool = Pool(processes=4)

@app.route('/')
def authenticate_user():
	token = client_credentials_manager.get_access_token()
	if token:
		sp = spotipy.Spotify(auth=token)
		return redirect("/index")
	else:
		return "No token"

@app.route('/index')
def homepage():
	return render_template('index.html')

@app.route('/results', methods=['POST','GET'])
def friendship_app():
	similarities = unpickle()
	
	query_a = request.form['query_a']
	query_b = request.form['query_b']

	# REWRITE THIS TO CHECK BOTH USERS' QUERIES
	for q in [query_a, query_b]:
		if check_query_format(q) == False:
			return "One or both of your queries don't look quite right, make sure they look like the example. They should include the name of the song, and the artist, separated by a comma."
		else:
			pass

	# Sorting each user's input tracks by whether they are in/not in the db
	# Results are sorted into a dict

	user_a_in_db, user_a_to_get = sort_inputs(query_a)
	user_b_in_db, user_b_to_get = sort_inputs(query_b)

	metadata_a = gather_metadata(user_a_to_get)
	metadata_b = gather_metadata(user_b_to_get)

	not_null_a, spotify_features_a = get_spotify_features(metadata_a)
	not_null_b, spotify_features_b = get_spotify_features(metadata_b)
	
	track_list = prepare_tracklist(not_null_a, not_null_b)

	result_object = [pool.apply_async(single_librosa_pipeline, args=(x,)) for x in track_list]
	librosa_features_a, librosa_features_b = parse_result(not_null_a, not_null_b, result_object)
	
	all_features_a = combine_all_features(metadata_a, librosa_features_a, spotify_features_a)
	all_features_b = combine_all_features(metadata_b, librosa_features_b, spotify_features_b)
	
	# Creating a df with the feature vectors of each user's input tracks
	user_a_df = generate_user_df(user_a_in_db, all_features_a)
	user_b_df = generate_user_df(user_b_in_db, all_features_b)

	# # storing songs that couldn't be analyzed, separate loops because dicts could be different lengths
	# no_preview = {}
	# for k in no_url_a.keys():
	# 	no_preview[k] = no_url_a[k]
	
	# for l in no_url_b.keys():
	# 	no_preview[l] = no_url_b[l]

	# Mapping generalized genres to df
	user_a_df = remap_genres(user_a_df)
	user_b_df = remap_genres(user_b_df)
	
	# # Finding most similar songs to each user's input
	user_a_recs = get_similar_track_ids(user_a_df,user_a_in_db,similarities)
	user_b_recs = get_similar_track_ids(user_b_df,user_b_in_db,similarities)

	user_a_index, user_a_array = get_feature_vector_array(user_a_recs)
		
	user_b_index, user_b_array = get_feature_vector_array(user_b_recs)


	cosine_df = create_similarity_matrix(user_a_array,
										 user_a_index,
										 user_b_array,
										 user_b_index)


	recommendations = get_combined_recommendations(cosine_df)
	#======= PASS THIS INTO RESULTS.HTML =======#
	# if len(no_preview) > 0:
	# 	print("Could not get recommendations for:")
	# 	for i in no_preview.values():
	# 		print(i)


	return render_template('results.html', 
							tables=[recommendations[['track_name','artist','genre']].to_html(classes='data')], 
							titles=[recommendations.columns.values])

if __name__ == '__main__':
	app.run()