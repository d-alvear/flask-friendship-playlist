#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, redirect, Response
from project_utils import *
from genre_replace import genre_replace
from secret import *
import pandas as pd
import numpy as np
import psycopg2 as pg
import librosa
import spotipy
import requests


# connect to spotify_db
conn = pg.connect(database=sql_credentials['database'],
                  user=sql_credentials['user'], 
                  password=sql_credentials['password'],
                  host=sql_credentials['host'])


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
													  client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


app = Flask(__name__, static_folder='static', template_folder='views')

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

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/test_cases')
def test_cases():
	return render_template('test_cases.html')

@app.route('/results', methods=['POST','GET'])
def friendship_app():
	
	query_a = request.form['query_a']
	query_b = request.form['query_b']

	# REWRITE THIS TO CHECK BOTH USERS' QUERIES
	for q in [query_a, query_b]:
		if check_query_format(q) == False:
			return "One or both of your queries don't look quite right, make sure they look like the example. They should include the name of the song, and the artist, separated by a comma."
		else:
			pass

	user_a_in_db, user_a_to_get = sort_inputs(query_a)
	user_b_in_db, user_b_to_get = sort_inputs(query_b)

	user_a_df = None
	user_b_df = None
	for user_list in enumerate([user_a_to_get,user_b_to_get]):
		if (len(user_list[1]) == 0) and (user_list[0] == 0):
			user_a_df = user_a_in_db
			print("Got It!")
		elif (len(user_list[1]) == 0) and (user_list[0] == 1):
			user_b_df = user_b_in_db
			print("Got It!")
		elif (len(user_list[1]) > 0) and (user_list[0] == 0):
			user_a_df = not_in_database_pipeline(user_list[1],user_a_in_db)
			print("Need to Get")
		elif (len(user_list[1]) > 0) and (user_list[0] == 1):
			user_b_df = not_in_database_pipeline(user_list[1],user_b_in_db)
			print("Need to Get")
	
	user_a_recs = [get_similar_track_ids(user_a_df, user_a_in_db)]
	user_b_recs = [get_similar_track_ids(user_b_df, user_b_in_db)]
		
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