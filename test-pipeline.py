#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os import environ
from secret import spotify_credentials,sql_password
from flask import Flask, request, render_template, redirect, Response
from project_utils import *
import pandas as pd
import numpy as np
import psycopg2 as pg
from psycopg2 import Error
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import spotipy
import spotipy.util as util
import requests
from genre_replace import genre_replace

# connect to spotify_db
conn = pg.connect(database="spotify_db",
				  user="postgres", 
				  password=sql_password)


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
                                                      client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

##====User A: pop, rock, classic rock
##====User B: classic rock, classic rock, hip hop
# query_a = "BFF, Kesha; teenagers, my chemical romance; you're so vain, carly simon"
# query_b = "ball and chain, janis joplin; fire and rain, james taylor; in my feelings, drake"
# query_a = "bad and boujee, migos; god's plan, drake; fade, kanye west"
# query_b = "once in a lifetime, talking heads; crazy on you, heart; ramble on, led zeppelin"
query_a = "never let you go, third eye blind; come as you are, nirvana; my song 5, haim"
query_b = "dear to me, electric guest; good as hell, lizzo; talia, king princess"

def friendship_app(query_a,query_b):
	##~~~~ NEED TO CREATE SECOND INPUT FORM FOR USER B ~~~~##
	# query_a = request.form['query']
	# #query_b = request.form['query']

	# # REWRITE THIS TO CHECK BOTH USERS' QUERIES
	# if check_query_format(query) == False:
	# 	return "Your query doesn't look quite right, make sure it looks like the example. It should include the name of the song, and the artist, separated by a comma."
	# else:
	# 	pass

	# Sorting each user's input tracks by whether they are in/not in the db
	# Results are sorted into a dict
	initial_inputs = parse_and_sort_inputs(query_a,query_b)

	# Creating a df with the feature vectors of each user's input tracks
	user_a_df, no_url_a = generate_user_df(initial_inputs['user_a'])
	user_b_df, no_url_b = generate_user_df(initial_inputs['user_b'])

	# storing songs that couldn't be analyzed, separate loops because dicts
	# could be different lengths
	no_preview = {}
	for k in no_url_a.keys():
		no_preview[k] = no_url_a[k]
    
	for l in no_url_b.keys():
		no_preview[l] = no_url_b[l]

	# Mapping generalized genres to df
	user_a_df = remap_genres(user_a_df)
	user_b_df = remap_genres(user_b_df)
	
	# Finding most similar songs to each user's input
	user_a_recs = []
	for i,row in user_a_df.iterrows():
		rec = get_similar_track_ids(row)
		user_a_recs.extend(rec)
		
	user_a_index, user_a_array = get_feature_vector_array(user_a_recs)

	user_b_recs = []
	for i,row in user_b_df.iterrows():
		rec = get_similar_track_ids(row)
		user_b_recs.extend(rec)
		
	user_b_index, user_b_array = get_feature_vector_array(user_b_recs)

	cosine_df = create_similarity_matrix(user_a_array,
										 user_a_index,
										 user_b_array,
										 user_b_index)

	recommendations = get_combined_recommendations(cosine_df)
	
	if len(no_preview) > 0:
		print("Could not get recommendations for:")
		for i in no_preview.values():
			print(i)
	return recommendations

recs = friendship_app(query_a,query_b)
print(recs)