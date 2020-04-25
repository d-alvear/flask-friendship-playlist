#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os import environ
from secret import *
from test_project_utils import *
import pandas as pd
import numpy as np
import psycopg2 as pg
from psycopg2 import Error
import librosa
import spotipy
import requests
from genre_replace import genre_replace
import time
import multiprocessing as mp

# connect to spotify_db
conn = pg.connect(database="spotify_db",
				  user="postgres", 
				  password="damara1004")


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
													  client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

##====User A: pop, rock, classic rock
##====User B: classic rock, classic rock, hip hop
query_a = "BFF, Kesha; teenagers, my chemical romance; you're so vain, carly simon"
query_b = "ball and chain, janis joplin; fire and rain, james taylor; in my feelings, drake"
# query_a = "bad and boujee, migos; god's plan, drake; fade, kanye west"
# query_b = "once in a lifetime, talking heads; crazy on you, heart; ramble on, led zeppelin"
# query_a = "never let you go, third eye blind; summer girl, haim; my song 5, haim"
# query_b = "juice, lizzo; good as hell, lizzo; talia, king princess"
# query_a="never let you go, third eye blind; summer girl, haim; my song 5, haim"
# query_b="once in a lifetime, talking heads; never let you go, third eye blind; fade, kanye west"
# query_a = "need your love, tennis; good bad times, hinds; automatic driver, la roux"
# query_b = "after the storm, kali uchis; see you again, tyler the creator; the big country, talking heads"

# query_a = "never let you go, RAC; good bad times, hinds; forever, Charli XCX"
# query_b = "what love is, vista kicks; see you again, tyler the creator; the big country, talking heads"

def friendship_app(query_a,query_b):
	# checks query format for both inputs
	# if check_query_format(query) == False:
	# 	return "Your query doesn't look quite right, make sure it looks like the example. It should include the name of the song, and the artist, separated by a comma."
	# else:
	# 	pass
	start = time.time()
	user_a_in_db, user_a_to_get = sort_inputs(query_a)
	user_b_in_db, user_b_to_get = sort_inputs(query_b)
	end = time.time()
	print(f"Sort Inputs: {end-start}")

	start = time.time()
	metadata_a = gather_metadata(user_a_to_get)
	metadata_b = gather_metadata(user_b_to_get)
	end = time.time()
	print(f"Gather Metadata: {end-start}")

	start = time.time()
	not_null_a, spotify_features_a = get_spotify_features(metadata_a)
	not_null_b, spotify_features_b = get_spotify_features(metadata_b)
	end = time.time()
	print(f"Get Spotify Features: {end-start}")


	# i = len(list(not_null_a.keys())) 
	# j = len(list(not_null_b.keys()))

	# track_list = (list(not_null_a.keys()) + list(not_null_b.keys()))
	start = time.time()
	track_list = prepare_tracklist(not_null_a, not_null_b)
	end = time.time()
	print(f"Prepare Tracklist: {end-start}")

	# ====== MULTIPROCESSING METHOD GOES HERE ====== #
	
	start = time.time()
	# POOL MAP
	# result = pool.map(single_librosa_pipeline, track_list)

	# REGULAR
	# result = [single_librosa_pipeline(t) for t in track_list]

	# # APPLY ASYNC ---fastest so far
	result_object = [pool.apply_async(single_librosa_pipeline, args=(x,)) for x in track_list]
	

	# POOL IMAP
	# result_object = pool.imap(single_librosa_pipeline, track_list)
	# result = [r for r in result_object]

	end = time.time()
	print(f"Librosa Pipeline: {end-start}")
	# ============================================== #
	start = time.time()
	librosa_features_a, librosa_features_b = parse_result(not_null_a, not_null_b, result_object)
	end = time.time()
	print(f"Parse Result: {end-start}")
	# librosa_features_a = result[:i]
	# librosa_features_b = result[-j:]

	start = time.time()
	all_features_a = combine_all_features(metadata_a, librosa_features_a, spotify_features_a)
	all_features_b = combine_all_features(metadata_b, librosa_features_b, spotify_features_b)
	end = time.time()
	print(f"Combine All Features: {end-start}")

	start = time.time()
	user_a_df = generate_user_df(user_a_in_db, all_features_a)
	user_b_df = generate_user_df(user_b_in_db, all_features_b)
	end = time.time()
	print(f"Generate User DF: {end-start}")


	# # Mapping generalized genres to df

	start = time.time()
	user_a_df = remap_genres(user_a_df)
	user_b_df = remap_genres(user_b_df)
	end = time.time()
	print(f"Remap Genres: {end-start}")

	# user_a_ids = list(user_a_in_db['track_id'].values)
	# user_b_ids = list(user_b_in_db['track_id'].values)
	# Finding most similar songs to each user's input
	start = time.time()
	user_a_recs = get_similar_track_ids(user_a_df)
	user_b_recs = get_similar_track_ids(user_b_df)
	end = time.time()
	print(f"Get Similar Track IDs: {end-start}")

	start = time.time()	
	user_a_index, user_a_array = get_feature_vector_array(user_a_recs)
		
	user_b_index, user_b_array = get_feature_vector_array(user_b_recs)
	end = time.time()
	print(f"Get Feature Vector Array: {end-start}")


	start = time.time()
	cosine_df = create_similarity_matrix(user_a_array,
										 user_a_index,
										 user_b_array,
										 user_b_index)
	end = time.time()
	print(f"Create Similarity Matrix: {end-start}")

	start = time.time()
	recommendations = get_combined_recommendations(cosine_df)
	end = time.time()
	print(f"Get Recommendations: {end-start}")
	
	# # if len(no_preview) > 0:
	# # 	print("Could not get recommendations for:")
	# # 	for i in no_preview.values():
	# # 		print(f"{i[0]}, by {i[1]}")
	
	return recommendations

if __name__ ==  '__main__':

	with mp.Pool(processes=4) as pool:
		
		#===== TEST 1: POOL USING MAP =====#
		# start_1 = time.time()
		
		# r = friendship_app(query_a,query_b)

		# end_1 = time.time()
		# print(f"Pool Map -- Total: {end_1 - start_1}")
		# print(r)

	
		#===== TEST 2: REGULAR PROCESS =====# 
		# start_2 = time.time()
		
		# r = friendship_app(query_a,query_b)

		# end_2 = time.time()
		# print(f"Regular -- Total: {end_2 - start_2}")
		# print(len(r))
	
		
		#===== TEST 4: APPLY ASYNC =====#
		start_4 = time.time()

		r = friendship_app(query_a,query_b)

		end_4 = time.time()
		print(f"Pool ApplyAsync -- Total: {end_4 - start_4}")
		# print(len(r))

		#===== TEST 5: POOL IMAP =====#
		# start_5 = time.time()
		
		# friendship_app(query_a,query_b)

		# end_5 = time.time()
		# print(f"Pool iMap -- Total: {end_5 - start_5}")
		# print(len(r), len(s))