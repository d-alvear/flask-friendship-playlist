#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import sys
import os
import psutil
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
# import pickle

# connect to spotify_db
conn = pg.connect(database="spotify_db",
				  user="postgres", 
				  password="damara1004")


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
													  client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

query_a = "call me maybe, carly rae jepsen; dancing queen, ABBA; all the small things, blink-182"
query_b = "electric feel, MGMT; got to give it up, marvin gaye; hotel california, eagles"

# query_a = "bad romance, lady gaga; fantasy, mariah carey; cry me a river, justin timberlake"
# query_b = "smells like teen spirit, nirvana; island in the sun, weezer; never let you go, third eye blind"
# query_a = "the middle, jimmy eat world; mr. brightside, the killers; californication, red hot chili peppers"
# query_b = "hey ya, outkast; in my feelings, drake; waterfalls, tlc"

def friendship_app(query_a,query_b):
	# checks query format for both inputs
	# if check_query_format(query) == False:
	# 	return "Your query doesn't look quite right, make sure it looks like the example. It should include the name of the song, and the artist, separated by a comma."
	# else:
	# 	pass
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

	print(list(user_a_df.loc[:,'track_id']))
	print(list(user_b_df.loc[:,'track_id']))
	# not_null_list = []
	# spotify_features_list = []
	# metadata_list = []
	# for user_list in [user_a_to_get, user_b_to_get]:
	# 	if len(user_list) == 0:
	# 		print("Skipped")
	# 		not_null_list.append(None)
	# 		spotify_features_list.append(None)
	# 		metadata_list.append(None)
	# 		pass
		
	# 	elif len(user_list) > 0:
	# 		print("Executing Block")
	# 		metadata = gather_metadata(user_list)
	# 		not_null, spotify_features = get_spotify_features(metadata)
	# 		not_null_list.append(not_null)
	# 		spotify_features_list.append(spotify_features)
	# 		metadata_list.append(metadata)
	
	# del user_a_to_get
	# del user_b_to_get

	# librosa_list = []
	# if not_null_list != [None, None]:
	# 	track_list = prepare_tracklist(not_null_list)
	# 	result_object = [pool.apply_async(single_librosa_pipeline, args=(x,)) for x in track_list]
	# 	librosa_features_a, librosa_features_b = parse_result(not_null_list[0], not_null_list[1], result_object)
	# 	librosa_list.append(librosa_features_a)
	# 	librosa_list.append(librosa_features_b)
	# else:
	# 	print("Nothing to see here!")



	# ====== MULTIPROCESSING METHOD GOES HERE ====== #
	# POOL MAP
	# result = pool.map(single_librosa_pipeline, track_list)

	# REGULAR
	# result = [single_librosa_pipeline(t) for t in track_list]

	# # APPLY ASYNC ---fastest so far
	# result_object = [pool.apply_async(single_librosa_pipeline, args=(x,)) for x in track_list]


	# POOL IMAP
	# result_object = pool.imap(single_librosa_pipeline, track_list)
	# result = [r for r in result_object]

	# ============================================== #
	# user_a_len = len(user_a_in_db)
	# user_b_len = len(user_b_in_db)
	# for user_list in [user_a_len, user_b_len]:
	# 	if user_list < 3:
	# user_df_list = []
	# if len(librosa_list) > 0:
	# 	print("generating features")
	# 	for m,l,s,i in zip(metadata_list, librosa_list, spotify_features_list, [user_a_in_db,user_b_in_db]):
	# 		all_features = combine_all_features(m,l,s)
	# 		user_df = generate_user_df(i,all_features)
	# 		user_df = remap_genres(user_df)
	# 		user_df_list.append(user_df)
	# else:
	# 	print("nothing to generate")
	# 	user_df_list = [user_a_in_db, user_b_in_db]
		# ----->  user_df_list = in_db
	# all_features_a = combine_all_features(metadata[0], librosa_features_a, spotify_features[0])
	# all_features_b = combine_all_features(metadata[1], librosa_features_b, spotify_features[1])


	# user_a_df = generate_user_df(user_a_in_db, all_features_list[0])
	# user_b_df = generate_user_df(user_b_in_db, all_features_list[1])


			# # Mapping generalized genres to df
	# user_a_df = remap_genres(user_a_df)
	# user_b_df = remap_genres(user_b_df)


	# user_a_ids = list(user_a_in_db['track_id'].values)
	# user_b_ids = list(user_b_in_db['track_id'].values)
	# Finding most similar songs to each user's input
	
	user_a_recs = [get_similar_track_ids(user_a_df, user_a_in_db)]
	user_b_recs = [get_similar_track_ids(user_b_df, user_b_in_db)]
	# user_b_recs = get_similar_track_ids(user_df_list[1],user_b_in_db,similarities)


	user_a_index, user_a_array = get_feature_vector_array(user_a_recs)
		
	user_b_index, user_b_array = get_feature_vector_array(user_b_recs)

	cosine_df = create_similarity_matrix(user_a_array,
										 user_a_index,
										 user_b_array,
										 user_b_index)
	
	recommendations = get_combined_recommendations(cosine_df)
	
	
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

	
		#===== TEST 2: REGULAR PROCESS =====# 
		# start_2 = time.time()
		
		# r = friendship_app(query_a,query_b)

		# end_2 = time.time()
		# print(f"Regular -- Total: {end_2 - start_2}")
		# print(len(r))
	
		
		#===== TEST 4: APPLY ASYNC =====#
		process = psutil.Process(os.getpid())
		start_4 = time.time()

		r = friendship_app(query_a,query_b)

		end_4 = time.time()
		print(f"Pool ApplyAsync -- Total: {end_4 - start_4}")
		memoryUse = process.memory_info()[0] / float(2 ** 20)  # memory use in GB...I think
		print('memory use:', memoryUse)
		print(r)

		#===== TEST 5: POOL IMAP =====#
		# start_5 = time.time()
		
		# friendship_app(query_a,query_b)

		# end_5 = time.time()
		# print(f"Pool iMap -- Total: {end_5 - start_5}")
		# print(len(r), len(s))