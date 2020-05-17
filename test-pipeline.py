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
from plotly.offline import plot
import plotly.graph_objs as go

# connect to spotify_db
conn = pg.connect(database="spotify_db",
				  user="postgres", 
				  password="")


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = SpotifyClientCredentials(client_id=spotify_credentials['client_id'],
													  client_secret=spotify_credentials['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# query_a = "call me maybe, carly rae jepsen; dancing queen, ABBA; all the small things, blink-182; on the radio, donna summer; hot stuff, donna summer"
# query_b = "electric feel, MGMT; got to give it up, marvin gaye; hotel california, eagles; real love baby, father john misty"

# query_a = "bad romance, lady gaga; fantasy, mariah carey; cry me a river, justin timberlake"
# query_b = "smells like teen spirit, nirvana; island in the sun, weezer; never let you go, third eye blind"
# query_a = "the middle, jimmy eat world; mr. brightside, the killers; californication, red hot chili peppers"
# query_b = "hey ya, outkast; in my feelings, drake; waterfalls, tlc"

# query_a = "don't start now, dua lipa; dance dance, fall out boy; zombie, the cranberries"
# query_b = "toxic, britney spears; boredom, tyler the creator; the passenger, Siouxsie and the Banshees"

query_a = "groove is in the heart, deelite; i wish, skeelo; don quichotte, magazine 60"
query_b = "roses, outkast; P.I.M.P, 50 cent; thriller, michael jackson"

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

	# Finding most similar songs to each user's input
	
	user_a_recs = [get_similar_track_ids(user_a_df, user_a_in_db)]
	user_b_recs = [get_similar_track_ids(user_b_df, user_b_in_db)]

	user_a_recs_i = tuple(user_a_recs[0])
	q = f'''SELECT track_id, track_name, artist, genre
			FROM track_metadata
			WHERE track_id IN {user_a_recs_i}
			'''
	a = run_query(q)
	# print(a)

	user_b_recs_i = tuple(user_b_recs[0])
	q = f'''SELECT track_id, track_name, artist, genre
			FROM track_metadata
			WHERE track_id IN {user_b_recs_i}
			'''
	b = run_query(q)
	# print(b)


	user_a_index, user_a_array = get_feature_vector_array(user_a_recs)
		
	user_b_index, user_b_array = get_feature_vector_array(user_b_recs)

	cosine_df = create_similarity_matrix(user_a_array,
										 user_a_index,
										 user_b_array,
										 user_b_index)
	
	cosine_df_names = create_similarity_matrix(user_a_array,
										 list(a.loc[:,'track_name']),
										 user_b_array,
										 list(b.loc[:,'track_name']))
	print(a.loc[:,['track_name','artist']])
	print(b.loc[:,['track_name','artist']])
	
	# headers = ["User A/User B","Nice to Meet Ya, Meghan Trainor","Genius, Sia","Star67, Drake","Uprising, Muse","Born For Greatness, Papa Roach","Can We, Phonte"]
	# rows = ["Boyfriend, Mabel","Please Don't Go, Mike Posner","Paradise Lost, The Used","The Kill, Thirty Seconds To Mars","Something Changed, Pulp","American Dreams, Papa Roach"]
	# cosine_df_names = cosine_df_names.round(4)
	# # headers = headers.insert(0,"User A/User B")
	# fig = go.Figure(data=[go.Table(columnwidth = [20,20],
    # header=dict(values=headers,
    #             fill_color='palegreen',
	# 			line_color='black',
    #             align=['left','center'],),
    # cells=dict(values=[rows,cosine_df_names.loc['Boyfriend',:], cosine_df_names.loc["Please Don't Go",:], cosine_df_names.loc["Paradise Lost, a poem by John Milton",:], 
	# 				   cosine_df_names.loc['The Kill (Bury Me)',:],cosine_df_names.loc['Something Changed',:],cosine_df_names.loc['American Dreams',:]],
    #            fill=dict(color=['palegreen', 'white']),
	# 		   line_color='black',
    #            align='left'))
	# 		   ])

	# fig.show()
	
	recommendations = get_combined_recommendations(cosine_df)
	
	# # combined = format_dataframe(user_a_df,user_b_df,recommendations)
	# # print(list(combined.loc[:,'feature']))
	# # print(list(combined.loc[:,'value']))
	
	return recommendations

if __name__ ==  '__main__':

	start_4 = time.time()

	r = friendship_app(query_a,query_b)

	end_4 = time.time()
	print(f"Total: {end_4 - start_4}")
	print(r)

		#===== TEST 5: POOL IMAP =====#
		# start_5 = time.time()
		
		# friendship_app(query_a,query_b)

		# end_5 = time.time()
		# print(f"Pool iMap -- Total: {end_5 - start_5}")
		# print(len(r), len(s))