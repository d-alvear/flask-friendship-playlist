#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, redirect, Response, Markup
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
from plotly.offline import plot
import plotly.graph_objs as go

# connect to spotify_db
# conn = pg.connect(database=sql_credentials['database'],
#                   user=sql_credentials['user'], 
#                   password=sql_credentials['password'],
#                   host=sql_credentials['host'])

conn = pg.connect(database='spotify_db',
					user='postgres',
					password=)


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


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/demo')
def test_cases():
	pr_fig = go.Figure()
	pr_fig.add_trace(go.Violin(x=pop_rock['trace_1']['x'],y=pop_rock['trace_1']['y'],name="Jackie's Pop Songs"))
	pr_fig.add_trace(go.Violin(x=pop_rock['trace_2']['x'],y=pop_rock['trace_2']['y'],name="Donna's Rock Songs"))
	pr_fig.add_trace(go.Violin(x=pop_rock['trace_3']['x'],y=pop_rock['trace_3']['y'],name="Recommended Songs"))
	pr_fig.update_traces(box_visible=True, meanline_visible=True)
	pr_fig.update_layout(title={'text': "Comparison of Audio Features",'yanchor':'middle','x':0.52},
					  	 		xaxis_title="Audio Features",yaxis_title="Normalized Feature Values",violinmode='group',font=dict(size=12),
					  	 		autosize=False,width=700,height=450,margin=dict(l=50,r=10,b=100,t=48,pad=1),
						legend={'orientation': "h",'y':-0.2,'x':0.1})
	pr_plot = plot(pr_fig,output_type='div')

	rhh_fig = go.Figure()
	rhh_fig.add_trace(go.Violin(x=rock_hh['trace_1']['x'],y=rock_hh['trace_1']['y'],name="Hyde's Rock Songs"))
	rhh_fig.add_trace(go.Violin(x=rock_hh['trace_2']['x'],y=rock_hh['trace_2']['y'],name="Fez's Hip Hop Songs"))
	rhh_fig.add_trace(go.Violin(x=rock_hh['trace_3']['x'],y=rock_hh['trace_3']['y'],name="Recommended Songs"))
	rhh_fig.update_traces(box_visible=True, meanline_visible=True)
	rhh_fig.update_layout(title={'text': "Comparison of Audio Features",'yanchor':'middle','x':0.52},
					  	 		xaxis_title="Audio Features",yaxis_title="Normalized Feature Values",violinmode='group',font=dict(size=12),
					  	 		autosize=False,width=700,height=450,margin=dict(l=50,r=10,b=100,t=48,pad=1),
						legend={'orientation': "h",'y':-0.2,'x':0.1})
	rhh_plot = plot(rhh_fig,output_type='div')

	mx_fig = go.Figure()
	mx_fig.add_trace(go.Violin(x=mixed['trace_1']['x'],y=mixed['trace_1']['y'],name="Jackie, Fez, and Hyde's Songs"))
	mx_fig.add_trace(go.Violin(x=mixed['trace_2']['x'],y=mixed['trace_2']['y'],name="Eric, Kelso, and Donna's Songs"))
	mx_fig.add_trace(go.Violin(x=mixed['trace_3']['x'],y=mixed['trace_3']['y'],name="Recommended Songs"))
	mx_fig.update_traces(box_visible=True, meanline_visible=True)
	mx_fig.update_layout(title={'text': "Comparison of Audio Features",'yanchor':'middle','x':0.52},
					  	 		xaxis_title="Audio Features",yaxis_title="Normalized Feature Values",violinmode='group',font=dict(size=12),
					  	 		autosize=False,width=700,height=450,margin=dict(l=50,r=10,b=100,t=48,pad=1),
						legend={'orientation': "h",'y':-0.2,'x':0.1})
	mx_plot = plot(mx_fig,output_type='div')


	return render_template('demo.html',div_placeholder_1=Markup(pr_plot), div_placeholder_2=Markup(rhh_plot), div_placeholder_3=Markup(mx_plot))

@app.route('/results', methods=['POST','GET'])
def friendship_app():
	
	query_a = request.form['query_a']
	query_b = request.form['query_b']
	print(query_a)
	print(query_b)

	# REWRITE THIS TO CHECK BOTH USERS' QUERIES
	for q in [query_a, query_b]:
		if check_query_format(q) == False:
			return render_template('error.html')
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
	recommendations.rename(columns={'track_name':'Track Name',
									'artist':'Artist',
									'genre':'Genre'},inplace=True)
	user_a_df.rename(columns={'track_name':'Track Name',
							  'artist':'Artist',
							  'genre':'Genre'},inplace=True)
	user_b_df.rename(columns={'track_name':'Track Name',
							  'artist':'Artist',
							  'genre':'Genre'},inplace=True)
	
	#generate result plot
	combined = format_dataframe(user_a_df,user_b_df,recommendations)
	plot_div = generate_plot(combined)
	print("finished!")
	return render_template('results.html', 
							user_a_table=user_a_df[['Track Name','Artist','Genre']],
							user_b_table=user_b_df[['Track Name','Artist','Genre']],
							rec_table=recommendations[['Track Name','Artist','Genre']],
							div_placeholder=Markup(plot_div)
							)

if __name__ == '__main__':
	app.run()