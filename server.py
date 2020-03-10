#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os import environ
from flask import Flask, request, render_template, redirect
from project_utils import *
import pandas as pd
import numpy as np
import psycopg2 as pg
from psycopg2 import Error
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
import librosa
import spotipy
import spotipy.util as util
import requests
import pickle

# connect to spotify_db
conn = pg.connect(database="spotify_db",
				  user="postgres", 
				  password=sql_password)


# Authenticate with Spotify using the Client Credentials flow
client_credentials_manager = SpotifyClientCredentials(client_id=environ.get('SPOTIPY_CLIENT_ID'),client_secret=environ.get('SPOTIPY_CLIENT_SECRET'))
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

@app.route('/search', methods=['POST','GET'])
def friendship_app():
	query = request.form['query']
	#load clustering model
	km = pickle.load(open('kmeans_model.pkl','rb'))

	in_db, not_in_db = sort_inputs(query)

	in_db_df = in_database(in_db)
	not_in_db_df = not_in_database(not_in_db)
	not_in_db_df = scale_features(not_in_db_df)
    
	input_df = combine_frames(in_db, not_in_db, in_db_df, not_in_db_df)

	cluster_df, new_fv = get_cluster_df(input_df, km)

	results = get_results(cluster_df, new_fv, input_df)
	return results.to_html()


if __name__ == '__main__':
	app.run()
	