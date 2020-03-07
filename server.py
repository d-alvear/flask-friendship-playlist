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
scope = 'playlist-modify-public'
if len(sys.argv) > 1:
    username = sys.argv[1]
else:
	print("Sorry")


app = Flask(__name__, static_folder='static', template_folder='views')

@app.route('/')
def authenticate_user():
	token = util.prompt_for_user_token(username, 
									scope, 
									client_id=environ.get('SPOTIPY_CLIENT_ID'), 
									client_secret=environ.get('SPOTIPY_CLIENT_SECRET'), redirect_uri=
									'http://127.0.0.1')
	sp = spotipy.Spotify(auth=token)
	return redirect('/index')

@app.route('/index')
def homepage():
	return render_template('index.html')

@app.route('/index', methods=['POST'])
def friendship_app():
	seed = request.form['seed']
	check = check_database(str(seed))
	
	if check==True:
		recs = in_database(str(seed))
		user_all_data = sp.current_user()
		return user_all_data
		# playlist = create_playlist(sp,recs)
		# return render_template('playlist.html', playlist=playlist)
	elif check==False:
		recs = not_in_database(str(seed))
		playlist = create_playlist(sp,recs)
		return render_template('playlist.html', playlist=playlist)

if __name__ == '__main__':
	app.run()
	