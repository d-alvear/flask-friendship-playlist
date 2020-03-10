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
	seed = request.form['seed']

	if check_database(str(seed)) == True:
		rec_in_db = in_database(seed)
		return rec_in_db.to_html()
		# return render_template('index.html', tables=[rec_in_db.to_html()])
	
	elif check_database(str(seed)) == False:
		rec_not_in_db = not_in_database(seed)
		return rec_not_in_db.to_html()
		# return render_template('index.html', tables=[rec_not_in_db.to_html()])
if __name__ == '__main__':
	app.run()
	