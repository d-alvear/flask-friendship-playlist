#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import environ
from flask import Flask, request, render_template, redirect
from secret import sql_password
import friendship_app as fp
import psycopg2 as pg
import project_utils as pu
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

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
		# sp = spotipy.Spotify(auth=token)
		return redirect("/index")
	else:
		return "No token"

@app.route('/index')
def homepage():
	return render_template('index.html')

@app.route('/index', methods=['POST'])
def friendship_app():
	seed = request.form['seed']
	check = fp.check_database(str(seed))
	if check==True:
		res = fp.in_database(str(seed))
		return render_template('index.html',  tables=[res.to_html(classes='data', header="true")])
	elif check==False:
		res = fp.not_in_database(str(seed))
		return render_template('index.html',  tables=[res.to_html(classes='data', header="true")])

if __name__ == '__main__':
	app.run()
	