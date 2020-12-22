import psycopg2 as pg
import pandas as pd 
from secret import *
import os

conn = pg.connect(database=sql_credentials['database'],
				  user=sql_credentials['user'], 
				  password=sql_credentials['password'])

tracks = pd.read_csv('data/tracks_genre.csv', sep=';', index_col=0)

def run_query(q):
	with conn:
		try:
			cur = conn.cursor()
			cur.execute(q)
			return pd.read_sql(q, conn)

		except (Exception, pg.DatabaseError) as error:
			print(error)

def run_command(c):
	with conn:
		try:
			cur = conn.cursor()
			cur.execute(c)
			cur.close()
			conn.commit()
			
		except (Exception, pg.DatabaseError) as error:
			print(error)

cur = conn.cursor()

# list to catch songs that were not inserted
err = []
for i in range(len(tracks)):
	record = tuple(tracks.iloc[i,:].values)
	
	new_record = tuple(elem if str(elem)!='nan' else None for elem in record)
	
	try:
		cur.execute("""INSERT INTO track_metadata (track_id, track_name, artist, artist_id, track_album_album_type, track_album_id,
												   track_album_name, track_duration_ms, track_popularity, track_preview_url, 
												   subgenres, genre, top_genre) 
										   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
										   ON CONFLICT DO NOTHING""", new_record)
		conn.commit()
		
	except:
		err.append(new_record)
		pass

	os.system('cls')
	print(f"{i+1}/{len(tracks)} records inserted")


with open('failed_insert.txt', 'w') as file:
	for e in err:
		file.write('%s\n' % e)
