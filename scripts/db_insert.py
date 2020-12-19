import psycopg2 as pg
import pandas as pd 
from secret import *


tracks = pd.read_csv('data/tracks.csv', sep=';', index_col=0)
print(tracks.columns)

# def run_query(q):
#     with conn:
#         try:
#             cur = conn.cursor()
#             cur.execute(q)
#             return pd.read_sql(q, conn)

#         except (Exception, pg.DatabaseError) as error:
#             print(error)

# def run_command(c):
#     with conn:
#         try:
#             cur = conn.cursor()
#             cur.execute(c)
#             cur.close()
#             conn.commit()
            
#         except (Exception, pg.DatabaseError) as error:
#             print(error)

# cur = conn.cursor()

# err = []
# for i in range(len(tracks)):
#     record = tuple(tracks.iloc[i,:].values)
    
#     new_record = tuple(elem if str(elem)!='nan' else None for elem in record)
    
#     try:
#         cur.execute("INSERT INTO track_metadata (track_id, track_name, artist, artist_id, genre_1, genre_2, genre_3) VALUES (%s, %s, %s, %s, %s, %s, %s)", new_record)
#         conn.commit()
        
#     except:
#         err.append(new_record)
#         pass