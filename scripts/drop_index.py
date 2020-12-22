import pandas as pd

file = "tracks_genre.csv"

df = pd.read_csv(f"data/{file}", sep=";", index_col=0)

out = "tracks_genre.csv"
df.to_csv(f"data/{out}", sep=";", index=False)