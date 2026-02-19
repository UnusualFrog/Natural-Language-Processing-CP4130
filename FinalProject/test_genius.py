import lyricsgenius

token = "1Y4W8rda4lIgTZL8djnafAnIrZqH-YsiyxVRmdx7Cii1m_h5XO-uWzwjxhnRZZVl"

genius = lyricsgenius.Genius(token)

# artist = genius.search_artist("Gorillaz", max_songs=3, sort="title")
# print(artist.songs)

song = genius.search_song(title="Dumb Litty")

print(song)

# for k,v in song.items():
#     print(k,v)