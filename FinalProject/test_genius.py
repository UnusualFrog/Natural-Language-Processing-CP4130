import lyricsgenius

token = "1Y4W8rda4lIgTZL8djnafAnIrZqH-YsiyxVRmdx7Cii1m_h5XO-uWzwjxhnRZZVl"

genius = lyricsgenius.Genius(token)

# artist = genius.search_artist("Gorillaz", max_songs=3, sort="title")
# print(artist.songs)

# song = genius.search_song(title="Dumb Litty")

# print(song)

# for k,v in song.items():
#     print(k,v)
input = -1
while input != 0:
    print("Please enter the title of a song: ")
    song_name = str(input())

    song_name = song_name.strip()

    song = genius.search_song(title=song_name)

    print(song)

    print("Select an option")
    print("0. Exit Program")
    print("1. Predict new song")
    input = input()
    