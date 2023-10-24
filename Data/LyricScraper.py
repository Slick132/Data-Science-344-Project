import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


# My API token is private, create a genius account to test our scraper
api_token = ""

def fetch_lyrics_genius(genius_api_token, artist, song_title):
    search_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {genius_api_token}"}
    params = {"q": f"{song_title} {artist}"}

    # Step 1: Search for the song using Genius API
    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code != 200:
        print("Genius API Error")
        print(response.status_code)
        return None

    search_results = response.json()["response"]["hits"]
    if not search_results:
        return None

    song_url = search_results[0]["result"]["url"]

    # Step 2: Scrape the lyrics
    response = requests.get(song_url)
    if response.status_code != 200:
        print("Genius API Error")
        print(response.status_code)
        return None
    html = BeautifulSoup(response.text, 'html.parser')

    lyrics_divs = html.findAll('div', {'data-lyrics-container': "true"})
    all_lyrics = ""

    for lyrics_div in lyrics_divs:
        lyrics_text = re.findall(r'>([^<]+)<', str(lyrics_div))
        all_lyrics += " ".join(lyrics_text) + "\n"

    # Remove identifiers like chorus, verse, etc.
    all_lyrics = re.sub(r'[\(\[].*?[\)\]]', '', all_lyrics)
    # Remove empty lines
    all_lyrics = os.linesep.join([s for s in all_lyrics.splitlines() if s])
    return all_lyrics

def check_the(song, artist):
    artist = remove_the(artist)
    url = f"https://www.officialcharts.com/songs/{artist}-{song}"
    response = requests.get(url)
    if response.status_code == 200:
        return True
    if response.status_code != 404:
        print("Official Charts error message:")
        print(response.status_code)
    return False

def hit(song, artist):
    def move_characters_to_end(input_string, index):
        if 0 <= index < len(input_string):
            return input_string[index + 1:] + ' ' + input_string[:index]
        else:
            return input_string
    
    if ('(feat.' in song):
        index = song.index('(feat.') - 1
        song = move_characters_to_end(song, index)
    
    song = song.replace('feat.', 'ft')
    song = re.sub(r"t's", "ts", song)
    song = re.sub(r"'", " ", song)
    song = re.sub(r'[^a-zA-Z0-9\s]', '', song)
    song = song.replace(' ','-')
    artist = re.sub(r'[^a-zA-Z0-9\s]', '', artist)
    artist = artist.replace(' ','-')
    url = f"https://www.officialcharts.com/songs/{artist}-{song}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return True
    else:
        return check_the(song, artist)

def clean_artist(artist_name):
    artist_name = artist_name.replace(" ", "").lower()  # Get the first artist name and remove all white spaces
    artist_name = remove_the(artist_name)
    return artist_name

def clean_song(song_name):
    song_name = song_name.replace(" ", "").lower()    # remove the spaces
    song_name = song_name.replace(",", "")                      # remove the commas
    song_name = song_name.replace("'", "")                      # remove apostriphe
    song_name = song_name.replace('"', "")
    song_name = song_name.replace('`', "")
    song_name = song_name.replace("?", "")
    song_name = song_name.replace("!", "")
    song_name = song_name.replace("&", "")
    song_name = song_name.replace("’", "")
    try:
        copy_until = song_name.index('-')
        song_name = song_name[:copy_until]
    except ValueError:
        song_name = song_name

    try:
        copy_until = song_name.index('(')
        if song_name[copy_until+1:copy_until+5] == 'with' or song_name[copy_until+1:copy_until+5] == 'feat':
            song_name = song_name[:copy_until]
        else:
            song_name = song_name.replace("(", "")
            song_name = song_name.replace(")", "")
    except ValueError:
        song_name = song_name
    return song_name

def remove_the(s):
    return s[4:] if s.startswith("The-") else s

def main():
    # Read in file
    df = pd.read_csv('The ONE.csv')
    output_file_name = "Results.csv"
    # Check if "lyrics" column exists, if not, add it
    if 'lyrics' not in df.columns:
        df['lyrics'] = ''

    # Check if "hit" column exists, if not, add it
    if 'hit' not in df.columns:
        df['hit'] = ''
    counter = 1
    already_scraped = False
    if os.path.exists(output_file_name):
        df = pd.read_csv(output_file_name)
        distinct_artists = df["artist_name"].dropna().unique().tolist()
        already_scraped = True

    for index, row in df.iterrows():
        if pd.isnull(row['lyrics']) or row['lyrics'] == '':
            try:
                if isinstance(row['artist_name'], float):
                    continue
                if isinstance(row['track_name'], float):
                    continue
                artist_name = clean_artist(row['artist_name'])
                song_name = clean_song(row['track_name'])
                lyrics = fetch_lyrics_genius(api_token, artist_name, song_name)
                if lyrics is None:
                    print(" *** Could not find the lyrics for: " + str(song_name) + " ***")
                if lyrics is not None:
                    print(" = = = Got lyrics for: " + str(song_name) + " = = = ")
                is_hit = hit(row['track_name'], row['artist_name'])
                print(str(song_name) + " is hit = " + str(is_hit))
                df.at[index, 'hit'] = is_hit
                df.at[index, 'lyrics'] = lyrics
                print("Writing to csv: " + str(song_name) + " by " + str(artist_name))
                df.to_csv(output_file_name, header=True, index=False)
                print("Scraped " + str(counter) + str(" songs"))
                counter += 1
            except:
                continue

main()
