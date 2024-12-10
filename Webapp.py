import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import pandas as pd 
import pickle
from bs4 import BeautifulSoup
import numpy as np
import requests
import re
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


with st.spinner('Loading models...'):
    model = keras.saving.load_model('Models/prediction_model.keras')

    with open('Models/tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('Models/rank_encoder.pkl', 'rb') as file:
        rank_encoder = pickle.load(file)

    with open('Models/popularity_encoder.pkl', 'rb') as file:
        popularity_encoder = pickle.load(file)

    with open('Models/score_encoder.pkl', 'rb') as file:
        score_encoder = pickle.load(file)

def get_genres():
    genres_types = ['Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love', 
                  'Comedy', 'Drama', 'Ecchi', 'Erotica', 'Fantasy', 'Girls Love', 
                  'Gourmet', 'Hentai', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 
                  'Slice of Life', 'Sports', 'Supernatural', 'Suspense']
    return genres_types

def get_studios():
    top_studios = ["Toei Animation", "Sunrise", "J.C.Staff", "Madhouse", 
        "TMS Entertainment", "Production I.G", "Studio Deen", 
        "Pierrot", "OLM", "Shin-Ei Animation", "A-1 Pictures", 
        "Nippon Animation", "AIC", "DLE", "Tatsunoko Production", "Trigger"]
    return top_studios

def get_producers():
    top_producers = ["Aniplex", "TV Tokyo", "Lantis", "Movic", 
                 "AT-X", "Bandai Visual", "Pony Canyon", "Kadokawa", 
                 "Dentsu", "Fuji TV", "NHK", "Sotsu", "KlockWorx", "Kodansha", "Shueisha"]
    return top_producers

def get_source():
    source_columns = ['Source_Book', 'Source_Game', 'Source_Light novel', 'Source_Manga', 'Source_Original', 'Source_Visual novel']
    return source_columns

def get_type():
    type_columns = ['Types_Movie','Types_Music','Types_ONA','Types_OVA','Types_Special','Types_TV']
    return type_columns

def scrape_all(anime_id):
    url = f"https://myanimelist.net/anime/{anime_id}"
    response = requests.get(url, timeout = (10, 15))
    
    if response.status_code != 200:
        st.write("Please Enter a Different ID")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    page_text = soup.get_text(separator="\n", strip=True)
    
    return page_text

def process_anime(anime_data):
    anime_list = anime_data.split('\n')
    anime_dict = {}

    # Add Title
    title = anime_list[0].split(' - ')[0]
    anime_list.pop(0)
    anime_dict['title'] = title

    # Get Type, Episodes, English, Source
    def parse_list(param, anime_list):
        key = param.replace(':', '').strip().lower()
        for i in range(len(anime_list)):
            if anime_list[i].startswith(param):
                anime_dict[key] = anime_list[i + 1].strip() 
                break
    
    parse_list('Type:', anime_list)
    parse_list('Episodes:', anime_list)
    if anime_dict.get('episodes') == 'Unknown':
        anime_dict['episodes'] = 12
    parse_list('English:', anime_list)
    parse_list('Source:', anime_list)

    # Get Synopsis
    synopsis_list = []
    found_index = -1

    for i in range(1, len(anime_list)):
        if anime_list[i] == "Synopsis" and "Edit" in anime_list[i - 1]:
            found_index = i
            break
            
    if found_index != -1:
        for i in range(found_index + 1, len(anime_list)):
            if (anime_list[i].startswith("[Written by") or 
                anime_list[i].startswith("Related Entries") or 
                anime_list[i].startswith("Background") or
                anime_list[i].startswith("Edit")):
                break
            synopsis_list.append(anime_list[i])

    combined_synopsis = " ".join((synopsis_list))
    anime_dict['synopsis'] = combined_synopsis

    def get_multiple(topic_list, anime_list):
        found = set()
        for entry in anime_list:
            if entry in topic_list:
                found.add(entry)
        result = ", ".join(found)
        return result

    anime_dict['genres'] = get_multiple(get_genres(), anime_list)
    anime_dict['studios'] = get_multiple(get_studios(), anime_list)
    anime_dict['producers'] = get_multiple(get_producers(), anime_list)

    df = pd.DataFrame([anime_dict])
    return df

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    custom_words = {'and', 'the', 'is', 'a', 'to', 'it', 's', 'like', 'year'}
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in custom_words) + r')\b'

    def is_capitalized(word):
        return word[0].isupper() and word.isalpha()


    if pd.isnull(text): 
        return text
    words = word_tokenize(text)

    lemmatized_words = [
        lemmatizer.lemmatize(word.lower()) for word in words
        if word.lower() not in stop_words and word.lower() not in custom_words and not is_capitalized(word)
    ]
    lemmatized_text = ' '.join(lemmatized_words)

    ## Futher clean anything lemmatization missed, remove spaces and characters
    cleaned_text = re.sub(r'[^\w\s]', '', lemmatized_text)
    cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def process_features(X_test):
    # Vectorize and lemmatize synopsis
    X_test['synopsis'] = X_test['synopsis'].apply(lemmatize_text)
    synopsis_tfidf = vectorizer.transform(X_test['synopsis'])
    tfidf_df = pd.DataFrame(synopsis_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.columns = ['tfidf_' + col for col in tfidf_df.columns]
    X_test = pd.concat([X_test, tfidf_df], axis=1)

    # Hot encode producers, studios, genres
    top_producers = get_producers()
    top_studios = get_studios()
    genres_types = get_genres()

    def hot_encode(topic_list, topic):
        for entry in topic_list:
            X_test[f'{topic}_{entry.replace(" ", "_").lower()}'] = False

        for index, row in X_test.iterrows():
            entries_in_row = [entry.strip() for entry in row[topic].split(',')]
            
            for entry in topic_list:
                column_name = f'{topic}_{entry.replace(" ", "_").lower()}'
                if entry in entries_in_row:
                    X_test.at[index, column_name] = True
    
    hot_encode(top_producers, "producers")
    hot_encode(top_studios, "studios")
    hot_encode(genres_types, "genres")

    # Hot encode source and type
    X_test['source'] = X_test['source'].replace('Unknown', np.nan)
    X_test['source'] = X_test['source'].replace('Mixed media', np.nan)
    X_test['source'] = X_test['source'].replace('Radio', np.nan)
    X_test['source'] = X_test['source'].replace('Card game', 'Game')
    X_test['source'] = X_test['source'].replace('Picture book', 'Book')
    X_test['source'] = X_test['source'].replace('Other', np.nan)
    X_test['source'] = X_test['source'].replace('Web manga', 'Manga')
    X_test['source'] = X_test['source'].replace('4-koma manga', 'Manga')
    X_test['source'] = X_test['source'].replace('Music', np.nan)
    X_test['source'] = X_test['source'].replace('Web novel', 'Book')
    X_test['source'] = X_test['source'].replace('Novel', 'Book')

    source_columns = get_source()
    type_columns = get_type()
    for col in source_columns:
        source_type = col.split('_')[-1]
        X_test[col] = X_test['source'].apply(lambda x: True if isinstance(x, str) and source_type in x else False)

    for col in type_columns:
        type_value = col.split('_')[-1]
        X_test[col] = X_test['type'].apply(lambda x: True if isinstance(x, str) and type_value in x else False)
    
    X_test.drop(columns = ['synopsis', 'source', 'genres', 'studios', 'producers'], inplace = True)
    scaler = preprocessing.MinMaxScaler()
    X_test[["episodes"]] = scaler.fit_transform(X_test[["episodes"]])

    df1 = pd.read_csv('Data/training_rows.csv')
    X_test = X_test.reindex(columns=df1.columns, fill_value=0)

    X_test.drop(columns = ['title', 'anime_id', 'Popularity_category', 'Rank_category', 'score', 'popularity', 'rank', 'studios', 'year'], inplace=True)

    return X_test

def get_image(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(5)  # Allow time for page to load

        # Parse page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find all matching image URLs
        images = soup.find_all("img", src=lambda src: src and src.startswith("https://cdn.myanimelist.net/images/anime"))
        image_urls = [img["src"] for img in images]

        # Print out the list of image URLs
        return image_urls[0]

    finally:
        driver.quit()

anime_id = st.text_input("Enter Anime ID")
st.write("Please Enter One ID Only")

if st.button('Get Predictions', key = 'Predicting'):
    with st.spinner('Scraping...'):
        anime_data = scrape_all(anime_id)

    with st.spinner('Processing...'):
        df_data = process_anime(anime_data)
        X_test = df_data
        X_test = process_features(X_test)

    y_pred = model.predict(X_test)
    y_pred_score = y_pred[0].argmax(axis=1)
    y_pred_score = score_encoder.inverse_transform(y_pred_score)

    y_pred_pop = y_pred[1].argmax(axis=1)
    y_pred_pop = popularity_encoder.inverse_transform(y_pred_pop)

    y_pred_rank = y_pred[2].argmax(axis=1)
    y_pred_rank = rank_encoder.inverse_transform(y_pred_rank)

    url = f"https://myanimelist.net/anime/{anime_id}"
    img = get_image(url)

    st.image(img)
    st.write(f"Title: {df_data.iloc[0]['title']}")
    st.write(f"Alt Name: {df_data.iloc[0]['english']}")
    st.write(f"Description: {df_data.iloc[0]['synopsis']}")
    st.write(f"Genres: {df_data.iloc[0]['genres']}")
    st.write(f"Studios: {df_data.iloc[0]['studios']}")
    st.write(f"Producers: {df_data.iloc[0]['producers']}")
    st.write(f"Episodes: {df_data.iloc[0]['episodes']}")
    st.write(f"Source: {df_data.iloc[0]['source']}")
    st.write(f"Type: {df_data.iloc[0]['type']}")
    st.write(f"Predicted Score: {y_pred_score}")
    st.write(f"Predicted Popularity: {y_pred_pop}")
    st.write(f"Predicted Rank: {y_pred_rank}")
