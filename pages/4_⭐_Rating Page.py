import streamlit as st
import subprocess
import csv
from streamlit import cache
import os


from datetime import datetime
# import streamlit as st
import argparse              
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from ipaddress import summarize_address_range
# import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import keras.optimizers
import keras.regularizers
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import requests
from typing import List
# @st.cache(suppress_st_warning=True, allow_output_mutation=True)


def load_movies():
    movies = pd.read_csv('movies.csv')
    return movies
st.set_page_config(page_title="Login Page")

if st.session_state["UserID"] == "" and st.session_state["UserName"] == "":
    st.error("Login First!!")
else:    
    username = str(st.session_state["UserName"])
    user_id = int(st.session_state["UserID"])
    # print("Hi else")
    

    # Open the login.csv file
    with open('login.csv', encoding="cp437", errors='ignore') as login_file:
        csv_reader = csv.reader(login_file)
        found_match = False
    movies = load_movies()
    ratings = pd.read_csv('ratings.csv')

    # Load login.csv file and create a dictionary to map usernames to new_user_ids
    login = pd.read_csv('login.csv')

    # Create a function to get the movieId based on the movie title

    def find_movie_id_by_title(title):
        with open('movies.csv', 'r',encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) # skip header row
            for row in reader:
                if row[1] == title:
                    return row[0]
        return None

    def find_movie_id_by_genres(title):
        with open('movies.csv', 'r',encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) # skip header row
            for row in reader:
                if row[1] == title:
                    return row[2]
        return None

    def find_user_rating(user_id, movie_id):
        user_ratings = ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)]
        if len(user_ratings) > 0:
            return user_ratings.index[0]
        else:
            return None


    # st.set_page_config(page_title="Rate a Movie", page_icon=":movie_camera:", layout="wide", initial_sidebar_state="expanded")

    # Retrieve the username and user ID arguments

    username = str(username)
    user_id = int(user_id)

    # Display the username and user ID
    st.write(f"Welcome, {username} who's UserID: {user_id}!")


    # Create the streamlit app
    st.title('Rate a Movie')

    # Get the movie title from the user
    # Get the new_user_id and movieId based on the inputs

    movie_title = st.selectbox("Movie Title", movies['title'].tolist())
    movie_id=find_movie_id_by_title(movie_title)
    movie_genres=find_movie_id_by_genres(movie_title)
    timestamp = str(int(datetime.now().timestamp()))
    # st.text(movie_title+"'s movie_id is "+movie_id+" of genres "+movie_genres+timestamp)
    row_index=find_user_rating(int(user_id),int(movie_id))
    if row_index:
        prevoius_rating=ratings.iloc[row_index].loc['rating']
    else:
        prevoius_rating=None
    st.subheader("Movie Details")    
    st.info(f"Movie Title: {movie_title}")
    st.info(f"Movie ID: {movie_id}")
    st.info(f"Movie Genres: {movie_genres}")
    st.info(f"Movie Rating:{prevoius_rating}")

    # Get the rating from the user
    rating = st.slider('Rating', 1.0, 5.0, step=0.5)
    col11, col22 = st.columns(2)

    rateBtn = col11.button("Rate Now")
    unrateBtn = col22.button("Unrate Now")

    st.text("Row Index is "+str(row_index))
    if rateBtn:
        if row_index is not None:
            # If the user has already rated this movie, update the rating in the ratings DataFrame
            ratings.at[row_index, 'rating'] = rating
            st.success(f"Rating has updated")
        else:
            # If the user has not yet rated this movie, add a new row to the ratings DataFrame
            new_row = {'userId': user_id, 'movieId': movie_id, 'rating': rating, 'timestamp': int(datetime.now().timestamp())}
            ratings = ratings.append(new_row, ignore_index=True)
            st.success(f"New rating has added ")

        # Write the updated ratings DataFrame to the ratings.csv file
        ratings.to_csv('ratings.csv', index=False)


    if unrateBtn:
        # If the "Unrate" button was clicked, delete the rating from the ratings DataFrame
        if row_index is not None:
            ratings.drop(row_index, inplace=True)
            st.success(f"Rating has deleted")
        else:
            st.warning("No rating found to delete.")
        
        # Write the updated ratings DataFrame to the ratings.csv file
        ratings.to_csv('ratings.csv', index=False)    
    