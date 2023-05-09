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

def load_movies():
    movies = pd.read_csv('movies.csv')
    return movies
st.set_page_config(page_title="Login Page")

if st.session_state["UserID"] == "" and st.session_state["UserName"] == "":
    st.error("Login First!!")
else:    
    username = str(st.session_state["UserName"])
    user_id = int(st.session_state["UserID"])
    # print(type(username))
    # print(type(user_id))
    movies = load_movies()
    ratings = pd.read_csv('ratings.csv')

    st.write(f"Welcome, {username} who's UserID: {user_id}!")
    
    # Load login.csv file and create a dictionary to map usernames to new_user_ids
    login = pd.read_csv('login.csv')
    st.sidebar.write("Sidebar")
    res=st.sidebar.radio("Select Any Movie Recommandation System",options=("Content Based Movie Prediction","Collaborative Movie Prediction"))
    st.header("MOVIE RECOMMADATION SYSTEM")

    if(res=="Content Based Movie Prediction"):
        # movielist=pickle.load(open('movie1Dict.pkl','rb'))
        # movie=pd.DataFrame(movielist)
        # st.selectbox("Choose your Favorite Movie: ",movie["title"].values)
        def fetch_poster(movie_id):
            url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
            data = requests.get(url)
            data = data.json()
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path

        def recommend(movie):
            index = movies[movies['title'] == movie].index[0]
            distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
            recommended_movie_names = []
            recommended_movie_posters = []
            for i in distances[1:6]:
                # fetch the movie poster
                movie_id = movies.iloc[i[0]].movie_id
                recommended_movie_posters.append(fetch_poster(movie_id))
                recommended_movie_names.append(movies.iloc[i[0]].title)

            return recommended_movie_names,recommended_movie_posters


        # st.header('Movie Recommender System')
        movie_dict=pickle.load(open('movie_list.pkl','rb'))
        movies=pd.DataFrame(movie_dict)

        similarity=pickle.load(open('sim.pkl','rb'))
        movies=pd.DataFrame(movie_dict)

        movie_list = movies['title'].values
        selected_movie = st.selectbox(
            "Type or select a movie from the dropdown",
            movies["title"].values
        )

        if st.button('Show Recommendation'):
            recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])


    elif(res=="Collaborative Movie Prediction"):
        st.text("Get movie prediction based on your rated hisory")
        kk=user_id
        
        pUser=kk
        if st.button("Predict"):
            df = pd.read_csv("ratings.csv")
            user_ids = df["userId"].unique().tolist()
            user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            userencoded2user = {i: x for i, x in enumerate(user_ids)}
            movie_ids = df["movieId"].unique().tolist()
            movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
            movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
            df["user"] = df["userId"].map(user2user_encoded)
            df["movie"] = df["movieId"].map(movie2movie_encoded)
            num_users = len(user2user_encoded)
            num_movies = len(movie_encoded2movie)
            # min and max ratings will be used to normalize the ratings later
            min_rating = min(df["rating"])
            max_rating = max(df["rating"])
            # cast the ratings to float32
            df["rating"] = df["rating"].values.astype(np.float32)

            df = df.sample(frac=1, random_state=42)

            x = df[["user", "movie"]].values
            ## print(type(x))

            # Normalize the targets between 0 and 1. Makes it easy to train.
            y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

            ## print(type(y))

            # Assuming training on 90% of the data and validating on 10%.
            # might change this to 99/1
            train_indices = int(0.99 * df.shape[0])

            x_train, x_val, y_train, y_val = (
                x[:train_indices],
                x[train_indices:],
                y[:train_indices],
                y[train_indices:],
            )

            EMBEDDING_SIZE = 50

            class RecommenderNet(keras.Model):
                def __init__(self, num_users, num_movies, embedding_size, **kwargs):
                    super(RecommenderNet, self).__init__(**kwargs)
                    self.num_users = num_users
                    self.num_movies = num_movies
                    self.embedding_size = embedding_size
                    self.user_embedding = layers.Embedding(
                        num_users,
                        embedding_size,
                        embeddings_initializer="he_normal",
                        embeddings_regularizer=keras.regularizers.l2(1e-6),
                    )
                    self.user_bias = layers.Embedding(num_users, 1)
                    self.movie_embedding = layers.Embedding(
                        num_movies,
                        embedding_size,
                        embeddings_initializer="he_normal",
                        embeddings_regularizer=keras.regularizers.l2(1e-6),
                    )
                    self.movie_bias = layers.Embedding(num_movies, 1)

                def call(self, inputs):
                    user_vector = self.user_embedding(inputs[:, 0])
                    user_bias = self.user_bias(inputs[:, 0])
                    movie_vector = self.movie_embedding(inputs[:, 1])
                    movie_bias = self.movie_bias(inputs[:, 1])
                    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
                    # Add all the components (including bias)
                    x = dot_user_movie + user_bias + movie_bias
                    # The sigmoid activation forces the rating to between 0 and 1
                    return tf.nn.sigmoid(x)

            model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)

            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
            )

            history = model.fit(
                x=x_train,
                y=y_train,
                batch_size=32,
                epochs=1,
                verbose=1,
                validation_data=(x_val, y_val),
            )
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            movie_df = pd.read_csv("movies.csv")

            # Let us get a user and see the top recommendations.

            # Pick a user at random.
            user_id = pUser  

            # Get all movies watched by the user.
            movies_watched_by_user = df[df.userId == user_id] 

            # Get the movies not watched by the user.
            movies_not_watched = movie_df[
                ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
            ]["movieId"]

            movies_not_watched = list(
                set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
            )
            movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

            user_encoder = user2user_encoded.get(user_id)

            user_movie_array = np.hstack(
                ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
            )

            ratings = model.predict(user_movie_array).flatten()

            top_ratings_indices = ratings.argsort()[-10:][::-1]

            recommended_movie_ids = [
                movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
            ]

            print("Showing recommendations for user: {}".format(user_id))
            st.subheader("Showing recommendations for user:")
            # st.header(f"",pUser)
            st.text("{}".format(pUser))
            print("====" * 9)
            # st.write("====" *9 )
            print("Movies with high ratings from user")
            st.subheader("Movies with high ratings from user")
            print("----" * 8)
            st.write("----" * 8)

            top_movies_user = (
                movies_watched_by_user.sort_values(by="rating", ascending=False)
                .head(5)
                .movieId.values
            )
            movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
            for row in movie_df_rows.itertuples():
                print(row.title, ":", row.genres)
                st.write(row.title, ":", row.genres)

            print("\n")
            print("----" * 8)
            st.write("----" * 8)
            print("Top 10 movie recommendations")
            st.subheader("Top 10 movie recommendations")
            print("----" * 8)
            st.write("----" * 8)
            recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
            for row in recommended_movies.itertuples():
                print(row.title, ":", row.genres)
                st.write(row.title, ":", row.genres)






