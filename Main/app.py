from matplotlib.pyplot import title
import streamlit as st
import pickle
import pandas as pd
import requests

# TMDB image path - https://image.tmdb.org/t/p/w500/

def fetch_poster(movie_id) :
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=a049224e373ecfb95170419e7176e7fc&language=en-US')
    data = response.json()  #covert this to json file

    return "https://image.tmdb.org/t/p/w500/" + data['poster_path'] 


movie_dict = pickle.load(open('movies_dict.pkl', 'rb')) # since we were unable to pass a dataframe from jupiter via pickle we passed a dictionary and loaded it
movies = pd.DataFrame(movie_dict) # ocnvert it to pandas dataframe

similarity = pickle.load(open('Similarity.pkl', 'rb')) # this is the similarity index taht we had passed and opened it in reading mode

def recommend(movie):  # similar finction as that we made in jupiter file
    movie_index = movies[movies['title'] == movie].index[0]
    # print(movie_index)
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]  
    
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]]['id'] #this is requeired for poster
        # movie_id = i[0]
        recommended_movies.append(movies.iloc[i[0]]['title'])
        # fetch poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters

st.title('Movie Recommender System')
Selected_movie_name = st.selectbox(
'ENTER THE MVOIE',
movies['title'].values)

if st.button('Recomend'):
    names, posters = recommend(Selected_movie_name)
    # for i in recommendations:
    #     st.write(i)
    col0, col1, col2, col3, col4 = st.columns(5)

    with col0:
        st.text(names[0])
        st.image(posters[0])

    with col1:
        st.text(names[1])
        st.image(posters[1])

    with col2:
        st.text(names[2])
        st.image(posters[2])

    with col3:
        st.text(names[3])
        st.image(posters[3])

    with col4:
        st.text(names[4])
        st.image(posters[4])

# to deploy the streamlit file on heroku we need 
# procfile, setup file, requirements file, .gitignore file
# last one is fetched by "pip freeze > reirements.txt"