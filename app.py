import streamlit as st  # type: ignore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")


movies = load_data()


# TF-IDF + KMeans
@st.cache_data
def build_model():
    tfidf = TfidfVectorizer(stop_words="english")
    movies_tfidf = tfidf.fit_transform(movies["genres"])

    kmeans = KMeans(n_clusters=10, random_state=28)
    clusters = kmeans.fit_predict(movies_tfidf)

    return movies_tfidf, clusters


movies_tfidf, clusters = build_model()
movies["cluster"] = clusters


# Recommendation function
def recommend_movie(movie_title, top_n=10):
    idx = movies[movies["title"].str.contains(movie_title, case=False)].index[0]

    movie_cluster = movies.loc[idx, "cluster"]
    cluster_indices = movies[movies["cluster"] == movie_cluster].index

    similarity = cosine_similarity(
        movies_tfidf[idx],  # type: ignore
        movies_tfidf[cluster_indices],  # type: ignore
    ).flatten()

    similar_indices = cluster_indices[similarity.argsort()[-top_n - 1 : -1][::-1]]

    return movies.loc[similar_indices, ["title", "genres"]]


# Streamlit UI

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommender using TF-IDF and KMeans clustering")

movie_input = st.text_input("Enter a movie title")

top_n = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie title.")
    else:
        try:
            recommendations = recommend_movie(movie_input, top_n)
            st.subheader("Recommended Movies")
            st.dataframe(recommendations)
        except IndexError:
            st.error("Movie not found. Please try another title.")
