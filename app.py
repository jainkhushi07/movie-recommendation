
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# 1️⃣ Load Data
# ------------------------------
@st.cache_data  # caches data for faster reloads
def load_data():
    # Load MovieLens dataset
    movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, names=range(24))
    movies['title'] = movies[1].astype(str)
    
    # Combine genres into one string
    genre_cols = list(range(5, 24))
    movies['combined'] = movies['title'] + " " + movies[genre_cols].astype(str).agg(' '.join, axis=1)
    
    return movies

movies = load_data()

# ------------------------------
# 2️⃣ Vectorize and Fit KNN
# ------------------------------
@st.cache_resource  # caches model for faster reloads
def fit_knn(movies):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(movies['combined'])
    
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(count_matrix)
    
    return cv, count_matrix, knn

cv, count_matrix, knn = fit_knn(movies)

# ------------------------------
# 3️⃣ Recommendation Function
# ------------------------------
def recommend_knn_with_scores(movie_name, n_recommendations=5):
    """
    Recommend movies similar to the given movie using KNN on content features.
    
    Returns a list of tuples: [(movie_title, similarity_score), ...]
    """
    # Find movie index
    matches = movies[movies['title'].str.contains(movie_name, case=False)]
    
    if len(matches) == 0:
        return [("Movie not found!", 0)]
    
    idx = matches.index[0]
    
    # Get nearest neighbors
    distances, indices = knn.kneighbors(count_matrix[idx], n_neighbors=n_recommendations+1)
    
    # Build recommendation list with similarity scores
    recommended_movies = []
    for i, dist in zip(indices[0][1:], distances[0][1:]):  # skip first (same movie)
        similarity_score = 1 - dist
        recommended_movies.append((movies.iloc[i]['title'], similarity_score))
    
    return recommended_movies

# ------------------------------
# 4️⃣ Streamlit Frontend
# ------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get similar movie recommendations!")

# Input: Movie name
movie_input = st.text_input("Enter a movie name:")

# Input: Number of recommendations
num_rec = st.slider("Number of recommendations:", 1, 10, 5)

# Recommendation output
if movie_input:
    recommendations = recommend_knn_with_scores(movie_input, num_rec)
    
    if recommendations[0][0] == "Movie not found!":
        st.error("Movie not found. Please check spelling or try another movie.")
    else:
        st.subheader("Recommended Movies:")
        for rec, score in recommendations:
            st.write(f"✅ {rec} — Similarity Score: {score:.2f}")
