import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, names=range(24))
movies['title'] = movies[1].astype(str)

# Create combined column (title + genres)
genre_cols = list(range(5, 24))
movies['combined'] = movies['title'] + " " + movies[genre_cols].astype(str).agg(' '.join, axis=1)

# Vectorize
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined'])

# Fit KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(count_matrix)

# Recommendation function
def recommend_knn(movie_name, n_recommendations=5):
    try:
        idx = movies[movies['title'].str.contains(movie_name, case=False)].index[0]
    except:
        return ["Movie not found!"]
    
    distances, indices = knn.kneighbors(count_matrix[idx], n_neighbors=n_recommendations+1)
    recommended_movies = [movies.iloc[i]['title'] for i in indices[0][1:]]
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter a movie name")

if st.button("Get Recommendations"):
    recommendations = recommend_knn(movie_name)
    st.subheader("Recommended Movies:")
    for m in recommendations:
        st.write("✅", m)
