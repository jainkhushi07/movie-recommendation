import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Page Config & Hide Footer/Menu
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}  /* hides hamburger menu */
    footer {visibility: hidden;}    /* hides footer with 'Made with Streamlit' or GitHub info */
    header {visibility: hidden;}    /* optional: hides header */
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
movies = pd.read_csv('u.item', sep='|', encoding='latin-_
