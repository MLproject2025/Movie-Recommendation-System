
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                          sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                         sep='|', encoding='latin-1',
                         names=['movieId', 'title', 'release_date', 'video_release_date',
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
                                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                'War', 'Western'])

    data = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
    return data, movies

def compute_models(data, movies):
    user_movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
    user_sim_matrix = cosine_similarity(user_movie_matrix.fillna(0))
    user_sim_df = pd.DataFrame(user_sim_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    movie_titles = movies[['movieId', 'title']].copy()
    movie_titles['title_cleaned'] = movie_titles['title'].str.lower()
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_titles['title_cleaned'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    return user_movie_matrix, user_sim_df, movie_titles, cosine_sim

def hybrid_recommendations(user_id, data, movies, user_sim_df, user_movie_matrix, movie_titles, cosine_sim, top_n=10, cf_weight=0.6):
    rated_movies = data[data['userId'] == user_id]['movieId'].tolist()
    unrated_movies = movies[~movies['movieId'].isin(rated_movies)]

    sim_users = user_sim_df[user_id]
    scores = []

    for movie_id in unrated_movies['movieId']:
        try:
            movie_ratings = user_movie_matrix[movie_id]
            weighted_ratings = sim_users * movie_ratings
            cf_score = weighted_ratings.sum() / sim_users[movie_ratings.notnull()].sum()
        except:
            cf_score = 2.5

        sim_scores = []
        for rated_id in rated_movies:
            try:
                idx1 = movie_titles.index[movie_titles['movieId'] == movie_id][0]
                idx2 = movie_titles.index[movie_titles['movieId'] == rated_id][0]
                sim_scores.append(cosine_sim[idx1, idx2])
            except:
                continue
        cb_score = np.mean(sim_scores) if sim_scores else 0

        hybrid_score = cf_weight * cf_score + (1 - cf_weight) * cb_score
        scores.append((movie_id, hybrid_score))

    top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    result = movies[movies['movieId'].isin([m[0] for m in top_movies])][['title']]
    result['Score'] = [m[1] for m in top_movies]
    return result

# Streamlit Interface
st.title("🎬 CineMatch: Hybrid Movie Recommendation System")

data, movies = load_data()
user_movie_matrix, user_sim_df, movie_titles, cosine_sim = compute_models(data, movies)

user_id = st.slider("Select User ID", min_value=1, max_value=943, value=10)
top_n = st.slider("Number of Recommendations", 5, 20, 10)
cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6)

recommendations = hybrid_recommendations(user_id, data, movies, user_sim_df, user_movie_matrix, movie_titles, cosine_sim, top_n, cf_weight)

st.write(f"Top {top_n} movie recommendations for User {user_id}:")
st.dataframe(recommendations)
