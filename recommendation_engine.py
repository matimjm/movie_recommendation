from get_data import get_data
from movie_embeddings import generate_movie_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import joblib
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import numpy as np
import pandas as pd

#TODO create separate functions for getting data into df and for getting embeddings into variable

def get_data_and_embeddings():

    # get the data in the form of both csv file raw data and 
    # the pickle format preprocessed data stored in data folder
    get_data()

    # getting the df from pickle to df


    # Create movie embeddings using BERT tokenizer if they do not exist
    generate_movie_embeddings()

def get_combined_features_and_model(df, embeddings):
    """
    Combines features and builds the NearestNeighbors model as a one-time process.
    """
    nn_model_path = Path("data/nn_model.pkl")
    combined_features_path = Path("data/combined_features.npy")

    if nn_model_path.exists() and combined_features_path.exists():
        print("Loading pre-built NN model and combined features.")
        nn_model = joblib.load(nn_model_path)
        combined_features = np.load(combined_features_path)
        return combined_features, nn_model

    print("NN model or combined features not found. Building them now...")
    
    # Exclude the id and text_soup columns for a clean numerical matrix
    numerical_features_df = df.drop(columns=['id', 'text_soup'])
    numerical_features_np = numerical_features_df.to_numpy()
    
    # Concatenate the BERT embeddings and the numerical features
    combined_features = np.hstack((embeddings, numerical_features_np))

    # Build the NearestNeighbors model and save it
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
    nn_model.fit(combined_features)
    joblib.dump(nn_model, nn_model_path)
    np.save(combined_features_path, combined_features)

    print("NN model and combined features saved.")
    return combined_features, nn_model

# --- The Main Recommendation Function ---

def find_personalized_recommendations(movie_id, liked_attributes, df, embeddings, combined_features, nn_model, top_n=5, weight=2.0):
    """
    Finds personalized recommendations based on what the user liked about a movie.
    """
    if movie_id not in df['id'].values:
        print(f"Movie with ID '{movie_id}' not found in the dataset.")
        return pd.DataFrame() # Return an empty DataFrame if the movie is not found

    movie_index = df[df['id'] == movie_id].index[0]
    
    # Get the feature vector for the selected movie from the combined features matrix
    movie_vector = combined_features[movie_index].reshape(1, -1)
    
    # This is the query vector that will be used to find similar movies
    query_vector = np.copy(movie_vector)
    
    # Find the indices of the numerical features that match the liked attributes
    numerical_cols = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity', 'adult', 'release_year']
    
    # Boost the liked attributes directly in the query vector
    for attr in liked_attributes:
        if attr in numerical_cols:
            attr_index_in_combined = embeddings.shape[1] + numerical_cols.index(attr)
            query_vector[0, attr_index_in_combined] *= weight
        else:
            # For non-numerical attributes like 'genres' or 'keywords', you can boost the entire text embedding section
            query_vector[0, :embeddings.shape[1]] *= weight
            
    # Use the NearestNeighbors model for an efficient search
    distances, indices = nn_model.kneighbors(query_vector, n_neighbors=top_n + 1)
    
    # Get the recommended movie IDs, skipping the first one (the movie itself)
    recommended_indices = indices.flatten()[1:]
    
    return df.iloc[recommended_indices]

def recommendation_engine(movie_id, liked_attributes):
    
    # Downloading the dataset and embeddings if it was not done already
    get_data_and_embeddings()

    # putting the data and embeddings from file into dataframe and numpy ndarray
    df_cleaned_20_percent = pd.read_pickle("data/random_20_percent_cleaned_dataframe.pkl").reset_index()
    df_raw = pd.read_pickle("data/raw_dataframe.pkl")
    movie_embeddings = np.load("data/random_20_percent_movie_embeddings.npy")
    print(df_cleaned_20_percent.head())
    #print(df_raw.shape[0])
    print(movie_embeddings.shape)
    
    combined_features_matrix, nn_model = get_combined_features_and_model(df_cleaned_20_percent, movie_embeddings)
        
    recommended_movies = find_personalized_recommendations(
            movie_id, 
            liked_attributes, 
            df_cleaned_20_percent, 
            movie_embeddings, 
            combined_features_matrix, 
            nn_model
    )

    list_of_ids = recommended_movies['id'].tolist()
    
    if recommended_movies is not None and not recommended_movies.empty:
        print(f"Top {len(recommended_movies)} personalized recommendations for movie ID {df_raw.loc[df_raw['id']==movie_id]['title']}:")
        print(df_raw[df_raw['id'].isin(list_of_ids)]["title"])


selected_movie_id = 222707
user_liked_attributes = ["Parti sans laisser d'adresse", "how the British brok the Braz"]

recommendation_engine(selected_movie_id, user_liked_attributes)


    