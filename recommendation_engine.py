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
from timeit import default_timer as timer

#TODO create separate functions for getting data into df and for getting embeddings into variable

def get_data_and_embeddings():

    # get the data in the form of both csv file raw data and 
    # the pickle format preprocessed data stored in data folder
    get_data()

    # getting the df from pickle to df


    # Create movie embeddings using BERT tokenizer if they do not exist
    generate_movie_embeddings()

# --- The Main Recommendation Function ---

def find_personalized_recommendations(liked_attributes, movie_embeddings):
    """
    Finds personalized recommendations based on what the user liked about a movie.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    
    tokenized_text = tokenizer(
                liked_attributes, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
    # 3. Move this small batch to the GPU
    tokenized_text = tokenized_text.to(device)

    # 4. Generate embeddings for this batch
    with torch.no_grad():
        outputs = model(**tokenized_text)
    
    # 5. Get the embeddings, move to CPU, and add to our list
    user_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    
    scores = cosine_similarity(user_embeddings, movie_embeddings)

    top_indices = np.argsort(scores[0])[::-1][:3]
    
    return top_indices


    

def recommendation_engine(liked_attributes):

    print(f"Creating recommendations for: {liked_attributes}")
    
    start_time = timer()
    # Downloading the dataset and embeddings if it was not done already
    get_data_and_embeddings()

    # putting the data and embeddings from file into dataframe and numpy ndarray
    cleaned_dataframe = pd.read_pickle("data/cleaned_dataframe.pkl").reset_index()

    df_raw = pd.read_pickle("data/raw_dataframe.pkl")
    movie_embeddings = np.load("data/movie_embeddings.npy")
    print(cleaned_dataframe.shape)
    #print(df_raw.shape[0])
    print(movie_embeddings.shape)
        
    recommended_movies_index = find_personalized_recommendations(
            liked_attributes, 
            movie_embeddings
    )

    list_titles = [df_raw.loc[i]["title"] for i in recommended_movies_index]
    
    if list_titles is not None:
        print(f"Top {len(list_titles)} personalized recommendations")
        print(list_titles)

    rec_time = round(timer()-start_time,5)

    print(f"The movies were recommended in {rec_time} s.")
    return list_titles, rec_time



    