from get_data import get_data
from movie_embeddings import generate_movie_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

#TODO create separate functions for getting data into df and for getting embeddings into variable

def recommendation_engine(movie_title, liked_attributes):

    # get the data in the form of both csv file raw data and 
    # the pickle format preprocessed data stored in data folder
    get_data()

    # Create movie embeddings using BERT tokenizer if they do not exist
    generate_movie_embeddings()


        


recommendation_engine("test,",[12,312,])


    