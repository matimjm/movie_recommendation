import kagglehub
import csv
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # download the data from the internet
    # Get the path to the parent directory (MOVIE_RECOMMENDATION_PROJECT)
    data_path = Path(__file__).parent / "data" / "TMDB_movie_dataset_v11.csv"
    

    if data_path.is_file():
        print(f"{data_path} already exists")
    else:
        print(f"Did not find {data_path} directory, creating one...")
        # Download latest version
        kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    
    data_path = Path(__file__).parent / "data" / "cleaned_dataframe.pkl"


    if data_path.is_file():
        print(f"{data_path} already exists")
    else:
        
        # Convert the CSV file to Pandas dataframe
        list_of_data = [] # create empty list of data

        # import csv directly into pandas dataframe
        df = pd.read_csv("data/TMDB_movie_dataset_v11.csv")

        

        # Replace missing release_date values with average date


        # convert the release_date string into a datetime object
        # coerce converts empty dates to NaT (not a time)

        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        average_date = df['release_date'].mean() # calculate the average date

        df['release_date'].fillna(average_date, inplace=True) # fill all missing release date with average date values

        df.fillna('',inplace=True) # replace the missing string values with empty string (there are no missing numeric values)
        
        # Saving the raw dataframe for further accessing when displaying recommendations
        df.to_pickle("data/raw_dataframe.pkl")

        # create a text soup for better analysis by the model
        df['text_soup'] = df['title'] + ' ' + \
                    df['overview'] + ' ' +\
                    df['tagline'] + ' ' +\
                    df['genres'] + ' ' +\
                    df['keywords'] + ' '+\
                    df['status'] + ' ' + \
                    df['original_language'] + ' ' + \
                    df['original_title'] + ' ' + \
                    df['production_companies'] + ' ' + \
                    df['production_countries'] + ' ' + \
                    df['spoken_languages']


        # normalizing numerical values

        # first convert release_date into a year only as it is the most important data for movies release
        
        df['release_year'] = df['release_date'].dt.year
        

        # Scaling the numerical values so that they are between 0 and 1
        
        numerical_cols = ['vote_average', 'vote_count', 'revenue', 'release_year', 'runtime', 'budget', 'popularity']
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # converting the adult from boolean into 0 or 1
        df['adult'] = df['adult'].astype(int)

        
        
        # dropping the unnecessary columns
        df.drop('imdb_id', axis=1, inplace=True)
        df.drop('backdrop_path', axis=1, inplace=True)
        df.drop('poster_path', axis=1, inplace=True)
        df.drop('homepage', axis=1, inplace=True)
        df.drop('release_date', axis=1, inplace=True)

        # deleting the columns included in the text soup
        columns_to_drop = ['overview', 'tagline', 'genres', 
                        'keywords', 'title', 'status', 
                        'original_language', 'original_title', 
                        'production_companies', 'production_countries',
                        'spoken_languages']
        df.drop(columns=columns_to_drop, inplace=True)

        # Save pre-processed df into pickle format
        df.to_pickle("data/cleaned_dataframe.pkl")

        print(df.head())








    

    





    
    


