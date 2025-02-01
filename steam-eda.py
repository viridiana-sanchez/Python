#Import Libraries and Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylot as plt
from datetime import datetime


#Import, read and save datasets to dataframes
#Memory optimization
games_df = pd.read_csv('games.csv', dtype={'app_id': 'int32', 'rating': 'category'})
recommendations_df = pd.read_csv('recommendations.csv', dtype={'app_id': 'int32', 'user_id': 'int32'})
users_df = pd.read_csv('users.csv', dtype={'user_id': 'int32', 'products': 'int32'})


#Perform exploration of all datasets prior to cleaning
#games_df
games_df.head()
games_df.info()
games_df.describe(include='object')
#recommendations_df
recommendations_df.head()
recommendations_df.info()
recommendations_df.describe(include='object')
#users_df
users_df.head()
users_df.info()
users_df.isnull().sum()

#Retrieve total sum of null values for each column within the data.
games_df.isnull().sum()
recommendations_df.isnull().sum()
users_df.isnull().sum()


#Data Cleaning and Transformations: games_df
games_df = (
    games_df.drop(columns="steam_deck", errors="ignore")#drop unnecessary columns
    .drop_duplicates()#drop duplicates
    .assign(
        release_year=lambda x: pd.to_datetime(x['date_release'], errors = 'coerce').dt.year,
        release_month=lambda x: pd.to_datetime(x['date_release'], errors = 'coerce').dt.month,
        rating=lambda x: x['rating'].replace({
            'Overwhelmingly Positive': 'Very Positive',
            'Very Positive': 'Very Positive',
            'Mostly Positive': 'Positive',
            'Mixed': 'Neutral',
            'Mostly Negative': 'Negative',
            'Very Negative': 'Very Negative',
            'Overwhelmingly Negative': 'Very Negative'
        })
    )
)

games_df = games_df.drop(columns=['date_release'])#drop original date_release


#Data Cleaning and Transformations: users_df
#drop duplicate values
users_df = users_df.drop_duplicates()


#Data Cleaning and Transformations: recommendations_df
recommendations_df = (
    recommendations_df.drop(columns="review_id", errors="ignore") #drop unnecessary columns
    .drop_duplicates() #drop duplicates
    .assign(
        rec_year=lambda x: pd.to_datetime(x['date'], errors = 'coerce').dt.year,
        rec_month=lambda x: pd.to_datetime(x['date'], errors = 'coerce').dt.month
    )
)

recommendations_df = recommendations_df.drop(columns= ['date'])#drop original 'date' column

#Rename columns for clarity.
games_df = games_df.rename(columns={'user_reviews': 'game_reviews'})
users_df = users_df.rename(columns={'reviews': 'user_reviews'})

#Save cleaned data to individual csv files
games_df.to_csv('cleaned_games.csv', index=False)
users_df.to_csv('cleaned_users.csv', index=False)
recommendations_df.to_csv('cleaned_recommendations.csv', index=False)