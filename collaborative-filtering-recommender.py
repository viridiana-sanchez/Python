## Install Libraries
import os
import json
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import botocore.exceptions


from io import StringIO
from pandas import DataFrame
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

### Data Preparation
#Handle data types when loading file
movies_metadata_df = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv', dtype={'id': 'str', 'popularity': 'str'})
ratings_df = pd.read_csv("/kaggle/input/the-movies-dataset/ratings.csv")
links_df = pd.read_csv('/kaggle/input/the-movies-dataset/links.csv')

#Display the first few rows of each DataFrame
print("Ratings DataFrame:")
print(ratings_df.head())

print("\nLinks DataFrame:")
print(links_df.head())

print("\nMovies Metadata DataFrame:")
print(movies_metadata_df.head())

##Merge datasets into one dataframe
#Merge 'ratings_df' and 'links_df' based on the "movieId" column
ratings_links_df = pd.merge(ratings_df, links_df, on='movieId', how='inner')

#Convert 'imdbId' in the merged DataFrame to object type
ratings_links_df['imdbId'] = ratings_links_df['imdbId'].astype(str)

#Merge based on the "imdbId" and "id" columns
df = pd.merge(ratings_links_df, movies_metadata_df, left_on='imdbId', right_on='id', how='inner')

#Display the resulting DataFrame
print("Merged DataFrame:")
print(df.head())
df.shape
df.describe(include='object')
df.info()

### Data Cleaning and Transformation
#Drop irrelevant columns
#drop irrelevant columns
df = df.drop(['tmdbId', 'homepage', 'belongs_to_collection', 'poster_path', 'original_title', 'overview', 'status', 'tagline', 'video'], axis=1)

#Check which columns have nulls
df.isnull().sum()

#Handle missing values
df['original_language'] = df['original_language'].fillna(df['original_language'].mode()[0])
df = df.dropna(subset=['release_date'])
df['runtime'] = df['runtime'].fillna(df['runtime'].median())

#Convert Data Types
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year
# Check the data types
print(df.dtypes)

#Check for remaining missing values
print(df.isnull().sum())

# Convert 'popularity' to float in the 'data' DataFrame
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

# Select relevant features
selected_features = df[['userId', 'movieId', 'rating', 'title', 'vote_average', 'popularity']]

# Display the updated DataFrame
print(selected_features.head())

# Drop columns not in the list
final_data = selected_features.copy()

# Specify the file path where you want to save the CSV file
csv_file_path = '/kaggle/working/final_data.csv'

# Save the DataFrame to CSV
final_data.to_csv(csv_file_path, index=False)

# Print a message indicating the successful save
print(f"DataFrame saved to {csv_file_path}")

final_data.info()

# Load data using Surprise
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(final_data[['userId', 'movieId', 'rating']], reader)