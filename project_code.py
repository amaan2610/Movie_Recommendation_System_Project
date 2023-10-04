import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to apandas dataframe

df = pd.read_csv(r'E:\Movie Recommendation System\movies.csv')
print(df.head())

# number of rows and columns in the data frame

df.shape

# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null valuess with null string

for feature in selected_features:
  df[feature] = df[feature].fillna('')

  # combining all the 5 selected features

combined_features = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']

print(combined_features)

# converting the text data to feature vectors

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

print(feature_vectors)


#   Cosine Similarity

# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)

print(similarity)

print(similarity.shape)


#   Getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')

# creating a list with all the movie names given in the dataset

list_of_all_titles = df['title'].tolist()
print(list_of_all_titles)

# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

# finding the index of the movie with title

index_of_the_movie = df[df.title == close_match]['index'].values[0]
print(index_of_the_movie)

# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)

# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = df[df.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

    #  Movie Recommendation System

movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = df['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = df[df.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = df[df.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1