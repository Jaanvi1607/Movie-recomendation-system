import numpy as np
import pandas as pd
import sklearn 
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies.head(1)
credits.head(1)
movies = movies.merge(credits,on='title')
movies.head(1)
movies['genres'].value_counts()
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.info()
movies.head(1)
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
#iloc describes a function from pandas library to retrieve data based on index
movies.iloc[0].genres
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
#here this is string list so convert function doesnt work so we have to convert string into list then do convert for that we have to import a library called ast
import ast
movies['genres'] = movies['genres'].apply(convert)
#then we'll do the same thing on keywords column
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
movies['genres'].info()
movies['cast'][0]
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
    return L
movies['cast'] = movies['cast'].apply(convert3)
movies.head(1)#see here we get only three names
movies['crew'][0]
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director': 
            L.append(i['name'])
            break     
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()
movies['overview'][0]
movies['overview'] = movies['overview'].apply(lambda x:x.split())
#here we are applying lambda function to split the strings into lists
movies.head()
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head(1)
movies.tags[0]
new_df = movies[['movie_id','title','tags']]
new_df
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head()
new_df['tags'][0]
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head(1)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors
cv.get_feature_names_out()
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
ps.stem('loving')
ps.stem('loves')
ps.stem('loved')
ps.stem('in the 22nd century, a paraplegic marine is di...')
new_df['tags'] = new_df['tags'].apply(stem)
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    
recommend('Autumn in New York')
new_df.iloc[1216].title
import pickle
pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))
new_df['title'].values
new_df.to_dict()
pickle.dump(similarity,open('similarity.pkl','wb'))

