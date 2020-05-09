import json
import string
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import flatten

json_file = "yummly.json"

input_list = ['paprika', 'banana','rice krispies','plain flour', 'ground pepper', 'salt', 'tomatoes']
Input_matrix = []


def Reading_jsondata(json_file ):
    with open(json_file) as file:
        Temp_data = json.load(file)
    data_frame = pd.DataFrame(Temp_data)
    return  data_frame

def Text_Normalization(data_frame,input_list):
    # print(input_list)
    list_for_matrix = []
    for i in range(len(data_frame['ingredients'])):
        text = " ".join(data_frame['ingredients'][i])
        text = text.lower()
        text = text.strip()
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub("r(\d)","", text)
        text = re.sub(r'\(.*?\)', '', text)
        tokens = word_tokenize(text)
        list_for_matrix.append(" ".join(tokens))
    input_list = " ".join(input_list)
    list_for_matrix.insert(0,input_list)
    return list_for_matrix

def Vectorizer (Normalized_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(Normalized_data)
    Input_matrix = matrix[0]
    new_matrix = matrix[1:]
    return new_matrix,Input_matrix

def Finging_Kneighbors(X,dataframe):
    LabelEncoder = preprocessing.LabelEncoder()
    LabelEncoder.fit(dataframe['cuisine'])
    Y = LabelEncoder.transform(dataframe['cuisine'])
    score_list = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
    for neighbors in range(25):
        K_neighbors = neighbors + 1
        KNN_model = KNeighborsClassifier(n_neighbors=K_neighbors, weights='uniform', algorithm="auto")
        KNN_model.fit(X_train, Y_train)
        Y_prediction = KNN_model.predict(X_test)
        score = metrics.accuracy_score(Y_test, Y_prediction)
        score_list.append (score)
        print("Accuracy score  ", score, "% for K Neighbors-Value:", K_neighbors)
    # print(score_list)
    Kneighbors = score_list.index(max(score_list))
    print("The number of neighbors with good accuracy score is :", Kneighbors+1 )
    return Kneighbors

def KNN_Classification_Model(X,dataframe,Input_matrix,Kneighbors):
    LabelEncoder = preprocessing.LabelEncoder()
    LabelEncoder.fit(dataframe['cuisine'])
    Y = LabelEncoder.transform(dataframe['cuisine'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
    KNN_model = KNeighborsClassifier(n_neighbors=Kneighbors, weights='uniform', algorithm="auto")
    KNN_model.fit(X_train, Y_train)
    Y_prediction = KNN_model.predict(Input_matrix)
    predicted_cuisine = LabelEncoder.inverse_transform(Y_prediction)
    cosine_score = cosine_similarity(Input_matrix, X)
    Cuisine_score = cosine_score.tolist()
    Cuisine_score= flatten(Cuisine_score)
    print("predicted_cuisine :",predicted_cuisine )
    dataframe['Cuisine_score'] = Cuisine_score
    # print( dataframe['Cuisine_score'])
    DataFrame = dataframe.sort_values(by ='Cuisine_score',ascending=False)
    Nearest_cuisine = DataFrame[['id', 'Cuisine_score']].head(Kneighbors)
    print("The closest 5 Recipie")
    print(Nearest_cuisine)

    return 0

dataframe = (Reading_jsondata(json_file))
Normalized_data = Text_Normalization(dataframe,input_list)
X = Vectorizer (Normalized_data)[0]
Input_matrix = Vectorizer (Normalized_data)[1]
Kneighbors = Finging_Kneighbors(X,dataframe)
KNN_Classification_Model(X,dataframe,Input_matrix,Kneighbors)




